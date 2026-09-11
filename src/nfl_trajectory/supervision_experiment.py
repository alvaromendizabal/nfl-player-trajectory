"""Matched real-data training/evaluation with fail-closed epoch checkpoint publication."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from nfl_trajectory.benchmark import error_metrics
from nfl_trajectory.motion import KEYS
from nfl_trajectory.motion_supervision import MatchedState, supervised_batch
from nfl_trajectory.supervision_batches import TrainingBatches, scheduled_learning_rate
from nfl_trajectory.supervision_evidence import load_generation, save_generation

CheckpointPublisher = Callable[[Path, dict[str, Any]], None]
ProgressCallback = Callable[[dict[str, float | int]], None]


@dataclass(frozen=True)
class TrainingSettings:
    """Exposure frozen from training-only throughput before validation scoring."""

    epochs: int
    batch_plays: int
    warmup_steps: int
    seed: int = 2026
    width: int = 96
    peak_learning_rate: float = 0.001
    floor_learning_rate: float = 0.00001

    def total_steps(self, training_plays: int) -> int:
        batches = math.ceil(training_plays / self.batch_plays)
        return self.epochs * batches

    def validate(self, training_plays: int) -> None:
        total = self.total_steps(training_plays)
        if self.epochs < 1 or self.batch_plays < 1 or self.width < 8:
            raise ValueError("Positive exposure and a valid model width are required.")
        if not 1 <= self.warmup_steps < total - 1:
            raise ValueError("Warmup must be inside the frozen training exposure.")
        if not 0 < self.floor_learning_rate <= self.peak_learning_rate:
            raise ValueError("Learning-rate endpoints are invalid.")


def experiment_signature(
    source_signature: str,
    sample_sha256: str,
    config_sha256: str,
    arm: str,
) -> str:
    """Bind checkpoints to source, private input, frozen config and one named arm."""
    import hashlib
    import json
    import re

    values = (source_signature, sample_sha256, config_sha256)
    if arm not in {"coordinate", "velocity"}:
        raise ValueError("Unknown supervision arm.")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
        raise ValueError("Experiment provenance requires full SHA256 values.")
    payload = {
        "source_signature": source_signature,
        "sample_sha256": sample_sha256,
        "config_sha256": config_sha256,
        "arm": arm,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _split_samples(
    samples: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training = [sample for sample in samples if str(sample["split"]) == "train"]
    validation = [sample for sample in samples if str(sample["split"]) == "validation"]
    if len(training) + len(validation) != len(samples):
        raise ValueError("Only declared train and validation samples are allowed.")
    if not training or not validation:
        raise ValueError("Both train and validation samples are required.")
    return training, validation


def _checkpoint_epoch(
    state: MatchedState,
    folder: Path,
    signature: str,
    publisher: CheckpointPublisher,
) -> dict[str, Any]:
    """Commit locally, then require independent remote verification before continuing."""
    receipt = save_generation(folder, state.payload(), signature)
    publisher(folder, receipt)
    pointer = folder / "checkpoint.json"
    current = __import__("json").loads(pointer.read_text())
    if current != receipt:
        raise ValueError("Checkpoint pointer changed during publication.")
    return receipt


def train_arm(
    samples: Sequence[dict[str, Any]],
    arm: str,
    velocity_scale: float,
    coordinate_denominator: float,
    velocity_denominator: float,
    settings: TrainingSettings,
    folder: Path,
    signature: str,
    publisher: CheckpointPublisher,
    stop_after_steps: int | None = None,
    progress: ProgressCallback | None = None,
    max_seconds: float | None = None,
) -> tuple[MatchedState, list[dict[str, float | int]]]:
    """Train one fixed arm and require remote checkpoint verification every epoch."""
    training, _ = _split_samples(samples)
    settings.validate(len(training))
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Training time budget must be finite and positive.")
    batches = TrainingBatches(training, settings.batch_plays, settings.seed)
    total_steps = settings.total_steps(len(training))
    if total_steps != settings.epochs * batches.batches_per_epoch:
        raise ValueError("Frozen exposure and batch plan disagree.")
    weight = 0.0 if arm == "coordinate" else 0.1 if arm == "velocity" else None
    if weight is None:
        raise ValueError("Unknown supervision arm.")
    pointer = folder / "checkpoint.json"
    if pointer.is_file():
        state = MatchedState.restore(load_generation(folder, signature))
    else:
        state = MatchedState(settings.width, settings.seed, weight, velocity_scale)
    if state.weight != weight or state.steps > total_steps:
        raise ValueError("Checkpoint arm or cursor does not match the frozen experiment.")
    curve: list[dict[str, float | int]] = []
    limit = total_steps if stop_after_steps is None else min(total_steps, stop_after_steps)
    if limit < state.steps:
        raise ValueError("Stop cursor precedes the durable checkpoint.")
    started = time.monotonic()
    while state.steps < limit:
        cursor = state.steps
        for group in state.optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(
                cursor,
                total_steps,
                settings.warmup_steps,
                settings.peak_learning_rate,
                settings.floor_learning_rate,
            )
        features, labels = supervised_batch(batches.batch(cursor))
        stats = state.step(
            features,
            labels,
            coordinate_denominator,
            velocity_denominator,
        )
        row = {
            "step": state.steps,
            "epoch": (state.steps - 1) // batches.batches_per_epoch + 1,
            "learning_rate": float(state.optimizer.param_groups[0]["lr"]),
            "loss": float(stats["loss"]),
            "coordinate_sse": float(stats["coordinate_sse"]),
            "coordinates": int(stats["coordinates"]),
            "velocity_sse_normalized": float(stats["velocity_sse_normalized"]),
            "velocity_coordinates": int(stats["velocity_coordinates"]),
            "elapsed_seconds": float(time.monotonic() - started),
        }
        curve.append(row)
        if progress is not None:
            progress(row)
        epoch_complete = state.steps % batches.batches_per_epoch == 0
        final_partial = state.steps == limit
        checkpointed = epoch_complete or final_partial
        if checkpointed:
            _checkpoint_epoch(state, folder, signature, publisher)
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            if not checkpointed:
                _checkpoint_epoch(state, folder, signature, publisher)
            raise TimeoutError("Arm exceeded its fixed training-time budget after checkpointing.")
    return state, curve


def evaluate_ema(
    state: MatchedState,
    samples: Sequence[dict[str, Any]],
    batch_plays: int = 64,
) -> pd.DataFrame:
    """Score every validation request exactly once using the fixed final EMA."""
    _, validation = _split_samples(samples)
    ordered = sorted(validation, key=lambda sample: tuple(sample["keys"][0, :2]))
    rows: list[pd.DataFrame] = []
    state.ema.eval()
    with torch.no_grad():
        for start in range(0, len(ordered), batch_plays):
            part = ordered[start : start + batch_plays]
            features, labels = supervised_batch(part)
            prediction = state.ema(features)["coordinate"]
            for index, sample in enumerate(part):
                count = len(sample["keys"])
                delta = (prediction[index, :count] - labels["coordinate"][index, :count]).numpy()
                if delta.shape != (count, 2) or not np.isfinite(delta).all():
                    raise ValueError("Validation errors are missing or nonfinite.")
                frame = pd.DataFrame(sample["keys"], columns=KEYS)
                frame[["dx", "dy"]] = delta
                role_index = sample["player"].astype(np.int64)
                frame["role_id"] = sample["role"][role_index]
                frame["forecast_second"] = np.ceil(sample["keys"][:, 3] / 10).astype(int)
                rows.append(frame)
    errors = pd.concat(rows, ignore_index=True)
    if errors.duplicated(KEYS).any():
        raise ValueError("Validation forecast keys are duplicated.")
    expected = sum(len(sample["keys"]) for sample in validation)
    if len(errors) != expected:
        raise ValueError("Validation row count changed during evaluation.")
    return errors.sort_values(KEYS).reset_index(drop=True)


def _game_sufficient_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Unique nonempty forecast errors are required.")
    work = frame.assign(sse=frame.dx**2 + frame.dy**2)
    grouped = work.groupby("game_id", sort=True).sse.agg(["sum", "count"])
    if not np.isfinite(grouped.to_numpy()).all() or (grouped["count"] <= 0).any():
        raise ValueError("Invalid game error totals.")
    return grouped


def paired_game_bootstrap(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    repeats: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    """Resample identical whole games and compare pooled coordinate RMSE."""
    if repeats < 100:
        raise ValueError("At least 100 paired game resamples are required.")
    left = control[KEYS].sort_values(KEYS).reset_index(drop=True)
    right = treatment[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("Matched arms must score identical forecast keys.")
    a = _game_sufficient_statistics(control)
    b = _game_sufficient_statistics(treatment)
    if not a.index.equals(b.index) or not a["count"].equals(b["count"]):
        raise ValueError("Matched arms must have identical game coordinate counts.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), (repeats, len(a)))
    counts = a["count"].to_numpy()[indices].sum(axis=1)
    control_rmse = np.sqrt(a["sum"].to_numpy()[indices].sum(axis=1) / (2 * counts))
    treatment_rmse = np.sqrt(b["sum"].to_numpy()[indices].sum(axis=1) / (2 * counts))
    delta = treatment_rmse - control_rmse
    low, high = np.quantile(delta, [0.025, 0.975])
    return {
        "delta_mean": float(delta.mean()),
        "delta_ci95_low": float(low),
        "delta_ci95_high": float(high),
    }


def diagnostic_slices(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Secondary diagnostics only; never choose the treatment from these slices."""
    rows: list[dict[str, Any]] = []
    for dimension in ("role_id", "forecast_second"):
        for value, group in frame.groupby(dimension, sort=True):
            rows.append(
                {
                    "dimension": dimension,
                    "value": int(value),
                    "rows": len(group),
                    **error_metrics(group),
                }
            )
    return rows


def summarize_pair(
    control: pd.DataFrame,
    treatment: pd.DataFrame,
    minimum_relative_gain: float = 0.01,
    repeats: int = 10000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Apply the predeclared official-RMSE and paired-game continuation gate."""
    control_keys = control[KEYS].sort_values(KEYS).reset_index(drop=True)
    treatment_keys = treatment[KEYS].sort_values(KEYS).reset_index(drop=True)
    if not control_keys.equals(treatment_keys):
        raise ValueError("Arm evaluation keys differ.")
    control_metrics = error_metrics(control)
    treatment_metrics = error_metrics(treatment)
    control_rmse = control_metrics["coordinate_rmse_yards"]
    treatment_rmse = treatment_metrics["coordinate_rmse_yards"]
    if control_rmse <= 0:
        raise ValueError("Control RMSE must be positive.")
    relative_gain = 1 - treatment_rmse / control_rmse
    paired = paired_game_bootstrap(control, treatment, repeats, seed)
    continue_gate = relative_gain >= minimum_relative_gain and paired["delta_ci95_high"] < 0
    return {
        "control": control_metrics,
        "velocity": treatment_metrics,
        "relative_rmse_gain": float(relative_gain),
        "paired_game_bootstrap": paired,
        "continuation_gate_passed": bool(continue_gate),
        "minimum_relative_gain": minimum_relative_gain,
        "bootstrap_resamples": repeats,
        "bootstrap_seed": seed,
        "validation_games": int(control.game_id.nunique()),
        "validation_rows": len(control),
        "control_slices": diagnostic_slices(control),
        "velocity_slices": diagnostic_slices(treatment),
    }
