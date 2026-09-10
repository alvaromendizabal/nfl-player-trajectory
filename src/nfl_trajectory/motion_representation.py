"""Matched end-to-end use of observed, multi-scale player motion states."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from nfl_trajectory.domain_features import candidates
from nfl_trajectory.domain_probe import screen
from nfl_trajectory.motion import KEYS
from nfl_trajectory.motion_research import feature_group
from nfl_trajectory.runtime import Run
from nfl_trajectory.temporal_data import CHANNELS, reflect
from nfl_trajectory.temporal_model import (
    TemporalModel,
    collate,
    coordinate_loss,
    load_checkpoint,
    save_checkpoint,
)

SETTINGS: dict[str, Any] = {
    "epochs": 12,
    "batch_plays": 64,
    "width": 96,
    "learning_rate": 0.0002,
    "minimum_learning_rate": 0.00001,
    "weight_decay": 0.01,
    "ema_decay": 0.98,
    "gradient_clip": 1.0,
    "seed": 2026,
    "threads": 4,
    "training_seconds_limit": 450,
    "reflection_probability": 0.5,
    "selection": "fixed final-epoch EMA; no evaluation-based selection",
    "initialization": "fold-local saved 40-epoch EMA; fresh matched AdamW",
}
ARMS = ("control", "smoothed_state")


def motion_state(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute the previously tested thirty states for every observed player."""
    count = len(sample["ids"])
    query = {
        **sample,
        "player": np.arange(count),
        "time": np.zeros(count),
        "baseline": np.zeros((count, 2)),
    }
    values, names, _ = candidates(query)
    indexes = [i for i, name in enumerate(names) if feature_group(name) == "smoothed_state"]
    if len(indexes) != 30:
        raise ValueError("The frozen motion-state schema must contain thirty candidates.")
    return values[:, indexes].astype(np.float32), [names[i] for i in indexes]


def attach_motion(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values, _ = motion_state(sample)
    return {**sample, "motion": values}


def reflect_motion(sample: dict[str, np.ndarray], odd: np.ndarray) -> dict[str, np.ndarray]:
    result = reflect(sample)
    result["motion"][:, odd] *= -1
    return result


def fit_motion_screen(samples: list[dict[str, np.ndarray]], names: list[str]) -> dict[str, Any]:
    """Use all training players and their lawful reflected states, never evaluation."""
    values = np.concatenate([s["motion"] for s in samples if str(s["split"]) == "train"])
    reflected = values.copy()
    reflected[:, [n.endswith("_y") for n in names]] *= -1
    return screen(np.concatenate([values, reflected]), names, ["motion"] * len(names))


def collate_motion(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch = collate(samples)
    values = torch.zeros((*batch["valid"].shape, 30), dtype=torch.float32)
    for i, sample in enumerate(samples):
        values[i, : len(sample["ids"])] = torch.from_numpy(sample["motion"])
    batch["motion"] = values
    return batch


class MotionModel(TemporalModel):
    """Expand the temporal input with zero-initialized, interpretable state channels."""

    def __init__(
        self,
        initial: dict[str, torch.Tensor],
        fitted: dict[str, Any],
        enabled: bool,
        width: int = 96,
    ) -> None:
        super().__init__(width)
        self.load_state_dict(initial)
        old = cast(nn.Conv1d, self.temporal[0])
        expanded = nn.Conv1d(len(CHANNELS) + 31, 48, 3, padding=1)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : len(CHANNELS)] = old.weight[:, : len(CHANNELS)]
            expanded.weight[:, -1] = old.weight[:, -1]
            expanded.bias.copy_(old.bias)
        self.temporal[0] = expanded
        self.enabled = enabled
        self.register_buffer("motion_mean", torch.tensor(fitted["mean"], dtype=torch.float32))
        self.register_buffer("motion_scale", torch.tensor(fitted["scale"], dtype=torch.float32))
        self.register_buffer("motion_keep", torch.tensor(fitted["retained"], dtype=torch.bool))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        motion = ((batch["motion"] - self.motion_mean) / self.motion_scale).clamp(-10, 10)
        motion = motion * self.motion_keep * float(self.enabled)
        history = batch["history"]
        augmented = torch.cat(
            [history, motion[:, :, None, :].expand(-1, -1, history.shape[2], -1)], -1
        )
        return super().forward({**batch, "history": augmented})


@torch.inference_mode()
def evaluate(
    model: MotionModel, samples: list[dict[str, np.ndarray]], batch_plays: int = 64
) -> pd.DataFrame:
    model.eval()
    parts = []
    for start in range(0, len(samples), batch_plays):
        subset = samples[start : start + batch_plays]
        predictions = model(collate_motion(subset)).numpy()
        for i, sample in enumerate(subset):
            frame = pd.DataFrame(sample["keys"], columns=KEYS)
            frame[["dx", "dy"]] = (predictions[i, : len(frame)] - sample["truth"]) * sample["sign"]
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(KEYS).any() or not np.isfinite(result[["dx", "dy"]]).all().all():
        raise ValueError("Evaluation must preserve every unique finite forecast request.")
    return result


def train(
    model: MotionModel,
    samples: list[dict[str, np.ndarray]],
    names: list[str],
    folder: Path,
    signature: str,
    run: Run,
    settings: dict[str, Any] | None = None,
    interrupt_after_batches: int | None = None,
) -> tuple[MotionModel, dict[str, Any]]:
    """Continue the entire model with exact batch, optimizer and RNG recovery."""
    config = SETTINGS if settings is None else settings
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    training = [s for s in samples if str(s["split"]) == "train"]
    if not training:
        raise ValueError("Training examples are required.")
    odd = np.array([n.endswith("_y") for n in names])
    checkpoint = folder / "checkpoint.pt"
    state: dict[str, Any] = {
        "epoch": 0,
        "next_batch": 0,
        "curve": [],
        "seconds": 0.0,
        "sse": 0.0,
        "coordinates": 0,
    }
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, signature)
        model.load_state_dict(state.pop("model"))
        ema.load_state_dict(state.pop("ema"))
        optimizer.load_state_dict(state.pop("optimizer"))
        torch.set_rng_state(state.pop("rng"))
        run.event("representation_resumed", epoch=state["epoch"], next_batch=state["next_batch"])
    start, previous, batches_this_call = time.monotonic(), float(state["seconds"]), 0
    mean_rows = sum(len(s["time"]) for s in training) / len(training)

    def persist() -> None:
        state["seconds"] = previous + time.monotonic() - start
        save_checkpoint(
            checkpoint,
            {
                **state,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": torch.get_rng_state(),
            },
            signature,
        )

    try:
        while state["epoch"] < config["epochs"]:
            epoch = state["epoch"]
            order = np.random.default_rng(config["seed"] + epoch).permutation(len(training))
            count = math.ceil(len(order) / config["batch_plays"])
            cosine = (1 + math.cos(math.pi * epoch / max(config["epochs"] - 1, 1))) / 2
            lr = (
                config["minimum_learning_rate"]
                + (config["learning_rate"] - config["minimum_learning_rate"]) * cosine
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            for batch_index in range(state["next_batch"], count):
                if previous + time.monotonic() - start >= config["training_seconds_limit"]:
                    raise TimeoutError("The predeclared per-arm training budget was reached.")
                indices = order[
                    batch_index * config["batch_plays"] : (batch_index + 1) * config["batch_plays"]
                ]
                rng = np.random.default_rng(config["seed"] * 100000 + epoch * 1000 + batch_index)
                subset = [
                    reflect_motion(training[i], odd)
                    if rng.random() < config["reflection_probability"]
                    else training[i]
                    for i in indices
                ]
                batch = collate_motion(subset)
                optimizer.zero_grad(set_to_none=True)
                mse = coordinate_loss(model(batch), batch)
                loss = mse * batch["scored"].sum() / (len(indices) * mean_rows)
                if not torch.isfinite(loss):
                    raise ValueError("The training objective is not finite.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), model.parameters(), strict=True):
                        a.lerp_(b, 1 - config["ema_decay"])
                coordinates = 2 * int(batch["scored"].sum())
                state["sse"] += float(mse.detach()) * coordinates
                state["coordinates"] += coordinates
                state["next_batch"] = batch_index + 1
                batches_this_call += 1
                if interrupt_after_batches == batches_this_call:
                    raise KeyboardInterrupt("Deliberate recovery test.")
            row = {
                "epoch": epoch + 1,
                "training_rmse_augmented": math.sqrt(state["sse"] / state["coordinates"]),
                "learning_rate": lr,
            }
            state["curve"].append(row)
            state.update(epoch=epoch + 1, next_batch=0, sse=0.0, coordinates=0)
            persist()
            run.event("representation_epoch_completed", **row, training_seconds=state["seconds"])
    except BaseException:
        persist()
        raise
    return ema, state
