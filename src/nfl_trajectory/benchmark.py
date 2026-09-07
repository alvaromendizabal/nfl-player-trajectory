"""Resumable, week-bounded training and evaluation under a frozen temporal protocol."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.models import (
    BASELINES,
    INPUT_COLUMNS,
    design,
    fit_statistics,
    predict,
    predict_from_design,
    sufficient_statistics,
)
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

STATE_COLUMNS = [
    "x",
    "y",
    "ball_land_x",
    "ball_land_y",
    "num_frames_output",
    "vx",
    "vy",
    "ax",
    "ay",
    "smooth_vx",
    "smooth_vy",
]
MODELS = [*BASELINES, "role_ridge"]


def protocol(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lock date boundaries before modeling; an edited split cannot silently replace them."""
    source = root / "artifacts/game_splits.csv"
    audit_path = root / "artifacts/audit_summary.json"
    if not source.exists() or not audit_path.exists():
        raise ValueError("Run nfl download and nfl audit before nfl benchmark.")
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "passed":
        raise ValueError("A passed real-data audit is required.")
    split = pd.read_csv(source)
    if not {"game_id", "game_date", "split"}.issubset(split) or split.game_id.duplicated().any():
        raise ValueError("Split manifest must assign each game exactly once.")
    if split.isna().any().any() or set(split.split) != {"train", "validation", "holdout"}:
        raise ValueError("Complete train, validation, and holdout partitions are required.")
    date = pd.to_datetime(split.game_date, errors="raise")
    expected = pd.to_datetime(split.game_id.astype(str).str[:8], format="%Y%m%d")
    if not date.equals(expected):
        raise ValueError("Split dates disagree with game identifiers.")
    bounds = split.groupby("split").game_date.agg(["min", "max"])
    if not (
        bounds.loc["train", "max"] < bounds.loc["validation", "min"]
        and bounds.loc["validation", "max"] < bounds.loc["holdout", "min"]
    ):
        raise ValueError("Temporal partitions overlap or run backward.")
    audited_games = [game for pair in audit["pairs"] for game in pair["games"]]
    if len(set(audited_games)) != len(audited_games) or set(split.game_id) != set(audited_games):
        raise ValueError("Split manifest does not exactly cover the audited games.")
    value = {
        "format": 1,
        "competition": audit["competition"],
        "split_sha256": sha256(source),
        "partitions": {
            name: {
                "games": int(len(rows)),
                "first_date": rows.game_date.min(),
                "last_date": rows.game_date.max(),
            }
            for name, rows in split.groupby("split")
        },
        "development_split": "validation",
        "holdout_evaluation": "not_run",
        "selection_metric": "coordinate_rmse_yards",
        "ridge_alpha": 0.001,
        "confidence_interval": {
            "method": "game_cluster_percentile_bootstrap",
            "resamples": 2000,
            "seed": 2026,
        },
    }
    destination = root / "artifacts/benchmark/protocol.json"
    if destination.exists() and json.loads(destination.read_text()) != value:
        raise ValueError(
            "Frozen benchmark protocol differs from current inputs. Review the protocol explicitly."
        )
    if not destination.exists():
        atomic_json(destination, value)
    return split, audit


def signature(root: Path, inputs: list[Path], parameters: Any) -> str:
    # Numerical stages depend on these modules, not presentation-only source changes.
    files = {str(path.relative_to(root)): sha256(path) for path in inputs}
    for name in ["benchmark", "models", "motion"]:
        files[f"src/nfl_trajectory/{name}.py"] = sha256(Path(__file__).with_name(f"{name}.py"))
    lock = root / "uv.lock"
    if lock.exists():
        files["uv.lock"] = sha256(lock)
    payload = {"files": files, "parameters": parameters}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_design(
    path: Path, state: pd.DataFrame, features: np.ndarray, truth: pd.DataFrame, split: pd.DataFrame
) -> None:
    aligned = state[KEYS].merge(
        truth[KEYS + ["x", "y"]], on=KEYS, how="left", validate="one_to_one"
    )
    labels = state[["game_id"]].merge(
        split[["game_id", "split"]], on="game_id", validate="many_to_one"
    )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        keys=state[KEYS].to_numpy(dtype=np.int64),
        state=state[STATE_COLUMNS].to_numpy(dtype=float),
        roles=state.player_role.to_numpy(dtype=str),
        features=features,
        truth=aligned[["x", "y"]].to_numpy(dtype=float),
        split=labels.split.to_numpy(dtype=str),
    )
    atomic_bytes(path, buffer.getvalue())


def load_design(path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        state = pd.DataFrame(stored["keys"], columns=KEYS)
        state[STATE_COLUMNS] = stored["state"]
        state["player_role"] = stored["roles"]
        truth = state[KEYS].copy()
        truth[["x", "y"]] = stored["truth"]
        return state, stored["features"], truth, stored["split"]


def prepare_week(root: Path, pair: dict[str, Any], splits: pd.DataFrame, destination: Path) -> None:
    games = set(splits.loc[splits.split != "holdout", "game_id"])
    x_path = root / "data/raw/train" / pair["file"]
    y_path = x_path.with_name(x_path.name.replace("input_", "output_"))
    x = pd.read_csv(x_path, usecols=INPUT_COLUMNS)
    y = pd.read_csv(y_path, usecols=KEYS + ["x", "y"])
    x = x[x.game_id.isin(games)]
    y = y[y.game_id.isin(games)]
    require_keys(y)
    state, features = design(x, y[KEYS])
    save_design(destination / "design.npz", state, features, y, splits)
    train_games = set(splits.loc[splits.split == "train", "game_id"])
    mask = state.game_id.isin(train_games).to_numpy()
    statistics = (
        sufficient_statistics(state.loc[mask], features[mask], y[y.game_id.isin(train_games)])
        if mask.any()
        else {}
    )
    atomic_json(destination / "statistics.json", statistics)
    entities = state.loc[mask].drop_duplicates(ENTITY)
    train_inputs = x[x.game_id.isin(train_games)]
    horizons = entities.num_frames_output.to_numpy() / 10
    speed = np.linalg.norm(entities[["smooth_vx", "smooth_vy"]].to_numpy(), axis=1)
    eda = {
        "file": pair["file"],
        "input_rows": len(train_inputs),
        "target_rows": int(mask.sum()),
        "games": int(train_inputs.game_id.nunique()),
        "trajectories": len(entities),
        "plays": len(train_inputs[["game_id", "play_id"]].drop_duplicates()),
        "roles": {str(k): int(v) for k, v in entities.player_role.value_counts().items()},
        "horizon_bins_seconds": [0, 0.5, 1, 1.5, 2, 3, 5, 20],
        "horizon_counts": np.histogram(horizons, bins=[0, 0.5, 1, 1.5, 2, 3, 5, 20])[0].tolist(),
        "speed_bins_yards_per_second": [0, 2, 4, 6, 8, 10, 15, 100],
        "speed_counts": np.histogram(speed, bins=[0, 2, 4, 6, 8, 10, 15, 100])[0].tolist(),
        "missing_observed_cells": int(train_inputs.isna().sum().sum()),
        "split": "train",
    }
    atomic_json(destination / "eda.json", eda)


def error_metrics(frame: pd.DataFrame) -> dict[str, float]:
    require_keys(frame)
    delta = frame[["dx", "dy"]].to_numpy()
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite prediction errors.")
    distances = frame[KEYS].copy()
    distances["distance"] = np.linalg.norm(delta, axis=1)
    final = distances.sort_values(KEYS).groupby(ENTITY, sort=False).tail(1)
    return {
        "coordinate_rmse_yards": float(np.sqrt(np.mean(delta**2))),
        "ade_frame_weighted_yards": float(distances.distance.mean()),
        "ade_trajectory_weighted_yards": float(distances.groupby(ENTITY).distance.mean().mean()),
        "fde_trajectory_weighted_yards": float(final.distance.mean()),
        "p95_displacement_yards": float(distances.distance.quantile(0.95)),
        "coordinate_mae_yards": float(np.abs(delta).mean()),
    }


def bootstrap_scores(frame: pd.DataFrame, repeats: int = 2000, seed: int = 2026) -> np.ndarray:
    """Resample whole games and recompute pooled coordinate RMSE, never mean game RMSE."""
    if repeats < 1 or frame.empty:
        raise ValueError("Positive resample count and nonempty errors are required.")
    values = (
        frame.assign(sse=frame.dx**2 + frame.dy**2)
        .groupby("game_id", sort=True)
        .sse.agg(["sum", "count"])
    )
    samples = np.random.default_rng(seed).integers(0, len(values), (repeats, len(values)))
    return np.sqrt(
        values["sum"].to_numpy()[samples].sum(axis=1)
        / (2 * values["count"].to_numpy()[samples].sum(axis=1))
    )


def evaluate_week(cache: Path, fitted: dict[str, Any], output: Path) -> None:
    state, features, truth, labels = load_design(cache)
    selected = labels == "validation"
    records = []
    if selected.any():
        state = state.loc[selected].reset_index(drop=True)
        features = features[selected]
        truth = truth.loc[selected].reset_index(drop=True)
        for model in MODELS:
            prediction = predict_from_design(state, features, model, fitted)
            result = state[KEYS + ["player_role", "num_frames_output"]].copy()
            result[["dx", "dy"]] = prediction[["x", "y"]].to_numpy() - truth[["x", "y"]].to_numpy()
            result["model"] = model
            records.append(result)
    columns = KEYS + ["player_role", "num_frames_output", "dx", "dy", "model"]
    errors = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)
    atomic_bytes(output, errors.to_csv(index=False).encode())


def summarize(errors: pd.DataFrame) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    reference = bootstrap_scores(errors[errors.model == "constant_velocity"])
    reference_keys = (
        errors.loc[errors.model == "constant_velocity", KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    for model, frame in errors.groupby("model", sort=True):
        keys = frame[KEYS].sort_values(KEYS).reset_index(drop=True)
        if not keys.equals(reference_keys):
            raise ValueError("Every model must score the exact same validation rows.")
        sampled = bootstrap_scores(frame)
        interval = np.quantile(sampled, [0.025, 0.975])
        delta = np.quantile(sampled - reference, [0.025, 0.975])
        scores.append(
            {
                "model": str(model),
                **error_metrics(frame),
                "rmse_ci95_low": float(interval[0]),
                "rmse_ci95_high": float(interval[1]),
                "delta_vs_velocity_ci95_low": float(delta[0]),
                "delta_vs_velocity_ci95_high": float(delta[1]),
            }
        )
        for role, rows in frame.groupby("player_role"):
            slices.append(
                {
                    "model": str(model),
                    "dimension": "role",
                    "value": str(role),
                    "rows": len(rows),
                    "coordinate_rmse_yards": error_metrics(rows)["coordinate_rmse_yards"],
                }
            )
        for second, rows in frame.groupby(np.ceil(frame.frame_id / 10).astype(int)):
            slices.append(
                {
                    "model": str(model),
                    "dimension": "forecast_second",
                    "value": str(second),
                    "rows": len(rows),
                    "coordinate_rmse_yards": error_metrics(rows)["coordinate_rmse_yards"],
                }
            )
    ordered = sorted(scores, key=lambda item: item["coordinate_rmse_yards"])
    cv = next(item for item in scores if item["model"] == "constant_velocity")
    return {
        "status": "passed",
        "split": "validation",
        "holdout_evaluation": "not_run",
        "models": ordered,
        "slices": slices,
        "selected_model": ordered[0]["model"],
        "improvement_vs_velocity_percent": (
            100 * (1 - ordered[0]["coordinate_rmse_yards"] / cv["coordinate_rmse_yards"])
            if cv["coordinate_rmse_yards"] > 0
            else 0.0
        ),
        "validation_games": int(errors.game_id.nunique()),
        "validation_rows_per_model": len(reference_keys),
        "validation_trajectories": len(reference_keys[ENTITY].drop_duplicates()),
    }


def measure_latency(root: Path, fitted: dict[str, Any], selected: str) -> dict[str, Any]:
    """Time complete per-play feature construction and prediction after one warmup."""
    for path in sorted((root / "artifacts/benchmark/weeks").glob("*/design.npz")):
        state, _, _, labels = load_design(path)
        validation = state.loc[labels == "validation"]
        if validation.empty:
            continue
        observed = pd.read_csv(
            root / "data/raw/train" / f"{path.parent.name}.csv", usecols=INPUT_COLUMNS
        )
        plays = (
            validation[["game_id", "play_id"]]
            .drop_duplicates()
            .sort_values(["game_id", "play_id"])
            .head(32)
        )
        durations: list[float] = []
        for row in plays.itertuples(index=False):
            x = observed[(observed.game_id == row.game_id) & (observed.play_id == row.play_id)]
            target = validation.loc[
                (validation.game_id == row.game_id) & (validation.play_id == row.play_id), KEYS
            ]
            if not durations:
                predict(x, target, selected, fitted)
            started = time.perf_counter()
            result = predict(x, target, selected, fitted)
            durations.append((time.perf_counter() - started) * 1000)
            if len(result) != len(target):
                raise ValueError("Timed prediction changed the target row count.")
        return {
            "model": selected,
            "plays": len(durations),
            "median_ms": float(np.median(durations)),
            "p95_ms": float(np.quantile(durations, 0.95)),
            "includes": "feature construction and prediction; one complete play per call",
            "excludes": "CSV loading, networking, server startup",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    raise ValueError("No validation plays available for latency measurement.")


def benchmark(root: Path, run: Run) -> None:
    """Prepare only development games, fit on train, and report validation once per signature."""
    destination = root / "artifacts/benchmark"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        splits, audit = protocol(root)
        development = set(splits.loc[splits.split != "holdout", "game_id"])
        pairs = [pair for pair in audit["pairs"] if set(pair["games"]) & development]
        caches = []
        for index, pair in enumerate(pairs, 1):
            cache = destination / "weeks" / Path(pair["file"]).stem
            outputs = [cache / name for name in ["design.npz", "statistics.json", "eda.json"]]
            x_path = root / "data/raw/train" / pair["file"]
            y_path = x_path.with_name(x_path.name.replace("input_", "output_"))
            sig = signature(
                root, [x_path, y_path, root / "artifacts/game_splits.csv"], {"stage": "prepare"}
            )
            stage(
                root,
                f"benchmark-prepare-{x_path.stem}",
                sig,
                outputs,
                partial(prepare_week, root, pair, splits, cache),
                run,
            )
            caches.append(cache)
            run.event(
                "benchmark_progress", phase="prepare", completed_weeks=index, total_weeks=len(pairs)
            )
        fitted_path = destination / "model.json"
        statistics_paths = [cache / "statistics.json" for cache in caches]

        def fit() -> None:
            statistics = [json.loads(path.read_text()) for path in statistics_paths]
            fitted = fit_statistics([item for item in statistics if item])
            fitted["training_games"] = sorted(
                int(v) for v in splits.loc[splits.split == "train", "game_id"]
            )
            fitted["split_sha256"] = sha256(root / "artifacts/game_splits.csv")
            atomic_json(fitted_path, fitted)

        stage(
            root,
            "benchmark-fit",
            signature(root, statistics_paths, {"alpha": 0.001}),
            [fitted_path],
            fit,
            run,
        )
        fitted = json.loads(fitted_path.read_text())
        paths = []
        for index, cache in enumerate(caches, 1):
            output = cache / "errors.csv"
            stage(
                root,
                f"benchmark-evaluate-{cache.name}",
                signature(root, [cache / "design.npz", fitted_path], {"models": MODELS}),
                [output],
                partial(evaluate_week, cache / "design.npz", fitted, output),
                run,
            )
            paths.append(output)
            run.event(
                "benchmark_progress",
                phase="evaluate",
                completed_weeks=index,
                total_weeks=len(caches),
            )
        summary_path = destination / "summary.json"

        def aggregate() -> None:
            frames = [pd.read_csv(path) for path in paths]
            errors = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
            summary = summarize(errors)
            if summary["validation_games"] != int((splits.split == "validation").sum()):
                raise ValueError("Validation scoring did not cover every frozen validation game.")
            summary["split_sha256"] = fitted["split_sha256"]
            summary["data_sha256"] = {
                pair["file"]: sha256(root / "data/raw/train" / pair["file"]) for pair in pairs
            }
            summary["numerical_signature"] = signature(root, paths + [fitted_path], {"summary": 1})
            summary["runtime"] = {
                "python": platform.python_version(),
                "platform": platform.system(),
                "machine": platform.machine(),
            }
            atomic_json(summary_path, summary)

        stage(
            root,
            "benchmark-summary",
            signature(root, paths + [fitted_path], {"summary": 1}),
            [summary_path],
            aggregate,
            run,
        )
        summary = json.loads(summary_path.read_text())
        latency_path = destination / "latency.json"
        stage(
            root,
            "benchmark-latency",
            signature(root, [summary_path, fitted_path], {"plays": 32}),
            [latency_path],
            lambda: atomic_json(
                latency_path, measure_latency(root, fitted, summary["selected_model"])
            ),
            run,
        )
        # Presentation is cheap and always rebuilt from verified numerical artifacts.
        from nfl_trajectory.visualization import build_report

        build_report(root, run)
        run.event(
            "benchmark_completed",
            selected_model=summary["selected_model"],
            validation_rmse=summary["models"][0]["coordinate_rmse_yards"],
            holdout_evaluation="not_run",
            elapsed_total_seconds=round(time.monotonic() - run.started, 3),
        )
