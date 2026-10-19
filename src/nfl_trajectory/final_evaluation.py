"""Key-aligned scoring primitives; labels enter only after predictions exist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_trajectory.motion import ENTITY, KEYS, require_keys


def score_predictions(
    truth: pd.DataFrame, keys: pd.DataFrame, predictions: np.ndarray
) -> pd.DataFrame:
    """An exact outer key join prevents row loss or positional target misalignment."""
    require_keys(truth)
    require_keys(keys)
    if predictions.shape != (len(keys), 2) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be aligned finite coordinates.")
    joined = keys.assign(
        px=predictions[:, 0], py=predictions[:, 1], request_order=np.arange(len(keys))
    ).merge(
        truth[KEYS + ["x", "y"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if (
        not joined._merge.eq("both").all()
        or not np.isfinite(joined[["x", "y"]].to_numpy(float)).all()
    ):
        raise ValueError("Every sealed prediction must match exactly one finite outcome.")
    joined = joined.sort_values("request_order").reset_index(drop=True)
    return joined[KEYS].assign(dx=joined.px - joined.x, dy=joined.py - joined.y)


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
