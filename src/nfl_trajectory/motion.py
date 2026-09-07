"""Leakage-safe motion baselines and the competition's coordinate RMSE."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["game_id", "play_id", "nfl_id", "frame_id"]
ENTITY = KEYS[:3]


def require_keys(frame: pd.DataFrame) -> None:
    if not set(KEYS).issubset(frame.columns) or frame.empty:
        raise ValueError("Nonempty tracking rows with all four keys are required.")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Tracking keys must be nonnull and unique.")
    values = frame[KEYS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Tracking identifiers must be finite integers.")
    if (frame["frame_id"] < 1).any():
        raise ValueError("Frame ids start at one.")


def constant_velocity(inputs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Use last two observed frames; output frame 1 is 0.1 s after the final input.

    Velocities are derived from positions to avoid unverified orientation conventions.
    No target coordinates, future player motion, or post-play statistics are accessed.
    """
    require_keys(inputs)
    require_keys(targets)
    if not {"x", "y"}.issubset(inputs.columns):
        raise ValueError("Observed x/y are required.")
    if not np.isfinite(inputs[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("Observed coordinates must be finite.")
    history = inputs.sort_values(KEYS).copy()
    differences = history.groupby(ENTITY, sort=False)[["frame_id", "x", "y"]].diff()
    for axis in ["x", "y"]:
        history[f"v{axis}"] = (differences[axis] / (differences["frame_id"] / 10)).fillna(0)
    last = history.groupby(ENTITY, sort=False).tail(1)
    columns = ENTITY + ["x", "y", "vx", "vy"]
    aligned = targets[KEYS].merge(
        last[columns], on=ENTITY, how="left", sort=False, validate="many_to_one"
    )
    if aligned[["x", "y", "vx", "vy"]].isna().any().any():
        raise ValueError("A prediction target has no matching observed player.")
    result = aligned[KEYS].copy()
    for axis in ["x", "y"]:
        result[axis] = aligned[axis] + aligned[f"v{axis}"] * aligned["frame_id"] / 10
    # No arbitrary clipping: out-of-bounds target rows remain valid measurements.
    return result


def trajectory_metrics(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    """Coordinate RMSE is sqrt(sum(dx²+dy²)/(2N)); ADE/FDE use Euclidean yards."""
    require_keys(truth)
    require_keys(prediction)
    if len(truth) != len(prediction):
        raise ValueError("Predictions must have exactly the target row set.")
    joined = truth[KEYS + ["x", "y"]].merge(
        prediction[KEYS + ["x", "y"]],
        on=KEYS,
        how="outer",
        suffixes=("_true", "_pred"),
        indicator=True,
        validate="one_to_one",
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("Prediction and target keys differ.")
    coordinates = joined[["x_true", "y_true", "x_pred", "y_pred"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite; NaN/Inf never count as predictions.")
    delta = coordinates[:, :2] - coordinates[:, 2:]
    joined["distance"] = np.linalg.norm(delta, axis=1)
    final = joined.sort_values(KEYS).groupby(ENTITY, sort=False).tail(1)
    return {
        "coordinate_rmse_yards": float(np.sqrt(np.mean(delta**2))),
        "ade_frame_weighted_yards": float(joined["distance"].mean()),
        "ade_trajectory_weighted_yards": float(joined.groupby(ENTITY)["distance"].mean().mean()),
        "fde_trajectory_weighted_yards": float(final["distance"].mean()),
        "p95_displacement_yards": float(joined["distance"].quantile(0.95)),
        "coordinate_mae_yards": float(np.abs(delta).mean()),
    }


def split_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reserve latest 15% of game dates; previous 15% validation; rest training.

    All plays, players, and frames of a game remain together. Dates are parsed from
    the published YYYYMMDD## game_id format, which is validated rather than guessed.
    """
    ids = games[["game_id"]].drop_duplicates().copy()
    dates = pd.to_datetime(ids["game_id"].astype(str).str[:8], format="%Y%m%d", errors="raise")
    ids["game_date"] = dates
    days = sorted(dates.unique())
    if len(days) < 10:
        raise ValueError("At least ten distinct game dates are required for temporal splitting.")
    first, second = int(len(days) * 0.70), int(len(days) * 0.85)
    ids["split"] = np.where(
        dates < days[first], "train", np.where(dates < days[second], "validation", "holdout")
    )
    return ids.sort_values(["game_date", "game_id"]).reset_index(drop=True)
