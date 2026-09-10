"""Earlier-date motion priors conditioned on role, forecast time and approach.

Outcomes update training tables only after an entire date has been encoded.
Deployment accepts observed covariates and a frozen JSON-serializable state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

GROUPS = {
    "role_time": ["role", "time_bin"],
    "approach": ["role", "time_bin", "approach_bin"],
    "speed": ["role", "time_bin", "speed_bin"],
    "arrival_phase": ["role", "time_bin", "phase_bin"],
}
STATS = ("mean_x", "mean_y", "log_support", "reliability", "dispersion")
NAMES = [f"conditional_history__{g}__{s}" for g in GROUPS for s in STATS]
SMOOTHING = 20.0


def observed_rows(sample: dict[str, np.ndarray]) -> pd.DataFrame:
    """Extract bins and local axes without accessing any outcome or fitted baseline."""
    slot = sample["player"]
    seconds = sample["time"].astype(float)
    horizon = sample["horizon"][slot].astype(float) / 10
    ball = sample["static"][slot, 2:4].astype(float) * 20
    distance = np.linalg.norm(ball, axis=1)
    unit = ball / np.maximum(distance[:, None], 1e-8)
    # At the landing point, canonical x is a deterministic, reflection-safe axis.
    unit[distance < 1e-8] = [1, 0]
    velocity = sample["velocity"][slot].astype(float)
    if (seconds <= 0).any() or (horizon < seconds - 1e-6).any():
        raise ValueError("Forecast times must be positive and within supplied horizons.")
    if not np.isfinite(np.column_stack([unit, velocity, seconds, horizon])).all():
        raise ValueError("Observed covariates must be finite.")
    result = pd.DataFrame(sample["keys"], columns=["game_id", "play_id", "nfl_id", "frame_id"])
    result["date"] = result.game_id // 100
    result["role"] = sample["role"][slot]
    result["time_bin"] = np.digitize(seconds, [0.5, 1, 1.5, 2, 3])
    result["approach_bin"] = np.digitize((velocity * unit).sum(1), [-1, 1])
    result["speed_bin"] = np.digitize(np.linalg.norm(velocity, axis=1), [2.5, 5, 7.5])
    result["phase_bin"] = np.digitize(seconds / horizon, [1 / 3, 2 / 3])
    result[["ux", "uy"]] = unit
    return result


def training_targets(sample: dict[str, np.ndarray], rows: pd.DataFrame) -> np.ndarray:
    """Motion bias relative to unfitted constant velocity, in yards/second.

    The cached truth is residual to a baseline; adding that same baseline
    reconstructs actual displacement before subtracting observed constant velocity.
    """
    if str(sample["split"]) != "train":
        raise ValueError("Only training outcomes may update conditional motion history.")
    seconds = sample["time"].astype(float)
    displacement = sample["truth"].astype(float) + sample["baseline"].astype(float)
    bias = displacement / seconds[:, None] - sample["velocity"][sample["player"]]
    unit = rows[["ux", "uy"]].to_numpy()
    return np.column_stack(
        [(bias * unit).sum(1), unit[:, 0] * bias[:, 1] - unit[:, 1] * bias[:, 0]]
    )


def _keys(rows: pd.DataFrame, columns: list[str]) -> list[str]:
    return [":".join(map(str, row)) for row in rows[columns].to_numpy(dtype=np.int64)]


def transform(rows: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    """Frozen lookup. No labels are accepted and the supplied state is never mutated."""
    units = rows[["ux", "uy"]].to_numpy()
    parent_mean, parent_second = np.zeros((len(rows), 2)), np.zeros(len(rows))
    output = []
    for name, columns in GROUPS.items():
        table = state["tables"][name]
        stats = np.array([table.get(k, [0.0] * 4) for k in _keys(rows, columns)])
        n, sums, square = stats[:, 0], stats[:, 1:3], stats[:, 3]
        mean = (sums + SMOOTHING * parent_mean) / (n[:, None] + SMOOTHING)
        second = (square + SMOOTHING * parent_second) / (n + SMOOTHING)
        dispersion = np.sqrt(np.maximum(second - (mean**2).sum(1), 0))
        canonical = np.column_stack(
            [
                mean[:, 0] * units[:, 0] - mean[:, 1] * units[:, 1],
                mean[:, 0] * units[:, 1] + mean[:, 1] * units[:, 0],
            ]
        )
        output.append(np.column_stack([canonical, np.log1p(n), n / (n + SMOOTHING), dispersion]))
        if name == "role_time":
            parent_mean, parent_second = mean, second
    return np.concatenate(output, axis=1).astype(np.float32)


def _update(rows: pd.DataFrame, labels: np.ndarray, state: dict[str, Any]) -> None:
    if labels.shape != (len(rows), 2) or not np.isfinite(labels).all():
        raise ValueError("Training motion labels must be finite and aligned.")
    values = rows.copy()
    values[["radial", "tangent"]] = labels
    for name, columns in GROUPS.items():
        # One observation per player/play/context bin: frame counts are not sample support.
        entities = ["game_id", "play_id", "nfl_id", *columns]
        units = values.groupby(entities, sort=True)[["radial", "tangent"]].mean().reset_index()
        units["square"] = units.radial**2 + units.tangent**2
        for key, part in units.groupby(columns, sort=True):
            token = ":".join(str(int(v)) for v in key)
            delta = np.array([len(part), part.radial.sum(), part.tangent.sum(), part.square.sum()])
            old = np.asarray(state["tables"][name].get(token, [0.0] * 4))
            state["tables"][name][token] = (old + delta).tolist()


def fit_transform(
    rows: pd.DataFrame, labels: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode training before each date's update; freeze history for later evaluation."""
    if len(rows) == 0 or train.dtype != bool or not train.any() or train.all():
        raise ValueError("Nonempty boolean training and later evaluation partitions are required.")
    if labels.shape != (len(rows), 2):
        raise ValueError("Labels must preserve row alignment.")
    dates = rows.date.to_numpy()
    if dates[train].max() >= dates[~train].min():
        raise ValueError("Evaluation must follow every training date.")
    state: dict[str, Any] = {
        "version": 1,
        "smoothing": SMOOTHING,
        "last_training_date": int(dates[train].max()),
        "tables": {name: {} for name in GROUPS},
    }
    encoded = np.zeros((len(rows), len(NAMES)), dtype=np.float32)
    for date in np.unique(dates[train]):
        selected = train & (dates == date)
        current = rows.loc[selected]
        encoded[selected] = transform(current, state)
        _update(current, labels[selected], state)
    encoded[~train] = transform(rows.loc[~train], state)
    return encoded, state
