"""Observed-only play tensors and invertible geometry for temporal forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_candidates import HISTORY_NAMES
from nfl_trajectory.features import ROLES, WIDTH, _history
from nfl_trajectory.motion import ENTITY, KEYS, require_keys

HISTORY = 20
# Fixed physical units, not statistics fitted on development data.
CHANNELS = {
    "relative_x": 10.0,
    "relative_y": 10.0,
    "vx": 10.0,
    "vy": 10.0,
    "ax": 20.0,
    "ay": 20.0,
    "dir_sin": 1.0,
    "dir_cos": 1.0,
    "o_sin": 1.0,
    "o_cos": 1.0,
    "ball_dx": 20.0,
    "ball_dy": 20.0,
    "ball_distance": 20.0,
    "closing_speed": 10.0,
    "lateral_speed": 10.0,
    "radial_acceleration": 20.0,
    "lateral_acceleration": 20.0,
    "turn_rate": 10.0,
    "s": 10.0,
    "a": 10.0,
    "telemetry__s__present": 1.0,
    "telemetry__a__present": 1.0,
    "telemetry__dir__present": 1.0,
    "telemetry__o__present": 1.0,
    "receiver_dx": 20.0,
    "receiver_dy": 20.0,
    "receiver_present": 1.0,
    "passer_dx": 20.0,
    "passer_dy": 20.0,
    "passer_present": 1.0,
}
ODD_CHANNELS = [
    list(CHANNELS).index(c)
    for c in (
        "relative_y",
        "vy",
        "ay",
        "dir_cos",
        "o_cos",
        "ball_dy",
        "lateral_speed",
        "lateral_acceleration",
        "turn_rate",
        "receiver_dy",
        "passer_dy",
    )
]
STATIC_NAMES = [
    "x_from_midfield",
    "y_from_midfield",
    "ball_dx",
    "ball_dy",
    "vx",
    "vy",
    "horizon_seconds",
    "last_observation_age",
    "history_fraction",
    "scored_player",
    *HISTORY_NAMES,
]
ODD_STATIC = [1, 3, 5, 13, 18]
EDGE_NAMES = [
    "dx",
    "dy",
    "dvx",
    "dvy",
    "distance",
    "closing_speed",
    "closest_time",
    "closest_distance",
    "same_side",
    "receiver",
    "passer",
    "arrival_distance",
]


def history_lookup(state: dict[str, Any], entities: pd.DataFrame) -> np.ndarray:
    """Frozen deployment encoder: no outcomes or updates are accepted."""
    result = np.zeros((len(entities), 10), dtype=np.float32)
    for i, row in enumerate(entities.itertuples(index=False)):
        for offset, group, key in ((0, "player", str(row.nfl_id)), (5, "role", row.player_role)):
            n, sx, sy, square = state["tables"][group].get(key, [0.0] * 4)
            denominator = n + state["smoothing"]
            result[i, offset : offset + 5] = (
                np.log1p(n),
                float(n == 0),
                sx / denominator,
                sy / denominator,
                np.sqrt(max(square / denominator - (sx**2 + sy**2) / denominator**2, 0)),
            )
    return result


def play_features(raw: pd.DataFrame, priors: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Encode exactly one observed play. Future coordinates are not an argument.

    Slots use stable player identities throughout time. IDs are keys only, never
    numerical covariates. Anchor joins match observed frame IDs, not future rows.
    """
    return encode_play(observed_history(raw), priors)


def observed_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Vectorize observed kinematics across a week, preserving supplied target flags."""
    if "player_to_predict" not in raw or not raw.player_to_predict.isin([True, False]).all():
        raise ValueError("Observed input needs boolean player_to_predict flags.")
    return _history(raw).merge(
        raw[KEYS + ["player_to_predict"]], on=KEYS, validate="one_to_one", sort=False
    )


def encode_play(h: pd.DataFrame, priors: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Encode a play from observed_history; public inference uses play_features."""
    if h[KEYS[:2]].drop_duplicates().shape[0] != 1:
        raise ValueError("A temporal example must contain exactly one observed play.")
    terminal = h.groupby(ENTITY, sort=True).tail(1).sort_values("nfl_id")
    count = len(terminal)
    if count == 0 or count > 22:
        raise ValueError("A play must contain between one and 22 observed players.")
    cutoff = int(h.frame_id.max())
    # Differences and acceleration are calculated only from observed histories.
    h = h[h.frame_id > cutoff - HISTORY].copy()
    for prefix, role in (("receiver", "Targeted Receiver"), ("passer", "Passer")):
        anchor = h[h.player_role.eq(role)].sort_values(["frame_id", "nfl_id"])
        anchor = anchor.drop_duplicates("frame_id").set_index("frame_id")
        for axis in ("x", "y"):
            h[f"{prefix}_d{axis}"] = (h.frame_id.map(anchor[axis]) - h[axis]).fillna(0)
        h[f"{prefix}_present"] = h.frame_id.isin(anchor.index).astype(float)
    slot = {int(n): i for i, n in enumerate(terminal.nfl_id)}
    players = h.nfl_id.map(slot).to_numpy(int)
    frames = h.frame_id.to_numpy(int) - cutoff + HISTORY - 1
    values = np.zeros((count, HISTORY, len(CHANNELS)), dtype=np.float32)
    present = np.zeros((count, HISTORY), dtype=bool)
    values[players, frames] = h[list(CHANNELS)].to_numpy(float) / np.array(list(CHANNELS.values()))
    present[players, frames] = True
    historical = np.zeros((count, 10), dtype=np.float32)
    historical[:, [1, 6]] = 1
    if priors is not None:
        if priors.duplicated(ENTITY).any():
            raise ValueError("Historical encodings must have unique player/play keys.")
        joined = terminal[ENTITY].merge(priors, on=ENTITY, how="left", validate="one_to_one")
        available = joined[HISTORY_NAMES].notna().all(axis=1).to_numpy()
        historical[available] = joined.loc[available, HISTORY_NAMES].to_numpy(float)
    static = np.column_stack(
        [
            (terminal.x - 60) / 60,
            (terminal.y - WIDTH / 2) / (WIDTH / 2),
            terminal.ball_dx / 20,
            terminal.ball_dy / 20,
            terminal.vx / 10,
            terminal.vy / 10,
            terminal.num_frames_output / 10,
            (cutoff - terminal.frame_id) / 10,
            present.mean(axis=1),
            terminal.player_to_predict.astype(float),
            historical,
        ]
    ).astype(np.float32)
    roles = (
        terminal.player_role.map({name: i for i, name in enumerate(ROLES)})
        .fillna(len(ROLES) - 1)
        .to_numpy(np.int64)
    )
    side = terminal.player_side.eq("Offense").to_numpy(np.int64)
    xy = terminal[["x", "y"]].to_numpy(np.float32)
    velocity = terminal[["vx", "vy"]].to_numpy(np.float32)
    result = {
        "history": values,
        "observed": present,
        "static": static,
        "role": roles,
        "side": side,
        "xy": xy,
        "velocity": velocity,
        "ids": terminal.nfl_id.to_numpy(np.int64),
        "sign": np.array(float(terminal.sign.iloc[0]), dtype=np.float32),
        "horizon": terminal.num_frames_output.to_numpy(np.int64),
    }
    if any(not np.isfinite(a).all() for a in result.values()):
        raise ValueError("Temporal features must be finite.")
    return result


def edge_features(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Directed observed geometry; closest approach is a kinematic hypothesis."""
    xy, velocity = sample["xy"], sample["velocity"]
    delta = xy[None] - xy[:, None]
    dv = velocity[None] - velocity[:, None]
    distance = np.linalg.norm(delta, axis=-1)
    dot = (delta * dv).sum(-1)
    closing = -dot / np.maximum(distance, 0.1)
    horizon = sample["horizon"][:, None] / 10
    closest_t = np.clip(-dot / np.maximum((dv**2).sum(-1), 0.1), 0, horizon)
    closest = np.linalg.norm(delta + closest_t[..., None] * dv, axis=-1)
    arrival = np.linalg.norm(delta + horizon[..., None] * dv, axis=-1)
    count = len(xy)
    role = sample["role"]
    return np.stack(
        [
            delta[..., 0] / 20,
            delta[..., 1] / 20,
            dv[..., 0] / 10,
            dv[..., 1] / 10,
            distance / 20,
            closing / 10,
            closest_t / 5,
            closest / 20,
            (sample["side"][:, None] == sample["side"][None]).astype(float),
            np.broadcast_to(role == ROLES.index("Targeted Receiver"), (count, count)),
            np.broadcast_to(role == ROLES.index("Passer"), (count, count)),
            arrival / 20,
        ],
        axis=-1,
    ).astype(np.float32)


def attach_targets(
    sample: dict[str, np.ndarray],
    targets: pd.DataFrame,
    baseline: np.ndarray,
    truth: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Keys determine output requests. Coordinates are isolated from model inputs."""
    require_keys(targets)
    slots = pd.Index(sample["ids"]).get_indexer(targets.nfl_id)
    frames = targets.frame_id.to_numpy(np.int64)
    if (slots < 0).any() or (frames < 1).any() or (frames > sample["horizon"][slots]).any():
        raise ValueError("Every requested row needs an observed player and a valid horizon.")
    if baseline.shape != (len(targets), 2) or not np.isfinite(baseline).all():
        raise ValueError("Baseline predictions must cover all requested coordinates.")
    sign = float(sample["sign"])
    canonical = baseline * sign + (1 - sign) / 2 * np.array([120, WIDTH])
    result = {
        **sample,
        "player": slots,
        "time": frames.astype(np.float32) / 10,
        "baseline": (canonical - sample["xy"][slots]).astype(np.float32),
        "keys": targets[KEYS].to_numpy(np.int64),
    }
    if truth is not None:
        if truth.shape != baseline.shape or not np.isfinite(truth).all():
            raise ValueError("Training truth must cover every finite coordinate.")
        result["truth"] = ((truth - baseline) * sign).astype(np.float32)
    return result


def reflect(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reflect all geometry and target residuals across the field's lateral axis."""
    result = {k: v.copy() for k, v in sample.items()}
    result["history"][..., ODD_CHANNELS] *= -1
    result["static"][..., ODD_STATIC] *= -1
    result["xy"][:, 1] = WIDTH - result["xy"][:, 1]
    result["velocity"][:, 1] *= -1
    for name in ("baseline", "truth"):
        if name in result:
            result[name][:, 1] *= -1
    return result
