"""Synchronized observed interaction tensors; not inferred coverage labels.

Retain the player-pair and time axes instead of averaging them into heuristic
soft-affinity columns. Fixed physical scales require no label-dependent fit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

NAMES = (
    "dx",
    "dy",
    "dvx",
    "dvy",
    "distance",
    "closing_speed",
    "lateral_speed",
    "velocity_alignment",
    "bearing_rate",
    "separation_rate",
    "same_side",
)
SCALES = np.array([20, 20, 10, 10, 20, 10, 10, 1, 10, 10, 1], dtype=np.float64)
ODD_LATERAL = (1, 3, 6, 8)


def build_edges(
    position: np.ndarray,
    velocity: np.ndarray,
    observed: np.ndarray,
    side: np.ndarray,
    seconds_per_frame: float = 0.1,
) -> dict[str, np.ndarray]:
    """Return [source, destination, observed frame, channel] with explicit masks.

    Missing observations are never forward-filled or compressed in time. Rates
    use adjacent observed frames only. Coincident and stationary cases have
    channel-specific validity, not fabricated geometric measurements.
    """
    seen = np.asarray(observed)
    xy, v = np.asarray(position), np.asarray(velocity)
    teams = np.asarray(side)
    if seen.dtype != np.bool_ or seen.ndim != 2:
        raise ValueError("observed must be a two-dimensional boolean mask")
    n, t = seen.shape
    if not (1 <= n <= 22 and 1 <= t <= 20):
        raise ValueError("Require 1..22 players and 1..20 observed frame slots")
    if xy.shape != (n, t, 2) or v.shape != xy.shape:
        raise ValueError("Positions and velocities must have shape [players, frames, 2]")
    if teams.shape != (n,) or not np.isin(teams, [0, 1]).all():
        raise ValueError("side must contain exactly one 0/1 value per player")
    if not seen.any(axis=1).all():
        raise ValueError("Each player must have at least one observed frame")
    if not np.isfinite(seconds_per_frame) or seconds_per_frame <= 0:
        raise ValueError("seconds_per_frame must be finite and positive")
    if not np.isfinite(xy[seen]).all() or not np.isfinite(v[seen]).all():
        raise ValueError("Observed positions and velocities must be finite")
    xy = np.where(seen[..., None], xy, 0).astype(np.float64)
    v = np.where(seen[..., None], v, 0).astype(np.float64)
    pair = seen[:, None] & seen[None, :]
    pair &= ~np.eye(n, dtype=bool)[..., None]
    delta, dv = xy[None, :] - xy[:, None], v[None, :] - v[:, None]
    distance = np.linalg.norm(delta, axis=-1)
    radial = (delta * dv).sum(axis=-1)
    cross = delta[..., 0] * dv[..., 1] - delta[..., 1] * dv[..., 0]
    speed = np.linalg.norm(v, axis=-1)
    denominator = speed[:, None] * speed[None, :]
    alignment = (v[:, None] * v[None, :]).sum(axis=-1) / np.maximum(denominator, 1e-12)
    adjacent = np.zeros_like(pair)
    adjacent[..., 1:] = pair[..., 1:] & pair[..., :-1]
    angle_rate, distance_rate = np.zeros_like(distance), np.zeros_like(distance)
    previous, current = delta[..., :-1, :], delta[..., 1:, :]
    angular_cross = previous[..., 0] * current[..., 1] - previous[..., 1] * current[..., 0]
    angular_dot = (previous * current).sum(axis=-1)
    angle_rate[..., 1:] = np.arctan2(angular_cross, angular_dot) / seconds_per_frame
    distance_rate[..., 1:] = np.diff(distance, axis=-1) / seconds_per_frame
    same = np.broadcast_to((teams[:, None] == teams[None, :])[..., None], pair.shape)
    values = np.stack(
        [
            delta[..., 0],
            delta[..., 1],
            dv[..., 0],
            dv[..., 1],
            distance,
            -radial / np.maximum(distance, 0.1),
            cross / np.maximum(distance, 0.1),
            np.clip(alignment, -1, 1),
            angle_rate,
            distance_rate,
            same,
        ],
        axis=-1,
    )
    valid = np.repeat(pair[..., None], len(NAMES), axis=-1)
    valid[..., 5] &= distance >= 0.1
    valid[..., 6] &= distance >= 0.1
    valid[..., 7] &= (speed[:, None] >= 1e-3) & (speed[None, :] >= 1e-3)
    bearing_valid = adjacent & (distance >= 0.1)
    bearing_valid[..., 1:] &= distance[..., :-1] >= 0.1
    valid[..., 8], valid[..., 9] = bearing_valid, adjacent
    values = np.where(valid, values / SCALES, 0).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Interaction tensor contains nonfinite values")
    return {"values": values, "valid": valid, "pair_valid": pair, "adjacent": adjacent}


def from_sample(sample: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Adapt the existing observed cache; do not read truth, baseline or query rows."""
    from nfl_trajectory.temporal_data import CHANNELS

    seen = np.asarray(sample["observed"])
    history = np.asarray(sample["history"])
    anchors = np.asarray(sample["xy"])
    if seen.ndim != 2 or history.shape != (*seen.shape, len(CHANNELS)):
        raise ValueError("Observed cache channel layout differs from the project contract")
    if anchors.shape != (len(seen), 2) or not np.isfinite(anchors).all():
        raise ValueError("Finite observed terminal anchors are required")
    names = list(CHANNELS)
    indices = [names.index(name) for name in ("relative_x", "relative_y", "vx", "vy")]
    safe = np.where(seen[..., None], history[..., indices], 0).astype(np.float64)
    scales = [CHANNELS[name] for name in ("relative_x", "relative_y", "vx", "vy")]
    physical = safe * np.asarray(scales)
    return build_edges(
        physical[..., :2] + anchors[:, None], physical[..., 2:], seen, sample["side"]
    )
