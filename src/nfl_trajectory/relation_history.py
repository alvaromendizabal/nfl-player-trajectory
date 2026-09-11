"""Observed-only, frame-aligned player relationships for a future matched ablation.

This opt-in representation does not modify any existing model or experiment.
Inputs use temporal_data's common 20-frame grid and fixed physical-unit scales.
Each directed edge is neighbour j relative to focal player i. A joint observation
mask prevents comparing players observed at different times. No target, outcome,
player ID, or fitted statistic is accepted by this interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_trajectory.features import ROLES
from nfl_trajectory.temporal_data import CHANNELS, HISTORY

RELATION_NAMES = (
    "dx_over_20yd",
    "dy_over_20yd",
    "dvx_over_10yd_s",
    "dvy_over_10yd_s",
    "distance_over_20yd",
    "closing_over_10yd_s",
    "separation_supported",
    "velocity_alignment",
    "alignment_supported",
    "same_side",
    "neighbour_targeted_receiver",
    "neighbour_passer",
)
ODD_RELATIONS = (1, 3)
GEOMETRY_COLUMNS = (0, 1, 4, 6)
MOTION_COLUMNS = (2, 3, 5, 7, 8)
CONTEXT_COLUMNS = (9, 10, 11)
# Numerical support thresholds, not validation-tuned football cutoffs.
MIN_SEPARATION_YARDS = 1e-6
MIN_SPEED_YARDS_PER_SECOND = 1e-6


@dataclass(frozen=True)
class RelationHistory:
    """Fixed-size directed features and explicit support; all arrays are owned."""

    values: np.ndarray  # [focal, neighbour, observed frame, feature]
    observed: np.ndarray  # [focal, neighbour, observed frame]; self-edges are absent
    frame_seconds: np.ndarray  # [-1.9, ..., 0.0], relative to the observed cutoff

    def terminal_view(self) -> RelationHistory:
        """Retain the last jointly observed frame, without shifting it to time zero.

        Shape and channel count remain unchanged for a future capacity-matched
        history-versus-terminal ablation. Missing pairs stay missing. Selecting
        independent last observations for each player would create false geometry.
        """
        index = np.arange(HISTORY)
        last = np.where(self.observed, index, -1).max(axis=-1)
        keep = self.observed & (index == last[..., None])
        return RelationHistory(
            np.where(keep[..., None], self.values, 0.0),
            keep.copy(),
            self.frame_seconds.copy(),
        )


def pair_history(
    history: np.ndarray,
    observed: np.ndarray,
    side: np.ndarray,
    role: np.ndarray,
) -> RelationHistory:
    """Construct relationships only where both slots exist at the same frame.

    Ball-relative offsets share an organizer-supplied landmark: offset_i -
    offset_j equals position_j - position_i. Per-player displacement histories
    have different origins and must not be subtracted across players. The supplied
    landing point cancels algebraically; it is not an observed future ball path.
    Build per minibatch, not as a corpus-sized dense tensor cache.
    """
    history = np.asarray(history)
    observed = np.asarray(observed)
    side, role = np.asarray(side), np.asarray(role)
    channels = list(CHANNELS)
    if history.ndim != 3 or history.shape[1:] != (HISTORY, len(channels)):
        raise ValueError("Expected player by 20-frame by canonical-channel history.")
    players = history.shape[0]
    if not 1 <= players <= 22 or history.dtype.kind != "f":
        raise ValueError("One to 22 player slots with floating-point history are required.")
    if observed.shape != (players, HISTORY) or observed.dtype.kind != "b":
        raise ValueError("A boolean observation mask must match the player/frame grid.")
    for name, values, limit in (("side", side, 2), ("role", role, len(ROLES))):
        if (
            values.shape != (players,)
            or values.dtype.kind not in "iu"
            or (values < 0).any()
            or (values >= limit).any()
        ):
            raise ValueError(f"Invalid categorical {name}; explicit canonical slots required.")
    names = ("ball_dx", "ball_dy", "vx", "vy")
    selected = history[..., [channels.index(name) for name in names]]
    active = selected[observed]
    if not np.isfinite(active).all() or (np.abs(active) > np.finfo(np.float32).max).any():
        raise ValueError("Supported observed relationship inputs must be finite.")
    # Mask before any arithmetic: absent rows may contain NaN/Inf without poisoning edges.
    clean = np.where(observed[..., None], selected, 0.0).astype(np.float64)
    clean *= np.array([CHANNELS[name] for name in names], dtype=np.float64)
    offset, velocity = clean[..., :2], clean[..., 2:]
    delta = offset[:, None] - offset[None, :]
    dv = velocity[None, :] - velocity[:, None]
    distance = np.linalg.norm(delta, axis=-1)
    separated = distance >= MIN_SEPARATION_YARDS
    closing = np.zeros_like(distance)
    np.divide(-(delta * dv).sum(axis=-1), distance, out=closing, where=separated)
    speeds = np.linalg.norm(velocity, axis=-1)
    moving = speeds >= MIN_SPEED_YARDS_PER_SECOND
    alignment_supported = moving[:, None] & moving[None, :]
    alignment = np.zeros_like(distance)
    np.divide(
        (velocity[:, None] * velocity[None, :]).sum(axis=-1),
        speeds[:, None] * speeds[None, :],
        out=alignment,
        where=alignment_supported,
    )
    shape = distance.shape
    neighbour_receiver = np.broadcast_to(
        (role == ROLES.index("Targeted Receiver"))[None, :, None], shape
    )
    neighbour_passer = np.broadcast_to((role == ROLES.index("Passer"))[None, :, None], shape)
    same_side = np.broadcast_to((side[:, None] == side[None, :])[..., None], shape)
    values = np.stack(
        [
            delta[..., 0] / 20,
            delta[..., 1] / 20,
            dv[..., 0] / 10,
            dv[..., 1] / 10,
            distance / 20,
            closing / 10,
            separated,
            np.clip(alignment, -1.0, 1.0),
            alignment_supported,
            same_side,
            neighbour_receiver,
            neighbour_passer,
        ],
        axis=-1,
    )
    joint = observed[:, None] & observed[None, :]
    joint &= ~np.eye(players, dtype=bool)[..., None]
    values = np.where(joint[..., None], values, 0.0)
    if not np.isfinite(values).all() or np.abs(values).max() > np.finfo(np.float32).max:
        raise ValueError("Relationship features exceed finite float32 representation.")
    return RelationHistory(
        values.astype(np.float32),
        joint.copy(),
        np.arange(1 - HISTORY, 1, dtype=np.float32) / 10,
    )
