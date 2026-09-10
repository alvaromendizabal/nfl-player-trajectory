"""Observed-only uncertain matchup affinities, not annotated coverage assignments.

Every eligible opponent contributes; no nearest-player identity is selected.
Fixed physical scales and an unmatched option express uncertainty without labels.
"""

from __future__ import annotations

import numpy as np

from nfl_trajectory.features import ROLES
from nfl_trajectory.temporal_data import CHANNELS, HISTORY

WINDOWS = (3, 8, 20)
DISTANCE_SCALE = 5.0
VELOCITY_SCALE = 3.0
NULL_COST = 2.0


def candidates(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], list[str]]:
    """Produce query-aligned static and temporal soft-matchup features.

    Padded values are removed before arithmetic. Missing observations are not
    compressed in time. The offense/defense relation excludes the passer, whose
    motion is not a route-coverage assignment. Affinities are heuristic weights;
    they must not be presented as calibrated man/zone or assignment probabilities.
    """
    seen = np.asarray(sample["observed"], dtype=bool)
    history = np.where(seen[..., None], sample["history"], 0).astype(np.float64)
    if history.shape[:2] != seen.shape or seen.shape[1] != HISTORY:
        raise ValueError("Expected synchronized observed-history tensors.")
    if not seen.any(axis=1).all() or not np.isfinite(history).all():
        raise ValueError("Every player needs finite observed tracking.")
    channel = {n: history[..., i] * s for i, (n, s) in enumerate(CHANNELS.items())}
    xy = np.stack([channel["relative_x"], channel["relative_y"]], -1)
    xy += sample["xy"][:, None]
    velocity = np.stack([channel["vx"], channel["vy"]], -1)
    delta = xy[None] - xy[:, None]
    dv = velocity[None] - velocity[:, None]
    distance = np.linalg.norm(delta, axis=-1)
    side, role = sample["side"], sample["role"]
    runner = role != ROLES.index("Passer")
    eligible = (side[:, None] != side[None]) & runner[:, None] & runner[None]
    valid = seen[:, None] & seen[None] & eligible[..., None]
    cost = (distance / DISTANCE_SCALE) ** 2 + (dv**2).sum(-1) / VELOCITY_SCALE**2
    unnormalized = np.exp(-np.minimum(cost, 80)) * valid
    denominator = np.exp(-NULL_COST) + unnormalized.sum(axis=1)
    weights = unnormalized / denominator[:, None]
    null = np.exp(-NULL_COST) / denominator
    mass = weights.sum(axis=1)
    # Full distributions include an unmatched option; entropy remains finite
    # even when there are no observed eligible opponents.
    entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=1)
    entropy -= null * np.log(np.maximum(null, 1e-30))
    closing = -(delta * dv).sum(-1) / np.maximum(distance, 0.1)
    vi, vj = velocity[:, None], velocity[None]
    coupling = (vi * vj).sum(-1) / np.maximum(
        np.linalg.norm(vi, axis=-1) * np.linalg.norm(vj, axis=-1), 0.1
    )

    def weighted(values: np.ndarray) -> np.ndarray:
        return (weights * values).sum(axis=1)

    target = role == ROLES.index("Targeted Receiver")
    series = {
        "match_mass": mass,
        "entropy": entropy,
        "target_mass": (weights * target[None, :, None]).sum(axis=1),
        "distance": weighted(distance) / 20,
        "closing": weighted(closing) / 10,
        "coupling": weighted(coupling),
        "dx": weighted(delta[..., 0]) / 20,
        "dy": weighted(delta[..., 1]) / 20,
        "dvx": weighted(dv[..., 0]) / 10,
        "dvy": weighted(dv[..., 1]) / 10,
    }
    index = np.arange(HISTORY)
    last = np.max(np.where(seen, index, -1), axis=1)
    row = np.arange(len(seen))
    columns: dict[str, np.ndarray] = {}
    for name, value in series.items():
        columns["soft_static__" + name] = value[row, last]
    # An age flag makes stale terminal observations explicit.
    columns["soft_static__age"] = (HISTORY - 1 - last) / 10
    for window in WINDOWS:
        mask = seen & (index > last[:, None] - window)
        count = mask.sum(axis=1)
        first = np.min(np.where(mask, index, HISTORY), axis=1)
        time = (index - last[:, None]) / 10
        centered = (time - (time * mask).sum(1)[:, None] / count[:, None]) * mask
        denom = np.maximum((centered**2).sum(1), 1e-6)
        prefix = f"soft_temporal__w{window:02d}_"
        columns[prefix + "observation_fraction"] = count / window
        columns[prefix + "peer_fraction"] = ((mass > 0) & mask).sum(1) / count
        for name, value in series.items():
            columns[prefix + name + "_mean"] = (value * mask).sum(1) / count
            columns[prefix + name + "_slope"] = (value * centered).sum(1) / denom
        # Total variation over all opponents detects changing responsibility
        # without assigning a discrete, tie-sensitive nearest-player identity.
        distribution = np.concatenate([weights, null[:, None]], axis=1)
        variation = np.abs(np.diff(distribution, axis=2)).sum(1) / 2
        adjacent = mask[:, :-1] & mask[:, 1:]
        columns[prefix + "turnover"] = (variation * adjacent).sum(1) / np.maximum(
            adjacent.sum(1), 1
        )
        change = distribution[row, :, last] - distribution[row, :, first]
        columns[prefix + "net_change"] = np.abs(change).sum(1) / 2
    names = list(columns)
    values = np.column_stack(list(columns.values()))[sample["player"]]
    if not np.isfinite(values).all():
        raise ValueError("Soft-coverage features became nonfinite.")
    return values.astype(np.float32), names, [n.split("__")[0] for n in names]
