"""Training-only motion labels; never used as observed or inference features.

This is a new reconstruction, not the missing historical supervision source.
Differences respect player identity and consecutive actual forecast frame IDs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from nfl_trajectory.temporal_data import STATIC_NAMES


def motion_targets(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return displacement derivatives in yards/second and yards/second².

    The terminal observed position anchors frame zero only when its age is zero.
    Missing predecessors produce a false mask, not interpolated supervision.
    Input order is preserved and no input is mutated.
    """
    keys = np.asarray(sample["keys"])
    slots = np.asarray(sample["player"])
    displacement = np.asarray(sample["baseline"], dtype=np.float64) + sample["truth"]
    n = len(keys)
    if keys.shape != (n, 4) or displacement.shape != (n, 2) or slots.shape != (n,):
        raise ValueError("Motion label arrays must align to forecast keys.")
    if not np.isfinite(displacement).all():
        raise ValueError("Motion labels require finite target coordinates.")
    if not np.issubdtype(keys.dtype, np.integer) or (keys[:, 3] < 1).any():
        raise ValueError("Frame IDs must be positive integers.")
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("Player slots must be integers.")
    ids = np.asarray(sample["ids"])
    if (slots < 0).any() or (slots >= len(ids)).any():
        raise ValueError("Invalid player slot.")
    if not np.array_equal(ids[slots], keys[:, 2]):
        raise ValueError("Player slots disagree with target identities.")
    if n and len(np.unique(keys[:, :2], axis=0)) != 1:
        raise ValueError("One sample must contain exactly one play.")
    lookup = {tuple(key): i for i, key in enumerate(keys)}
    if len(lookup) != n:
        raise ValueError("Duplicate target keys.")
    age = np.asarray(sample["static"])[:, STATIC_NAMES.index("last_observation_age")]
    if not np.isfinite(age).all() or (age < 0).any():
        raise ValueError("Observation ages must be finite and nonnegative.")
    velocity, acceleration = np.zeros((n, 2)), np.zeros((n, 2))
    velocity_mask, acceleration_mask = np.zeros(n, bool), np.zeros(n, bool)

    def prior(i: int, offset: int) -> np.ndarray | None:
        key = keys[i]
        frame = int(key[3]) - offset
        if frame == 0 and age[slots[i]] == 0:
            return np.zeros(2)
        j = lookup.get((*key[:3], frame))
        return None if j is None else displacement[j]

    for i in range(n):
        previous = prior(i, 1)
        if previous is None:
            continue
        velocity[i] = (displacement[i] - previous) * 10
        velocity_mask[i] = True
        previous_two = prior(i, 2)
        if previous_two is not None:
            acceleration[i] = (displacement[i] - 2 * previous + previous_two) * 100
            acceleration_mask[i] = True
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "velocity_mask": velocity_mask,
        "acceleration_mask": acceleration_mask,
    }


def fit_motion_scales(samples: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Fit one shared x/y RMS per family, rejecting any non-training sample."""
    sse = {name: 0.0 for name in ("velocity", "acceleration")}
    count = dict.fromkeys(sse, 0)
    for sample in samples:
        if str(sample["split"]) != "train":
            raise ValueError("Motion scales accept training samples only.")
        labels = motion_targets(sample)
        for name in sse:
            values = labels[name][labels[name + "_mask"]]
            sse[name] += float(np.square(values).sum())
            count[name] += values.size
    if not all(count.values()):
        raise ValueError("Each motion family needs valid training labels.")
    return {name: max(0.1, float(np.sqrt(sse[name] / count[name]))) for name in sse}
