"""A disjoint, frozen decomposition of the observed motion candidate family."""

from __future__ import annotations

import numpy as np

GROUPS = ("instant_state", "facing", "smoothed_state", "maneuvers", "propagation")
ARMS = (*GROUPS, *("without_" + group for group in GROUPS))


def feature_group(name: str) -> str:
    if not name.startswith("motion_state__"):
        return "control" if name.startswith("control__") else "excluded"
    name = name.removeprefix("motion_state__")
    if "minus_baseline" in name:
        return "propagation"
    if "facing" in name or name == "backpedal":
        return "facing"
    if name.startswith("w"):
        return (
            "maneuvers"
            if name.endswith(("braking", "turning", "path_efficiency"))
            else "smoothed_state"
        )
    if (
        name.startswith(("supplied_velocity", "telemetry_velocity_gap"))
        or name == "velocity_gap_norm"
    ):
        return "instant_state"
    raise ValueError("An unclassified motion feature entered the experiment: " + name)


def keep_columns(names: list[str], retained: list[bool], arm: str) -> np.ndarray:
    if arm not in ARMS:
        raise ValueError("Unknown motion experiment arm.")
    allowed = {"control"}
    if arm.startswith("without_"):
        allowed.update(set(GROUPS) - {arm.removeprefix("without_")})
    else:
        allowed.add(arm)
    return np.array(
        [r and feature_group(n) in allowed for n, r in zip(names, retained, strict=True)]
    )
