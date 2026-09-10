"""Observed-only, physically interpretable candidates for a frozen temporal model.

These are hypotheses, not hard constraints: an intended receiver need not reach
the supplied landing point and a nearby defender need not be the assigned marker.
Every time derivative uses synchronized observed frames. No outcome, identity
value, estimated future receiver position, or external annotation is consumed.
"""

from __future__ import annotations

import numpy as np

from nfl_trajectory.features import ROLES, WIDTH
from nfl_trajectory.temporal_data import CHANNELS, HISTORY

FAMILIES = ("motion_state", "arrival_constraints", "coverage_dynamics", "field_geometry")
WINDOWS = (3, 8, 20)


def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(_norm(x)[..., None], 0.1)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(axis=-1)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def candidates(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return one row per forecast request, with a deterministic audited schema.

    Column names carry their family. Fixed units prevent squared-time features
    from dwarfing directional cosines; any later standardizer must be train-only.
    Vector signs are recomputed from the sample, so lateral reflection is exact.
    """
    history = sample["history"].astype(np.float64)
    seen = sample["observed"]
    p = len(sample["ids"])
    columns: dict[str, np.ndarray] = {}
    source = {name: history[..., i] * scale for i, (name, scale) in enumerate(CHANNELS.items())}
    last = np.max(np.where(seen, np.arange(HISTORY)[None], -1), axis=1)
    if (last < 0).any():
        raise ValueError("Every real player needs an observed frame.")
    rows = np.arange(p)
    xy = sample["xy"].astype(float)
    position = np.stack([source["relative_x"], source["relative_y"]], -1) + xy[:, None]
    velocity = np.stack([source["vx"], source["vy"]], -1)
    heading = np.stack([source["dir_sin"], source["dir_cos"]], -1)
    facing = np.stack([source["o_sin"], source["o_cos"]], -1)
    supplied = heading * source["s"][..., None]
    v, reported = velocity[rows, last], supplied[rows, last]
    direction, orientation = heading[rows, last], facing[rows, last]
    ball = sample["static"][:, 2:4].astype(float) * 20
    horizon = sample["horizon"].astype(float) / 10
    ball_unit = _unit(ball)
    distance = _norm(ball)
    slot, time = sample["player"], sample["time"].astype(float)
    fraction = time / horizon[slot]

    def add(family: str, name: str, value: np.ndarray, scale: float = 1) -> None:
        array = np.asarray(value, dtype=float) / scale
        if array.shape != (p,):
            raise ValueError("A player-level domain candidate has the wrong shape.")
        columns[f"{family}__{name}"] = array[slot]

    def vector(family: str, name: str, value: np.ndarray, scale: float = 1) -> None:
        for i, axis in enumerate(("x", "y")):
            add(family, name + "_" + axis, value[:, i], scale)

    def query(family: str, name: str, value: np.ndarray, scale: float = 1) -> None:
        if value.shape != (len(slot), 2):
            raise ValueError("A query-level domain vector has the wrong shape.")
        for i, axis in enumerate(("x", "y")):
            columns[f"{family}__{name}_{axis}"] = value[:, i] / scale

    family = "motion_state"
    vector(family, "supplied_velocity", reported, 10)
    vector(family, "telemetry_velocity_gap", reported - v, 10)
    add(family, "velocity_gap_norm", _norm(reported - v), 10)
    add(family, "heading_facing_dot", _dot(direction, orientation))
    add(family, "heading_facing_cross", _cross(direction, orientation))
    add(family, "facing_landing_dot", _dot(orientation, ball_unit))
    add(family, "facing_landing_cross", _cross(orientation, ball_unit))
    add(family, "backpedal", _dot(_unit(v), orientation))
    smooth: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        # Regression on the actual observed time indices, not compressed row offsets.
        mask = seen & (np.arange(HISTORY)[None] > last[:, None] - window)
        weight = mask.astype(float)
        count = weight.sum(axis=1).clip(1)
        t = (np.arange(HISTORY)[None] - last[:, None]) / 10
        mean_t = (t * weight).sum(axis=1) / count
        centered_t = (t - mean_t[:, None]) * weight
        denominator = (centered_t**2).sum(axis=1).clip(1e-6)
        slope = (position * centered_t[..., None]).sum(axis=1) / denominator[:, None]
        mean_v = (velocity * weight[..., None]).sum(axis=1) / count[:, None]
        mean_reported = (supplied * weight[..., None]).sum(axis=1) / count[:, None]
        trend = (supplied * centered_t[..., None]).sum(axis=1) / denominator[:, None]
        smooth[window] = trend
        vector(family, f"w{window:02d}_position_slope", slope, 10)
        vector(family, f"w{window:02d}_velocity_mean", mean_v, 10)
        vector(family, f"w{window:02d}_supplied_mean", mean_reported, 10)
        vector(family, f"w{window:02d}_supplied_trend", trend, 10)
        rms = np.sqrt(((_norm(velocity - supplied) ** 2) * weight).sum(axis=1) / count)
        add(family, f"w{window:02d}_telemetry_rms", rms, 10)
        add(family, f"w{window:02d}_coverage", count / window)
        add(family, f"w{window:02d}_braking", _dot(trend, _unit(reported)), 10)
        add(family, f"w{window:02d}_turning", _cross(_unit(reported), trend), 10)
        first = np.min(np.where(mask, np.arange(HISTORY)[None], HISTORY), axis=1)
        displacement = position[rows, last] - position[rows, first]
        adjacent = mask[:, 1:] & mask[:, :-1]
        path_length = (_norm(np.diff(position, axis=1)) * adjacent).sum(axis=1)
        add(family, f"w{window:02d}_path_efficiency", _norm(displacement) / path_length.clip(0.1))
    query(
        family,
        "supplied_cv_minus_baseline",
        reported[slot] * time[:, None] - sample["baseline"],
        10,
    )
    for decay in (0.25, 0.75, 1.5):
        integrated = decay * (time - decay * (-np.expm1(-time / decay)))
        displacement = reported[slot] * time[:, None] + smooth[8][slot] * integrated[:, None]
        query(family, f"damped_{decay:g}_minus_baseline", displacement - sample["baseline"], 10)

    family = "arrival_constraints"
    required = ball / horizon[:, None]
    gap = ball - reported * horizon[:, None]
    acceleration = 2 * gap / horizon[:, None] ** 2
    vector(family, "required_velocity", required, 10)
    vector(family, "velocity_deficit", required - reported, 10)
    vector(family, "arrival_gap", gap, 20)
    vector(family, "required_acceleration", acceleration, 20)
    add(family, "required_speed", distance / horizon, 10)
    add(family, "speed_margin", _norm(reported) - distance / horizon, 10)
    add(family, "radial_deficit", _dot(required - reported, ball_unit), 10)
    add(family, "lateral_deficit", _cross(ball_unit, required - reported), 10)
    angle = np.arctan2(_cross(direction, ball_unit), _dot(direction, ball_unit))
    add(family, "signed_turn_angle", angle, np.pi)
    add(family, "turn_rate_required", angle / horizon, np.pi)
    add(family, "time_at_current_speed", distance / _norm(reported).clip(0.5), 5)
    add(family, "current_speed_time_margin", horizon - distance / _norm(reported).clip(0.5), 5)
    for maximum in (6, 9, 12):
        add(family, f"speed_{maximum}_arrival_margin", maximum * horizon - distance, 20)
    for delay in (0.15, 0.3):
        remaining = np.maximum(horizon - delay, 0.1)
        deficit = (ball - reported * np.minimum(horizon, delay)[:, None]) / remaining[:, None]
        vector(family, f"reaction_{delay:g}_velocity", deficit, 10)
    # Three soft boundary-value hypotheses, never forced on a player or a label.
    f = fraction[:, None]
    cv = reported[slot] * time[:, None]
    query(family, "quadratic_bridge_minus_cv", gap[slot] * f**2, 10)
    hermite = ball[slot] * (3 * f**2 - 2 * f**3)
    hermite += reported[slot] * horizon[slot, None] * (f - 2 * f**2 + f**3)
    query(family, "stopping_bridge_minus_cv", hermite - cv, 10)
    minimum_jerk = (10 * f**3 - 15 * f**4 + 6 * f**5) * ball[slot]
    query(family, "rest_bridge_minus_cv", minimum_jerk - cv, 10)

    family = "coverage_dynamics"
    delta = xy[None] - xy[:, None]
    distance_matrix = _norm(delta)
    opposite = sample["side"][None] != sample["side"][:, None]
    available = opposite & (last[None] == last[:, None])
    anchors = []
    for rank in (0, 1):
        # Tie-break with observed geometry; no identity or arbitrary input ordering.
        indices = np.empty(p, dtype=int)
        valid = available.sum(axis=1) > rank
        for i in range(p):
            ordered = np.lexsort(
                (
                    v[:, 1],
                    v[:, 0],
                    xy[:, 1],
                    xy[:, 0],
                    np.where(available[i], distance_matrix[i], np.inf),
                )
            )
            indices[i] = ordered[rank] if valid[i] else i
        anchors.append((f"opponent{rank + 1}", indices, valid))
    for role_name, prefix in (("Targeted Receiver", "receiver"), ("Passer", "passer")):
        match = np.flatnonzero(sample["role"] == ROLES.index(role_name))
        if len(match) > 1:
            raise ValueError("A play must have at most one role anchor.")
        anchors.append(
            (prefix, np.full(p, match[0] if len(match) else 0), np.full(p, bool(len(match))))
        )
    for prefix, index, valid in anchors:
        relative = position[index] - position
        dv = supplied[index] - supplied
        separation = _norm(relative)
        synchronized = seen & seen[index] & valid[:, None]
        coupling = _dot(_unit(supplied), _unit(supplied[index]))
        closing = -_dot(relative, dv) / separation.clip(0.1)
        add(family, prefix + "_present", valid.astype(float))
        for window in WINDOWS:
            mask = synchronized & (np.arange(HISTORY)[None] > last[:, None] - window)
            count = mask.sum(axis=1).clip(1)
            first = np.min(np.where(mask, np.arange(HISTORY)[None], HISTORY - 1), axis=1)
            final = np.max(np.where(mask, np.arange(HISTORY)[None], 0), axis=1)
            has = mask.any(axis=1)
            for name, values, scale in (
                ("distance", separation, 20),
                ("closing", closing, 10),
                ("coupling", coupling, 1),
            ):
                mean = (values * mask).sum(axis=1) / count
                std = np.sqrt((((values - mean[:, None]) ** 2) * mask).sum(axis=1) / count)
                change = (values[rows, final] - values[rows, first]) * has
                for statistic, value in (("mean", mean), ("std", std), ("change", change)):
                    add(family, f"{prefix}_w{window:02d}_{name}_{statistic}", value, scale)
            add(family, f"{prefix}_w{window:02d}_coverage", mask.sum(axis=1) / window)
        vector(family, prefix + "_terminal_delta", delta[rows, index] * valid[:, None], 20)
        vector(family, prefix + "_terminal_dv", (reported[index] - reported) * valid[:, None], 10)
    # Who is closest may change; this is a geometric handoff proxy, not a coverage label.
    d_history = _norm(position[None] - position[:, None])
    pair_seen = seen[:, None] & seen[None] & opposite[..., None]
    nearest = np.argmin(np.where(pair_seen, d_history, np.inf), axis=1)
    any_peer = pair_seen.any(axis=1)
    pairs = any_peer[:, 1:] & any_peer[:, :-1]
    changes = ((nearest[:, 1:] != nearest[:, :-1]) & pairs).sum(axis=1)
    add(family, "nearest_changes", changes / pairs.sum(axis=1).clip(1))
    for radius in (2, 5, 10):
        close = (d_history < radius) & pair_seen
        add(family, f"within_{radius}_mean", close.sum(axis=1).mean(axis=1), 4)
        add(
            family, f"within_{radius}_change", close.sum(axis=1)[:, -1] - close.sum(axis=1)[:, 0], 4
        )

    family = "field_geometry"
    add(family, "left_sideline", xy[:, 1], WIDTH)
    add(family, "right_sideline", WIDTH - xy[:, 1], WIDTH)
    add(family, "endzone", 110 - xy[:, 0], 100)
    add(family, "landing_left_sideline", xy[:, 1] + ball[:, 1], WIDTH)
    add(family, "landing_right_sideline", WIDTH - xy[:, 1] - ball[:, 1], WIDTH)
    towards = np.where(reported[:, 1] >= 0, WIDTH - xy[:, 1], xy[:, 1])
    add(family, "time_to_sideline", towards / np.abs(reported[:, 1]).clip(0.5), 10)
    passer = np.flatnonzero(sample["role"] == ROLES.index("Passer"))
    present = np.full(p, bool(len(passer)), dtype=float)
    origin = xy[passer[0]] if len(passer) else np.zeros(2)
    ray = xy + ball - origin
    ray_unit = _unit(ray)
    relative = xy - origin
    add(family, "passer_present", present)
    add(family, "pass_axis_fraction", _dot(relative, ray_unit) / _norm(ray).clip(1) * present)
    add(family, "pass_axis_lateral", _cross(ray_unit, relative) * present, 20)
    add(family, "pass_axis_velocity", _dot(reported, ray_unit) * present, 10)
    add(family, "pass_axis_cross_velocity", _cross(ray_unit, reported) * present, 10)
    add(family, "pass_distance", _norm(ray) * present, 40)
    add(family, "implied_ball_speed", _norm(ray) / horizon * present, 30)
    # Explicit role/fraction interactions: conditional geometry has different meanings.
    for name, values in list(columns.items()):
        if name.startswith(("arrival_constraints__", "field_geometry__")):
            for role in ("Targeted Receiver", "Defensive Coverage"):
                columns[name + "__" + role.replace(" ", "_")] = values * (
                    sample["role"][slot] == ROLES.index(role)
                )
    names = list(columns)
    matrix = np.column_stack(list(columns.values())).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Domain candidate matrix contains nonfinite values.")
    return matrix, names, [name.split("__", 1)[0] for name in names]
