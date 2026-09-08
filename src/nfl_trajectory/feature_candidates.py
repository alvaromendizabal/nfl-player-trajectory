"""Additional pre-throw hypotheses with explicit provenance and availability.

The original bank is unchanged. These candidates are research inputs, not a
promoted inference representation. Historical encodings are supplied separately
by a fold-local, strictly previous-date encoder.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.features import (
    CHANNELS,
    GATE_INPUTS,
    NEIGHBOR_GROUPS,
    ROLES,
    WINDOWS,
    PlayerFeatures,
    feature_catalog,
)
from nfl_trajectory.motion import ENTITY, KEYS

HISTORY_NAMES = [
    f"prior__{key}__{stat}"
    for key in ("player", "role")
    for stat in ("log_count", "cold_start", "velocity_error_x", "velocity_error_y", "dispersion")
]
RATIONALE = {
    "history_lags": "Recent observed paths resolve motion direction and changes.",
    "history_summaries": "Multiple observed windows measure motion and landing-relative stability.",
    "availability": "History and telemetry masks distinguish missing inputs from physical zeros.",
    "interactions": "Nearby players and receiver/passer anchors constrain feasible movement.",
    "context": "Role, field position and supplied forecast horizon change movement constraints.",
    "forecast": "Known future frame indices parameterize a trajectory without using its positions.",
    "forecast_interactions": "Forecast time modulates observed motion and relative geometry.",
    "multiscale": "Short-minus-long summaries measure changes in tempo, turning and dispersion.",
    "nonlinear": "Bounded and signed-log responses distinguish regimes without a complex model.",
    "role_response": "The same observed geometry can imply different movement for different roles.",
    "projected_geometry": "Projected separation and landing-axis geometry describe congestion.",
    "player_history": "Earlier training games estimate motion bias with explicit cold starts.",
}


@lru_cache(maxsize=1)
def specifications() -> tuple[tuple[str, str, str, tuple[str, ...]], ...]:
    """Each tuple is name, family, operation, dependencies; names are data independent."""
    records: list[tuple[str, str, str, tuple[str, ...]]] = []
    for channel in CHANNELS:
        for short, long in zip(WINDOWS[:-1], WINDOWS[1:], strict=True):
            for stat in ("mean", "std", "slope"):
                records.append(
                    (
                        f"contrast{short:02d}_{long:02d}__{channel}__{stat}",
                        "multiscale",
                        "difference",
                        (f"w{short:02d}__{channel}__{stat}", f"w{long:02d}__{channel}__{stat}"),
                    )
                )
        for transform in ("signed_log", "bounded", "energy"):
            records.append(
                (f"{transform}__{channel}", "nonlinear", transform, (f"lag00__{channel}",))
            )
    for role in ROLES[:-1]:
        for base in GATE_INPUTS:
            for gate in ("level", "time", "fraction"):
                records.append(
                    (
                        f"role_response__{role}__{gate}__{base}",
                        "role_response",
                        gate,
                        (f"role__{role}", base),
                    )
                )
    for group, count in NEIGHBOR_GROUPS:
        for slot in range(1, count + 1):
            prefix = f"{group}{slot}"
            for operation in (
                "projected_dx",
                "projected_dy",
                "projected_distance",
                "projected_closing",
                "landing_longitudinal",
                "landing_lateral",
                "arrival_distance",
            ):
                records.append(
                    (
                        f"geometry__{prefix}__{operation}",
                        "projected_geometry",
                        operation,
                        tuple(f"{prefix}__{c}" for c in ("dx", "dy", "dvx", "dvy", "present")),
                    )
                )
    return tuple(records)


def research_catalog() -> pd.DataFrame:
    base = feature_catalog().copy()
    extra = pd.DataFrame(
        [(name, family, "research") for name, family, _, _ in specifications()],
        columns=["feature", "family", "signal"],
    )
    history = pd.DataFrame(
        [(name, "player_history", "historical_target") for name in HISTORY_NAMES],
        columns=extra.columns,
    )
    catalog = pd.concat([base, extra, history], ignore_index=True)
    catalog["rationale"] = catalog.family.map(RATIONALE)
    catalog["availability"] = np.where(
        catalog.family.eq("player_history"),
        "strictly earlier dates within fold training; frozen for evaluation",
        "pre-throw tracking plus organizer-supplied landing point, horizon and role",
    )
    catalog["provenance"] = np.where(
        catalog.family.eq("player_history"),
        "prior training outputs minus constant-velocity forecast, grouped by player or role",
        "organizer input tracking; deterministic transforms; output coordinates excluded",
    )
    if catalog.feature.duplicated().any() or catalog.rationale.isna().any():
        raise ValueError("Research catalog must have unique names and documented rationale.")
    return catalog


def candidate_matrix(
    bank: PlayerFeatures,
    targets: pd.DataFrame,
    history: np.ndarray,
    columns: list[str] | None = None,
) -> np.ndarray:
    """Expand a bounded batch; future coordinates in targets are deliberately ignored."""
    catalog_names = research_catalog().feature.tolist()
    chosen = catalog_names if columns is None else columns
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(catalog_names):
        raise ValueError("Unknown or duplicate research feature names.")
    if history.shape != (len(targets), len(HISTORY_NAMES)) or not np.isfinite(history).all():
        raise ValueError("History encoding shape or finite-value contract failed.")
    if len(targets) * len(chosen) * 4 > 256 * 1024**2:
        raise MemoryError("Research expansion exceeds 256 MiB; use bounded batches.")
    specs = {r[0]: r for r in specifications()}
    original = set(feature_catalog().feature)
    needed = set(chosen) & original
    for name in set(chosen) & set(specs):
        needed.update(specs[name][3])
    if any(name.startswith("geometry__") for name in chosen):
        needed.update(("lag00__ball_ux", "lag00__ball_uy", "horizon_seconds"))
    needed.update(("time_seconds", "fraction"))
    ordered = sorted(needed)
    raw = bank.matrix(targets[KEYS], ordered)
    lookup = {name: raw[:, i] for i, name in enumerate(ordered)}
    t, q = lookup["time_seconds"], lookup["fraction"]
    output = np.empty((len(targets), len(chosen)), dtype=np.float32)
    for i, name in enumerate(chosen):
        if name in original:
            value = lookup[name]
        elif name in HISTORY_NAMES:
            value = history[:, HISTORY_NAMES.index(name)]
        else:
            _, _, op, dependencies = specs[name]
            v = [lookup[k] for k in dependencies]
            if op == "difference":
                value = v[0] - v[1]
            elif op == "signed_log":
                value = np.sign(v[0]) * np.log1p(np.abs(v[0]))
            elif op == "bounded":
                value = v[0] / (1 + np.abs(v[0]))
            elif op == "energy":
                bounded = v[0] / (1 + np.abs(v[0]))
                value = bounded**2
            elif op in ("level", "time", "fraction"):
                value = v[0] * v[1] * {"level": 1.0, "time": t, "fraction": q}[op]
            else:
                dx, dy, vx, vy, present = v
                px, py = dx + vx * t, dy + vy * t
                distance = np.hypot(px, py)
                if op == "projected_dx":
                    value = px
                elif op == "projected_dy":
                    value = py
                elif op == "projected_distance":
                    value = distance
                elif op == "projected_closing":
                    value = -(px * vx + py * vy) / np.maximum(distance, 1e-6)
                elif op == "landing_longitudinal":
                    value = dx * lookup["lag00__ball_ux"] + dy * lookup["lag00__ball_uy"]
                elif op == "landing_lateral":
                    value = dx * lookup["lag00__ball_uy"] - dy * lookup["lag00__ball_ux"]
                else:
                    h = lookup["horizon_seconds"]
                    value = np.hypot(dx + vx * h, dy + vy * h)
                value = value * present
        output[:, i] = value
    if not np.isfinite(output).all():
        raise ValueError("Research candidates must be finite.")
    return output


def historical_encodings(
    observations: pd.DataFrame, training_games: set[int], smoothing: float = 20.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode each row before updating its date; evaluation labels never enter state.

    Observations are one row per player/play, including velocity-error summaries.
    Evaluation rows see the final training state. Whole-date batching excludes
    same-day outcomes when kickoff ordering is unavailable. The prior is zero,
    not a mean computed with future games. No learned baseline enters targets.
    """
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Positive smoothing is required.")
    required = ENTITY + ["player_role", "error_x", "error_y"]
    if not set(required).issubset(observations) or observations[ENTITY].duplicated().any():
        raise ValueError("Unique historical entity observations are required.")
    training = observations.game_id.isin(training_games).to_numpy()
    if not training.any():
        raise ValueError("Historical encoder needs nonempty training games.")
    dates = observations.game_id.to_numpy(np.int64) // 100
    if (~training).any() and dates[training].max() >= dates[~training].min():
        raise ValueError("Evaluation history must strictly follow training dates.")
    # Deliberately inspect outcomes only on training rows.
    if not np.isfinite(observations.loc[training, ["error_x", "error_y"]].to_numpy()).all():
        raise ValueError("Historical training outcomes must be finite.")
    result = np.zeros((len(observations), len(HISTORY_NAMES)), dtype=np.float32)
    tables: dict[str, dict[str, np.ndarray]] = {"player": {}, "role": {}}
    player_keys = observations.nfl_id.astype(str).to_numpy()
    role_keys = observations.player_role.astype(str).to_numpy()
    for date in np.unique(dates):
        indices = np.flatnonzero(dates == date)
        for row in indices:
            for offset, group, key in (
                (0, "player", player_keys[row]),
                (5, "role", role_keys[row]),
            ):
                n, sx, sy, square = tables[group].get(key, np.zeros(4))
                denominator = n + smoothing
                result[row, offset : offset + 5] = (
                    np.log1p(n),
                    float(n == 0),
                    sx / denominator,
                    sy / denominator,
                    np.sqrt(max(square / denominator - (sx**2 + sy**2) / denominator**2, 0)),
                )
        for row in indices[training[indices]]:
            ex, ey = observations.iloc[row][["error_x", "error_y"]].to_numpy(float)
            update = np.array([1.0, ex, ey, ex**2 + ey**2])
            for group, key in (("player", player_keys[row]), ("role", role_keys[row])):
                tables[group][key] = tables[group].get(key, np.zeros(4)) + update
    state = {
        group: {key: value.tolist() for key, value in table.items()}
        for group, table in tables.items()
    }
    return result, {
        "tables": state,
        "smoothing": smoothing,
        "training_games": sorted(training_games),
    }
