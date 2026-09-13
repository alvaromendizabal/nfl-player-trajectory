"""Observed-only movement hypotheses, not new labels or forced trajectories.

All vectors use canonical field yards. Each family is a correction to the same
constant-velocity reference used in Round 1. Roles gate the corrections, allowing
opposite responses by role. Fixed physical scales are not fitted on evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from origin_features import KEYS, ROLES, flags, validate_raw, velocity

WINDOWS = (5, 10)
SPEED_EPS = 1e-3


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    role: str
    window: int
    component: str
    units: str


FEATURES = tuple(
    Feature(f"{family}__{role}__w{w}__{component}", family, role, w, component,
            "yards / 10" if component in ("dx", "dy") else "seconds * support fraction")
    for family in ("turn", "brake")
    for role in ROLES for w in WINDOWS for component in ("dx", "dy", "support_t")
) + tuple(
    Feature(f"orientation__{role}__{component}", "orientation", role, 1, component,
            "yards / 10" if component in ("dx", "dy") else "seconds * availability")
    for role in ROLES for component in ("dx", "dy", "support_t")
)
NAMES = tuple(f.name for f in FEATURES)
FAMILIES = {family: np.array([i for i, f in enumerate(FEATURES) if f.family == family], int)
            for family in ("turn", "brake", "orientation")}


def turn_correction(v: np.ndarray, omega: float, tau: float) -> np.ndarray:
    """Exact constant-speed/constant-turn integral minus v*t; continuous at omega=0."""
    v = np.asarray(v, float)
    if v.shape != (2,) or not np.isfinite(v).all() or not np.isfinite([omega, tau]).all() or tau < 0:
        raise ValueError("Finite vector, angular rate and nonnegative time required")
    angle = omega * tau
    sine_integral = tau * np.sinc(angle / np.pi)
    cosine_integral = tau * (angle / 2) * np.sinc(angle / (2 * np.pi)) ** 2
    perpendicular = np.array([-v[1], v[0]])
    return (sine_integral - tau) * v + cosine_integral * perpendicular


def speed_correction(v: np.ndarray, acceleration: float, tau: float) -> np.ndarray:
    """Constant tangential acceleration hypothesis, stopping rather than reversing.

    No label or prediction clipping. The stop is a feature-definition constraint
    on this candidate physical trajectory, not a bound on the learned forecast.
    """
    v = np.asarray(v, float)
    if v.shape != (2,) or not np.isfinite(v).all() or not np.isfinite([acceleration, tau]).all() or tau < 0:
        raise ValueError("Finite vector, acceleration and nonnegative time required")
    speed = float(np.linalg.norm(v))
    if speed < SPEED_EPS:
        return np.zeros(2)
    active = min(tau, speed / -acceleration) if acceleration < 0 else tau
    distance = speed * active + 0.5 * acceleration * active * active
    return v / speed * distance - v * tau


def observed_rates(frames: np.ndarray, v: np.ndarray, available: np.ndarray,
                   origin: int, window: int) -> dict[str, float]:
    """Recent consecutive observations only; no derivative crosses a frame gap.

    Only the contiguous suffix ending at this player's last observed frame is
    eligible. A missing terminal velocity blocks the family. Invalid pairs do
    not become zeros counted as measurements.
    """
    f = np.asarray(frames)
    v, ok = np.asarray(v, float), np.asarray(available)
    if f.ndim != 1 or len(f) == 0 or v.shape != (len(f), 2) or ok.shape != f.shape or ok.dtype != bool:
        raise ValueError("Invalid observed-rate shapes/mask")
    if not np.isfinite(f).all() or (f != np.floor(f)).any() or (np.diff(f) <= 0).any() or f[-1] > origin:
        raise ValueError("Frame clock must be finite, increasing and observed")
    if window not in WINDOWS or not np.isfinite(v[ok]).all():
        raise ValueError("Unsupported window or nonfinite observed velocity")
    result = {k: 0.0 for k in ("omega", "acceleration", "turn_support", "brake_support")}
    if not ok[-1] or np.linalg.norm(v[-1]) < SPEED_EPS:
        return result
    cuts = np.flatnonzero(np.diff(f) != 1)
    start = int(cuts[-1] + 1) if len(cuts) else 0
    start = max(start, int(np.searchsorted(f, origin - window + 1)))
    f, v, ok = f[start:], v[start:], ok[start:]
    if len(f) < 2:
        return result
    pair = (np.diff(f) == 1) & ok[:-1] & ok[1:]
    speed = np.linalg.norm(np.where(ok[:, None], v, 0), axis=1)
    angle_ok = pair & (speed[:-1] >= SPEED_EPS) & (speed[1:] >= SPEED_EPS)
    if angle_ok.any():
        a, b = v[:-1][angle_ok], v[1:][angle_ok]
        angles = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
        result["omega"] = float(np.median(angles / 0.1))
        result["turn_support"] = float(angle_ok.sum() / (window - 1))
    if pair.any():
        result["acceleration"] = float(np.median(np.diff(speed)[pair] / 0.1))
        result["brake_support"] = float(pair.sum() / (window - 1))
    return result


def build_features(observed: pd.DataFrame, requests: pd.DataFrame,
                   *, origin: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return 60 candidate columns in request order. No future coordinate argument.

    requests must contain ONLY the four join keys. Its frame_id is future time
    relative to the actual throw, as supplied by the task. No earlier-origin
    augmentation, target encoding, fitted scaling, ball outcome, or model output
    enters this interface. Source input must be pre-throw tracking.
    """
    validate_raw(observed)
    if list(requests.columns) != KEYS or requests.empty or requests.duplicated(KEYS).any():
        raise ValueError("Nonempty unique requests with ONLY the four keys required")
    numeric = requests.to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric != np.floor(numeric)).any():
        raise ValueError("Request keys must be finite integers")
    if origin is None:
        origin = int(observed.frame_id.max())
    if not isinstance(origin, (int, np.integer)) or observed.frame_id.max() > origin:
        raise ValueError("Rows after the declared origin are not observed inputs")
    if set(map(tuple, requests[KEYS[:2]].drop_duplicates().to_numpy())) != set(map(tuple, observed[KEYS[:2]].drop_duplicates().to_numpy())):
        raise ValueError("Request play differs from the observed play")
    left = bool(observed.play_direction.iloc[0] == "left")
    states: dict[int, dict[str, Any]] = {}
    for ident, g in observed.groupby("nfl_id", sort=True):
        g = g.sort_values("frame_id")
        row = g.iloc[-1]
        v, valid = velocity(g, left)
        rates = {w: observed_rates(g.frame_id.to_numpy(), v, valid, origin, w) for w in WINDOWS}
        ori = np.zeros(2)
        orientation_ok = False
        if "o" in g and pd.notna(row.o) and np.isfinite(float(row.o)) and valid[-1] and np.linalg.norm(v[-1]) >= SPEED_EPS:
            angle = np.deg2rad(float(row.o))
            ori = np.array([np.sin(angle), np.cos(angle)]) * (-1 if left else 1)
            orientation_ok = True
        states[int(ident)] = {"role": str(row.player_role), "v": v[-1], "rates": rates,
                              "age": (origin - int(row.frame_id)) / 10,
                              "horizon": int(row.num_frames_output), "ori": ori,
                              "orientation_ok": orientation_ok,
                              "scored": bool(flags(pd.Series([row.player_to_predict]))[0])}
    x = np.zeros((len(requests), len(FEATURES)), np.float64)
    lookup = {(f.family, f.role, f.window, f.component): i for i, f in enumerate(FEATURES)}
    support = []
    for index, row in enumerate(requests.itertuples(index=False)):
        ident, frame = int(row.nfl_id), int(row.frame_id)
        if ident not in states:
            raise ValueError("Request player not observed")
        s = states[ident]
        if not s["scored"] or not 1 <= frame <= s["horizon"] or s["age"] >= 2:
            raise ValueError("Invalid requested player/horizon or stale observation >=2 seconds")
        tau = frame / 10 + s["age"]
        for family in ("turn", "brake"):
            for w in WINDOWS:
                rate = s["rates"][w]
                fraction = rate[family + "_support"]
                if fraction > 0:
                    correction = (turn_correction(s["v"], rate["omega"], tau) if family == "turn"
                                  else speed_correction(s["v"], rate["acceleration"], tau))
                    for k, value in zip(("dx", "dy", "support_t"), [*list(correction / 10), fraction * tau]):
                        x[index, lookup[(family, s["role"], w, k)]] = value
        if s["orientation_ok"]:
            correction = (np.linalg.norm(s["v"]) * s["ori"] - s["v"]) * tau
            for k, value in zip(("dx", "dy", "support_t"), [*list(correction / 10), tau]):
                x[index, lookup[("orientation", s["role"], 1, k)]] = value
    for ident in sorted(set(requests.nfl_id.astype(int))):
        s = states[ident]
        for family in ("turn", "brake", "orientation"):
            for w in WINDOWS if family != "orientation" else (1,):
                fraction = (s["rates"][w][family + "_support"] if family != "orientation" else float(s["orientation_ok"]))
                support.append({"role": s["role"], "family": family, "window": w,
                                "supported": int(fraction > 0), "fraction": fraction,
                                "age_seconds": s["age"]})
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite candidate features; no row filtering allowed")
    return x, support
