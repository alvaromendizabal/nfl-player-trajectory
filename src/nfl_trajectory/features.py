"""Pre-throw football feature bank; candidates are not claims of predictive value.

History is sampled in actual 0.1-second frame offsets, not row offsets. Missing
history and absent neighbours have explicit masks. Coordinates are never clipped.
Only a caller-supplied, pre-throw input table is observed; future target columns
are ignored. Models must fit screening/scaling on training games exclusively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.motion import ENTITY, KEYS, require_keys

WINDOWS = (3, 5, 8, 12, 20, 30, 40)
LAGS = tuple(range(20))
STATS = ("mean", "std", "min", "max", "rms", "slope", "change", "ewmean")
CHANNELS = (
    "relative_x", "relative_y", "vx", "vy", "ax", "ay", "speed", "acceleration",
    "s", "a", "dir_sin", "dir_cos", "o_sin", "o_cos", "orientation_alignment",
    "ball_dx", "ball_dy", "ball_distance", "ball_ux", "ball_uy", "closing_speed",
    "lateral_speed", "radial_acceleration", "lateral_acceleration", "turn_rate",
    "jerk", "velocity_alignment", "sideline_distance",
)
NEIGHBOR_CHANNELS = (
    "dx", "dy", "distance", "dvx", "dvy", "relative_speed", "closing_speed",
    "closest_time", "closest_distance", "ball_distance", "ball_distance_advantage",
    "heading_alignment", "age_seconds", "present",
)
NEIGHBOR_GROUPS = (("teammate", 6), ("opponent", 6), ("receiver", 1), ("passer", 1))
GATE_INPUTS = tuple(f"lag00__{c}" for c in CHANNELS) + tuple(
    f"{side}{slot}__{c}" for side, count in NEIGHBOR_GROUPS for slot in range(1, count + 1)
    for c in ("dx", "dy", "distance", "dvx", "dvy", "closing_speed",
              "closest_time", "closest_distance", "ball_distance_advantage")
)
ROLES = ("Targeted Receiver", "Defensive Coverage", "Pass Route", "Passer", "Unknown")
REQUIRED = KEYS + ["x", "y", "ball_land_x", "ball_land_y", "num_frames_output",
                   "play_direction", "player_side", "player_role"]
WIDTH = 160.0 / 3.0
MAX_MATRIX_BYTES = 256 * 1024**2


def feature_catalog() -> pd.DataFrame:
    """Return the deterministic schema without reading data or targets."""
    records = []
    for channel in CHANNELS:
        records += [(f"lag{lag:02d}__{channel}", "history_lags") for lag in LAGS]
        records += [(f"w{w:02d}__{channel}__{stat}", "history_summaries")
                    for w in WINDOWS for stat in STATS]
    records += [(f"lag{lag:02d}__present", "availability") for lag in LAGS]
    records += [(f"w{w:02d}__coverage", "availability") for w in WINDOWS]
    records += [(f"telemetry__{c}__present", "availability") for c in ("s", "a", "dir", "o")]
    for team, count in NEIGHBOR_GROUPS:
        for slot in range(1, count + 1):
            records += [(f"{team}{slot}__{c}", "interactions") for c in NEIGHBOR_CHANNELS]
        for radius in ((2, 5, 10, 20) if team in ("teammate", "opponent") else ()):
            records.append((f"{team}__within_{radius}yd", "interactions"))
    records += [(f"role__{role}", "context") for role in ROLES]
    records += [(name, "context") for name in (
        "field_x", "field_y", "line_of_scrimmage_dx", "line_of_scrimmage_present",
        "horizon_seconds", "history_seconds", "observation_age_seconds",
    )]
    records += [(name, "forecast") for name in (
        "time_seconds", "time_squared", "fraction", "fraction_squared", "fraction_cubed",
        "remaining_seconds",
    )]
    records += [(f"{gate}__{channel}", "forecast_interactions")
                for gate in ("time", "fraction", "fraction_squared")
                for channel in (c.removeprefix("lag00__") for c in GATE_INPUTS)]
    catalog = pd.DataFrame(records, columns=["feature", "family"])
    landing = ("ball_", "closing_speed", "lateral_speed", "radial_acceleration",
               "lateral_acceleration", "velocity_alignment")
    catalog["signal"] = ["interaction" if any(group in name for group, _ in NEIGHBOR_GROUPS)
                         else "landing" if any(term in name for term in landing)
                         else "motion" for name in catalog.feature]
    return catalog


def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.abs(b) > 1e-8)


def _history(inputs: pd.DataFrame) -> pd.DataFrame:
    require_keys(inputs)
    if not set(REQUIRED).issubset(inputs):
        raise ValueError(f"Missing feature inputs: {sorted(set(REQUIRED) - set(inputs))}")
    optional = [c for c in ("s", "a", "dir", "o", "absolute_yardline_number") if c in inputs]
    h = inputs[REQUIRED + optional].sort_values(KEYS).copy()
    numeric = h[["x", "y", "ball_land_x", "ball_land_y", "num_frames_output"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Positions, landing coordinates, and forecast horizons must be finite.")
    horizon = h.num_frames_output.to_numpy(float)
    if (horizon < 1).any() or not np.equal(horizon, np.floor(horizon)).all():
        raise ValueError("Forecast horizons must be positive integers.")
    if not h.play_direction.isin(["left", "right"]).all():
        raise ValueError("play_direction must be left or right.")
    if not h.player_side.isin(["Offense", "Defense"]).all():
        raise ValueError("player_side must be Offense or Defense.")
    h["player_role"] = h.player_role.fillna("Unknown").astype(str)
    constant = ["play_direction", "player_side", "player_role", "num_frames_output",
                "ball_land_x", "ball_land_y"]
    if (h.groupby(ENTITY)[constant].nunique(dropna=False) > 1).any().any():
        raise ValueError("Player metadata changed within an observed history.")
    play = h.groupby(KEYS[:2])
    if (play[["play_direction", "num_frames_output", "ball_land_x", "ball_land_y"]]
            .nunique(dropna=False) > 1).any().any():
        raise ValueError("Play direction, landing point, and horizon must agree within a play.")
    h["sign"] = np.where(h.play_direction.eq("right"), 1.0, -1.0)
    h["raw_x"], h["raw_y"] = h.x, h.y
    h["x"] = np.where(h.sign > 0, h.x, 120.0 - h.x)
    h["y"] = np.where(h.sign > 0, h.y, WIDTH - h.y)
    h["ball_land_x"] = np.where(h.sign > 0, h.ball_land_x, 120.0 - h.ball_land_x)
    h["ball_land_y"] = np.where(h.sign > 0, h.ball_land_y, WIDTH - h.ball_land_y)
    grouped = h.groupby(ENTITY, sort=False)
    delta = grouped[["frame_id", "x", "y"]].diff()
    dt = delta.frame_id.to_numpy(float) / 10.0
    for axis in ("x", "y"):
        h[f"v{axis}"] = _divide(delta[axis].to_numpy(float), dt)
        h[f"v{axis}"] = h[f"v{axis}"].fillna(0.0)
        dv = h.groupby(ENTITY, sort=False)[f"v{axis}"].diff().to_numpy(float)
        h[f"a{axis}"] = np.nan_to_num(_divide(dv, dt), nan=0.0)
        h.loc[h.groupby(ENTITY, sort=False).cumcount() < 2, f"a{axis}"] = 0.0
    h["speed"] = np.hypot(h.vx, h.vy)
    h["acceleration"] = np.hypot(h.ax, h.ay)
    last_xy = h.groupby(ENTITY, sort=False)[["x", "y"]].transform("last")
    h["relative_x"], h["relative_y"] = h.x - last_xy.x, h.y - last_xy.y
    for c in ("s", "a", "dir", "o"):
        v = pd.to_numeric(h[c], errors="raise").to_numpy(float) if c in h else np.full(len(h), np.nan)
        if np.isinf(v).any():
            raise ValueError("Telemetry may be missing but must not contain infinity.")
        h[f"telemetry__{c}__present"] = np.isfinite(v).astype(float)
        h[c] = np.nan_to_num(v, nan=0.0)
    for angle in ("dir", "o"):
        radians = np.deg2rad(h[angle].to_numpy(float))
        valid = h[f"telemetry__{angle}__present"].to_numpy(float)
        h[f"{angle}_sin"] = np.sin(radians) * h.sign * valid
        h[f"{angle}_cos"] = np.cos(radians) * h.sign * valid
    h["orientation_alignment"] = h.dir_sin * h.o_sin + h.dir_cos * h.o_cos
    h["ball_dx"], h["ball_dy"] = h.ball_land_x - h.x, h.ball_land_y - h.y
    h["ball_distance"] = np.hypot(h.ball_dx, h.ball_dy)
    h["ball_ux"] = _divide(h.ball_dx.to_numpy(), h.ball_distance.to_numpy())
    h["ball_uy"] = _divide(h.ball_dy.to_numpy(), h.ball_distance.to_numpy())
    h["closing_speed"] = h.vx * h.ball_ux + h.vy * h.ball_uy
    h["lateral_speed"] = h.vx * h.ball_uy - h.vy * h.ball_ux
    h["radial_acceleration"] = h.ax * h.ball_ux + h.ay * h.ball_uy
    h["lateral_acceleration"] = h.ax * h.ball_uy - h.ay * h.ball_ux
    h["velocity_alignment"] = _divide(h.closing_speed.to_numpy(), h.speed.to_numpy())
    previous = h.groupby(ENTITY, sort=False)[["vx", "vy", "ax", "ay"]].shift()
    cross = previous.vx * h.vy - previous.vy * h.vx
    dot = previous.vx * h.vx + previous.vy * h.vy
    h["turn_rate"] = np.nan_to_num(_divide(np.arctan2(cross, dot).to_numpy(), dt), nan=0.0)
    h["jerk"] = np.nan_to_num(_divide(np.hypot(h.ax - previous.ax, h.ay - previous.ay), dt), nan=0.0)
    h.loc[h.groupby(ENTITY, sort=False).cumcount() < 3, "jerk"] = 0.0
    h["sideline_distance"] = np.minimum(h.y, WIDTH - h.y)
    h["release_frame"] = h.groupby(KEYS[:2], sort=False).frame_id.transform("max")
    return h


def _summaries(values: np.ndarray, frames: np.ndarray) -> np.ndarray:
    t = (frames - frames[-1]) / 10.0
    centered = t - t.mean()
    denom = float(centered @ centered)
    slope = (centered @ values) / denom if denom > 0 else np.zeros(values.shape[1])
    weights = np.exp(t / max(float(-t.min()), 0.1))
    return np.stack((values.mean(0), values.std(0), values.min(0), values.max(0),
                     np.sqrt(np.mean(values**2, axis=0)), slope, values[-1] - values[0],
                     (weights @ values) / weights.sum()), axis=1)


def _neighbors(row: pd.Series, play: pd.DataFrame) -> list[float]:
    values = []
    for team, count in NEIGHBOR_GROUPS:
        if team in ("teammate", "opponent"):
            same = play.player_side.eq(row.player_side)
            candidates = play[(same if team == "teammate" else ~same) & play.nfl_id.ne(row.nfl_id)].copy()
        else:
            role = "Targeted Receiver" if team == "receiver" else "Passer"
            candidates = play[play.player_role.eq(role)].copy()
        candidates["distance"] = np.hypot(candidates.x - row.x, candidates.y - row.y)
        candidates = candidates.sort_values(["distance", "x", "y", "vx", "vy", "nfl_id"])
        for slot in range(count):
            if slot >= len(candidates):
                values.extend([0.0] * len(NEIGHBOR_CHANNELS))
                continue
            n = candidates.iloc[slot]
            dx, dy, dvx, dvy = n.x - row.x, n.y - row.y, n.vx - row.vx, n.vy - row.vy
            distance = float(n.distance)
            dot = dx * dvx + dy * dvy
            vv = dvx**2 + dvy**2
            closest = min(max(-dot / vv, 0.0), row.num_frames_output / 10.0) if vv > 1e-8 else 0.0
            closest_distance = float(np.hypot(dx + dvx * closest, dy + dvy * closest))
            alignment = (n.vx * row.vx + n.vy * row.vy) / max(n.speed * row.speed, 1e-8)
            values.extend([dx, dy, distance, dvx, dvy, np.sqrt(vv),
                           -dot / max(distance, 1e-8), closest, closest_distance,
                           n.ball_distance, n.ball_distance - row.ball_distance, alignment,
                           (row.release_frame - n.frame_id) / 10.0, 1.0])
        if team in ("teammate", "opponent"):
            values.extend([float((candidates.distance <= radius).sum()) for radius in (2, 5, 10, 20)])
    return values


@dataclass
class PlayerFeatures:
    """Static features stored once per player; future-row expansion is batch-bounded."""

    state: pd.DataFrame
    values: np.ndarray

    def matrix(self, targets: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
        require_keys(targets)
        catalog = feature_catalog()
        names = catalog.feature.tolist()
        chosen = names if columns is None else columns
        if len(chosen) != len(set(chosen)) or not set(chosen).issubset(names):
            raise ValueError("Requested feature names must be unique members of the feature schema.")
        if len(targets) * len(chosen) * 4 > MAX_MATRIX_BYTES:
            raise MemoryError("Feature expansion exceeds 256 MiB; use batches or selected columns.")
        index = targets[ENTITY].merge(self.state[ENTITY].assign(_row=np.arange(len(self.state))),
                                     on=ENTITY, how="left", sort=False, validate="many_to_one")
        if index._row.isna().any():
            raise ValueError("A target has no matching observed player.")
        rows = index._row.to_numpy(int)
        horizon = self.state.num_frames_output.to_numpy(float)[rows]
        frame = targets.frame_id.to_numpy(float)
        if np.any(frame > horizon):
            raise ValueError("Target frame exceeds the supplied horizon.")
        t, q = frame / 10.0, frame / horizon
        temporal = [t, t**2, q, q**2, q**3, (horizon - frame) / 10.0]
        result = np.empty((len(targets), len(chosen)), dtype=np.float32)
        lookup = {name: i for i, name in enumerate(names)}
        static_count = self.values.shape[1]
        for j, name in enumerate(chosen):
            k = lookup[name]
            if k < static_count:
                result[:, j] = self.values[rows, k]
            elif k < static_count + 6:
                result[:, j] = temporal[k - static_count]
            else:
                offset = k - static_count - 6
                gate, channel = divmod(offset, len(GATE_INPUTS))
                base = self.values[rows, lookup[GATE_INPUTS[channel]]]
                result[:, j] = base * (t, q, q**2)[gate]
        if not np.isfinite(result).all():
            raise ValueError("Feature expansion produced a nonfinite value.")
        return result


def build_player_features(inputs: pd.DataFrame, entities: pd.DataFrame | None = None) -> PlayerFeatures:
    """Build features from observed tracking only, including non-predicted neighbours."""
    h = _history(inputs)
    terminal = h.groupby(ENTITY, sort=False).tail(1).copy()
    if entities is None:
        wanted = set(map(tuple, terminal[ENTITY].to_numpy()))
    else:
        if not set(ENTITY).issubset(entities) or entities[ENTITY].isna().any().any():
            raise ValueError("Complete player entity keys are required.")
        wanted = set(map(tuple, entities[ENTITY].drop_duplicates().to_numpy()))
        available = set(map(tuple, terminal[ENTITY].to_numpy()))
        if not wanted or not wanted.issubset(available):
            raise ValueError("Requested entities must exist in observed inputs.")
    plays = {key: p for key, p in terminal.groupby(KEYS[:2], sort=False)}
    rows: list[list[float]] = []
    states: list[dict[str, Any]] = []
    for key, history in h.groupby(ENTITY, sort=False):
        if key not in wanted:
            continue
        row = history.iloc[-1]
        frames = history.frame_id.to_numpy(float)
        series = history[list(CHANNELS)].to_numpy(float)
        positions = {int(f): i for i, f in enumerate(frames)}
        lag_values = np.stack([series[positions[int(frames[-1]) - lag]]
                               if int(frames[-1]) - lag in positions else np.zeros(len(CHANNELS))
                               for lag in LAGS], axis=1)
        summaries = np.stack([_summaries(series[frames > frames[-1] - w],
                                         frames[frames > frames[-1] - w]) for w in WINDOWS], axis=1)
        values = np.concatenate([lag_values, summaries.reshape(len(CHANNELS), -1)], axis=1).ravel().tolist()
        values.extend([float(int(frames[-1]) - lag in positions) for lag in LAGS])
        values.extend([float((frames > frames[-1] - w).sum()) / w for w in WINDOWS])
        values.extend([float(row[f"telemetry__{c}__present"]) for c in ("s", "a", "dir", "o")])
        values.extend(_neighbors(row, plays[key[:2]]))
        role = row.player_role if row.player_role in ROLES else "Unknown"
        values.extend([float(role == r) for r in ROLES])
        los = float(row.get("absolute_yardline_number", np.nan))
        if np.isinf(los):
            raise ValueError("Line of scrimmage must not be infinite.")
        los = los if row.sign > 0 else 120.0 - los
        values.extend([float(row.x), float(row.y), row.x - los if np.isfinite(los) else 0.0,
                       float(np.isfinite(los)), row.num_frames_output / 10.0,
                       (frames[-1] - frames[0]) / 10.0, (row.release_frame - frames[-1]) / 10.0])
        rows.append(values)
        states.append(dict(zip(ENTITY, key, strict=True)) | {
            "x": float(row.raw_x), "y": float(row.raw_y), "sign": float(row.sign),
            "num_frames_output": int(row.num_frames_output), "player_role": str(row.player_role),
        })
    array = np.asarray(rows, dtype=np.float32)
    expected = len(feature_catalog()) - 6 - 3 * len(GATE_INPUTS)
    if array.shape != (len(states), expected) or not np.isfinite(array).all():
        raise ValueError("Static feature shape or finite-value contract failed.")
    return PlayerFeatures(pd.DataFrame(states), array)
