"""Translation- and rotation-equivariant motion models using only pre-throw inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.motion import ENTITY, KEYS, require_keys

BASIS = [
    "last_velocity",
    "recent_velocity",
    "acceleration",
    "ball_linear",
    "ball_quadratic",
    "ball_cubic",
]
INPUT_COLUMNS = KEYS + ["x", "y", "ball_land_x", "ball_land_y", "num_frames_output", "player_role"]
BASELINES = [
    "last_position",
    "constant_velocity",
    "smoothed_velocity",
    "constant_acceleration",
    "ball_arrival",
]


def observed_state(inputs: pd.DataFrame) -> pd.DataFrame:
    """Estimate terminal state and a five-frame least-squares velocity in yards/second."""
    require_keys(inputs)
    missing = set(INPUT_COLUMNS) - set(inputs.columns)
    if missing:
        raise ValueError(f"Missing observed input columns: {sorted(missing)}")
    history = inputs[INPUT_COLUMNS].sort_values(KEYS).copy()
    numeric = ["x", "y", "ball_land_x", "ball_land_y", "num_frames_output"]
    if not np.isfinite(history[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Observed features must be finite.")
    horizon = history.num_frames_output.to_numpy(dtype=float)
    if np.any(horizon < 1) or not np.equal(horizon, np.floor(horizon)).all():
        raise ValueError("Forecast horizons must be positive integers.")
    history["player_role"] = history.player_role.fillna("Unknown").astype(str)
    grouped = history.groupby(ENTITY, sort=False)
    delta = grouped[["frame_id", "x", "y"]].diff()
    history["dt"] = delta.frame_id / 10
    previous_dt = history.groupby(ENTITY, sort=False).dt.shift()
    for axis in ["x", "y"]:
        history[f"v{axis}"] = delta[axis] / history.dt
        previous_velocity = history.groupby(ENTITY, sort=False)[f"v{axis}"].shift()
        history[f"a{axis}"] = (history[f"v{axis}"] - previous_velocity) / (
            (history.dt + previous_dt) / 2
        )
    recent = history.groupby(ENTITY, sort=False).tail(5).copy()
    recent["t"] = recent.frame_id / 10
    center = recent.groupby(ENTITY, sort=False)[["t", "x", "y"]].transform("mean")
    recent["tt"] = (recent.t - center.t) ** 2
    for axis in ["x", "y"]:
        recent[f"t{axis}"] = (recent.t - center.t) * (recent[axis] - center[axis])
    sums = recent.groupby(ENTITY, sort=False)[["tt", "tx", "ty"]].sum()
    for axis in ["x", "y"]:
        sums[f"smooth_v{axis}"] = (sums[f"t{axis}"] / sums.tt.replace(0, np.nan)).fillna(0)
    last = history.groupby(ENTITY, sort=False).tail(1).drop(columns=["frame_id", "dt"])
    last[["vx", "vy", "ax", "ay"]] = last[["vx", "vy", "ax", "ay"]].fillna(0)
    return last.merge(sums[["smooth_vx", "smooth_vy"]], on=ENTITY, validate="one_to_one")


def design(inputs: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return aligned state and N × 2 × K vector features; target coordinates are ignored."""
    require_keys(targets)
    state = observed_state(inputs)
    aligned = targets[KEYS].merge(state, on=ENTITY, how="left", sort=False, validate="many_to_one")
    if aligned.x.isna().any():
        raise ValueError("A target has no matching observed player.")
    if (aligned.frame_id > aligned.num_frames_output).any():
        raise ValueError("Target frame exceeds the supplied forecast horizon.")
    t = aligned.frame_id.to_numpy(dtype=float)[:, None] / 10
    q = (
        aligned.frame_id.to_numpy(dtype=float)[:, None]
        / aligned.num_frames_output.to_numpy()[:, None]
    )
    velocity = aligned[["vx", "vy"]].to_numpy()
    smooth = aligned[["smooth_vx", "smooth_vy"]].to_numpy()
    acceleration = aligned[["ax", "ay"]].to_numpy()
    ball = aligned[["ball_land_x", "ball_land_y"]].to_numpy() - aligned[["x", "y"]].to_numpy()
    features = np.stack(
        [velocity * t, smooth * t, acceleration * t**2 / 2, ball * q, ball * q**2, ball * q**3],
        axis=-1,
    )
    if not np.isfinite(features).all():
        raise ValueError("Motion basis contains nonfinite values.")
    return aligned, features


def predict_from_design(
    state: pd.DataFrame, features: np.ndarray, model: str, fitted: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Apply a fixed physical baseline or a train-fitted role-specific ridge model."""
    if features.shape != (len(state), 2, len(BASIS)):
        raise ValueError("Motion basis shape mismatch.")
    displacement = np.zeros((len(state), 2))
    if model == "constant_velocity":
        displacement = features[:, :, 0]
    elif model == "smoothed_velocity":
        displacement = features[:, :, 1]
    elif model == "constant_acceleration":
        displacement = features[:, :, 0] + features[:, :, 2]
    elif model == "ball_arrival":
        displacement = features[:, :, 3]
    elif model == "role_ridge":
        if fitted is None or fitted.get("basis") != BASIS or fitted.get("format") != 1:
            raise ValueError("A compatible fitted role model is required.")
        for role in state.player_role.unique():
            mask = state.player_role.eq(role).to_numpy()
            parameters = fitted["roles"].get(role, fitted["global"])
            weights = np.asarray(parameters["coefficients"], dtype=float)
            if weights.shape != (len(BASIS),) or not np.isfinite(weights).all():
                raise ValueError("Invalid fitted motion coefficients.")
            displacement[mask] = np.einsum("ndk,k->nd", features[mask], weights)
    elif model != "last_position":
        raise ValueError(f"Unknown motion model: {model}")
    result = state[KEYS].copy()
    result[["x", "y"]] = state[["x", "y"]].to_numpy() + displacement
    if not np.isfinite(result[["x", "y"]].to_numpy()).all():
        raise ValueError("Prediction contains nonfinite coordinates.")
    return result


def predict(
    inputs: pd.DataFrame, targets: pd.DataFrame, model: str, fitted: dict[str, Any] | None = None
) -> pd.DataFrame:
    state, features = design(inputs, targets)
    return predict_from_design(state, features, model, fitted)


def sufficient_statistics(
    state: pd.DataFrame, features: np.ndarray, truth: pd.DataFrame
) -> dict[str, Any]:
    """Accumulate additive statistics: retries never require refitting previous weeks."""
    require_keys(truth)
    joined = state[KEYS].merge(truth[KEYS + ["x", "y"]], on=KEYS, validate="one_to_one", how="left")
    if len(truth) != len(state) or not np.isfinite(joined[["x", "y"]].to_numpy()).all():
        raise ValueError("Training targets must exactly match the feature rows.")
    target = joined[["x", "y"]].to_numpy() - state[["x", "y"]].to_numpy()
    roles: dict[str, Any] = {}
    for role in ["__global__", *sorted(state.player_role.unique())]:
        mask = (
            np.ones(len(state), dtype=bool)
            if role == "__global__"
            else state.player_role.eq(role).to_numpy()
        )
        x = features[mask].reshape(-1, len(BASIS))
        y = target[mask].reshape(-1)
        roles[role] = {"count": len(y), "gram": (x.T @ x).tolist(), "rhs": (x.T @ y).tolist()}
    return roles


def fit_statistics(statistics: list[dict[str, Any]], alpha: float = 0.001) -> dict[str, Any]:
    """Minimize mean squared displacement plus ridge on training-RMS-scaled features.

    Each coordinate/frame has equal weight, matching the official score. Shared
    x/y coefficients make the model equivariant without coordinate augmentation.
    Alpha is declared before validation; no validation targets enter this function.
    """
    if not statistics or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Nonempty training statistics and positive ridge alpha are required.")
    fitted: dict[str, Any] = {}
    for role in sorted({key for item in statistics for key in item}):
        records = [item[role] for item in statistics if role in item]
        count = sum(item["count"] for item in records)
        gram = sum(
            (np.asarray(item["gram"]) for item in records), np.zeros((len(BASIS), len(BASIS)))
        )
        rhs = sum((np.asarray(item["rhs"]) for item in records), np.zeros(len(BASIS)))
        if count <= 0 or not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise ValueError("Invalid training sufficient statistics.")
        scale = np.sqrt(np.maximum(np.diag(gram) / count, 1e-12))
        normalized = gram / np.outer(scale, scale) / count
        weights = (
            np.linalg.solve(normalized + alpha * np.eye(len(BASIS)), rhs / scale / count) / scale
        )
        fitted[role] = {"coordinate_count": int(count), "coefficients": weights.tolist()}
    if "__global__" not in fitted:
        raise ValueError("Global fallback statistics are required.")
    return {
        "format": 1,
        "basis": BASIS,
        "alpha": alpha,
        "global": fitted.pop("__global__"),
        "roles": fitted,
    }
