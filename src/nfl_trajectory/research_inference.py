"""Prediction from the exact training-fold-selected research representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.context_features import build_context_features
from nfl_trajectory.feature_candidates import HISTORY_NAMES, candidate_matrix
from nfl_trajectory.features import build_player_features
from nfl_trajectory.models import predict
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.representation_features import build_representation

INFERENCE_MODULES = (
    "runtime",
    "motion",
    "models",
    "features",
    "feature_candidates",
    "context_features",
    "representation_features",
    "research_inference",
)


def research_bundle(root: Path) -> dict[str, Any]:
    """Resolve current verified experiment lineage; never substitute an older model."""
    import json

    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.runtime import sha256

    snapshot = feature_research_snapshot(root)
    core = json.loads((root / "artifacts/research/development/models.json").read_text())
    chosen_stage, chosen_name = snapshot["selected_stage"], snapshot["selected_model"]
    core_name = chosen_name if chosen_stage == "research" else "plus_balanced"
    blocks = [{"kind": "research", "model": core["landing"]}]
    if core_name in core["additions"]:
        blocks.append({"kind": "research", "model": core["additions"][core_name]})
    routes = None
    if chosen_stage != "research":
        folder = root / "artifacts" / chosen_stage / "development"
        models = json.loads((folder / "models.json").read_text())
        if chosen_name in models["additions"]:
            blocks.append({"kind": chosen_stage, "model": models["additions"][chosen_name]})
        if chosen_stage == "representation":
            routes = json.loads((folder / "routes.json").read_text())
    return {
        "format": 1,
        "kind": "feature_research",
        "selected_stage": chosen_stage,
        "selected_model": chosen_name,
        "baseline": core["baseline"],
        "blocks": blocks,
        "history": core["history"],
        "routes": routes,
        "training_games": core["fold"]["training_games"],
        "validation_coordinate_rmse_yards": snapshot["selected_metrics"]["coordinate_rmse_yards"],
        "retained_features": snapshot["selected_retained_features"],
        "source_signatures": snapshot["source_signatures"],
        "inference_sources": {
            name: sha256(Path(__file__).with_name(name + ".py")) for name in INFERENCE_MODULES
        },
        "selection": "pooled chronological inner-fold RMSE; development is reporting only",
        "holdout_evaluation": "not_run",
        "final_model": False,
    }


def frozen_history(state: pd.DataFrame, fitted: dict[str, Any]) -> np.ndarray:
    result = np.zeros((len(state), len(HISTORY_NAMES)), dtype=np.float32)
    smoothing = float(fitted["smoothing"])
    for row, (player, role) in enumerate(
        zip(state.nfl_id.astype(str), state.player_role.astype(str), strict=True)
    ):
        for offset, group, key in ((0, "player", player), (5, "role", role)):
            n, sx, sy, square = fitted["tables"][group].get(key, [0.0, 0.0, 0.0, 0.0])
            denominator = n + smoothing
            result[row, offset : offset + 5] = (
                np.log1p(n),
                float(n == 0),
                sx / denominator,
                sy / denominator,
                np.sqrt(max(square / denominator - (sx * sx + sy * sy) / denominator**2, 0)),
            )
    return result


def linear_correction(x: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    if not fitted["features"]:
        return np.zeros((len(x), 2))
    width = len(fitted["features"])
    expected = {"mean": (width,), "scale": (width,), "coefficients": (width, 2), "intercept": (2,)}
    parameters = {name: np.asarray(fitted[name], dtype=float) for name in expected}
    if any(v.shape != expected[k] or not np.isfinite(v).all() for k, v in parameters.items()):
        raise ValueError("Research parameter shape or finite-value contract failed.")
    if (parameters["scale"] <= 0).any():
        raise ValueError("Research scales must be positive.")
    return ((x.astype(float) - parameters["mean"]) / parameters["scale"]) @ parameters[
        "coefficients"
    ] + parameters["intercept"]


def predict_research(
    observed: pd.DataFrame, targets: pd.DataFrame, bundle: dict[str, Any]
) -> pd.DataFrame:
    """Future coordinates are ignored; historical state is frozen and row order is retained."""
    require_keys(targets)
    if bundle.get("kind") != "feature_research" or bundle.get("format") != 1:
        raise ValueError("Use a verified research inference bundle.")
    training_dates = np.asarray(bundle["training_games"], dtype=np.int64) // 100
    if (
        len(targets) == 0
        or (targets.game_id.to_numpy(np.int64) // 100 <= training_dates.max()).any()
    ):
        raise ValueError("Research inference must strictly follow its frozen training dates.")
    bank = build_player_features(observed, targets[ENTITY])
    state = targets[ENTITY].merge(
        bank.state, on=ENTITY, how="left", sort=False, validate="many_to_one"
    )
    h = frozen_history(state, bundle["history"])
    sign = state.sign.to_numpy(float)[:, None]
    prediction = predict(observed, targets[KEYS], "role_ridge", bundle["baseline"])
    values = prediction[["x", "y"]].to_numpy()
    banks: dict[str, Any] = {}
    if any(block["kind"] == "context" for block in bundle["blocks"]):
        banks["context"] = build_context_features(observed, bank.state)
    if any(block["kind"] == "representation" for block in bundle["blocks"]):
        banks["representation"] = build_representation(bank, bundle["routes"])
    for start in range(0, len(targets), 1024):
        end = start + 1024
        for block in bundle["blocks"]:
            columns = block["model"]["features"]
            if not columns:
                continue
            x = (
                candidate_matrix(bank, targets.iloc[start:end], h[start:end], columns)
                if block["kind"] == "research"
                else banks[block["kind"]].matrix(targets.iloc[start:end], columns)
            )
            values[start:end] += sign[start:end] * linear_correction(x, block["model"])
    if values.shape != (len(targets), 2) or not np.isfinite(values).all():
        raise ValueError("Research predictions must be finite and aligned.")
    prediction[["x", "y"]] = values
    return prediction
