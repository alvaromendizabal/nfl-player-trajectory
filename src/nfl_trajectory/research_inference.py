"""Prediction from the exact training-fold-selected research representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.context_features import build_context_features
from nfl_trajectory.feature_candidates import HISTORY_NAMES, candidate_matrix
from nfl_trajectory.feature_contracts import (
    METADATA_COLUMNS,
    TELEMETRY_COLUMNS,
    family_missing,
    metadata_dependent,
    telemetry_dependent,
)
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
    "feature_contracts",
    "research_inference",
)


def split_fitted_blocks(fitted: dict[str, Any]) -> list[dict[str, Any]]:
    """Partition one fitted correction without double-counting its intercept."""
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.representation_features import representation_catalog

    groups = {
        "research": set(research_catalog().feature),
        "context": set(context_catalog().feature),
        "representation": set(representation_catalog().feature),
    }
    blocks: list[dict[str, Any]] = []
    for kind, names in groups.items():
        indices = [i for i, n in enumerate(fitted["features"]) if n in names]
        if not indices:
            continue
        model = {k: [fitted[k][i] for i in indices] for k in ("mean", "scale", "coefficients")}
        model["features"] = [fitted["features"][i] for i in indices]
        model["intercept"] = fitted["intercept"] if not blocks else [0.0, 0.0]
        blocks.append({"kind": kind, "model": model})
    if sum(len(b["model"]["features"]) for b in blocks) != len(fitted["features"]):
        raise ValueError("Fallback contains an unknown feature dependency.")
    return blocks


def robust_fallback(root: Path, sources: dict[str, str]) -> dict[str, Any] | None:
    """Load only our verified, current-source telemetry-independent fit."""
    import hashlib
    import json

    from nfl_trajectory.runtime import sha256

    folder = root / "artifacts/feature_ablation/development"
    path = folder / "robust_linear.json"
    if not path.exists():
        return None
    plan = json.loads((folder / "plan.json").read_text())
    source = plan.pop("source_signature")
    if source != hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest():
        raise ValueError("Fallback plan has inconsistent provenance.")
    if (
        plan["representations"] != sources
        or plan["script"] != sha256(root / "scripts/ablate_features.py")
        or plan["lock"] != sha256(root / "scripts/ablate_features.py.lock")
        or plan["contracts"] != sha256(Path(__file__).with_name("feature_contracts.py"))
    ):
        raise ValueError("Fallback fit is stale; rerun its ablation experiment.")
    for phase, output in (("fit", path), ("evaluate", folder / "robust_linear.csv")):
        receipt = json.loads(
            (root / ".state" / f"ablation-development-robust-linear-{phase}.json").read_text()
        )
        if (
            receipt.get("status") != "completed"
            or receipt.get("signature") != source
            or receipt.get("outputs", {}).get(str(output.relative_to(root))) != sha256(output)
        ):
            raise ValueError("Fallback requires verified fit and evaluation checkpoints.")
    payload = json.loads(path.read_text())
    names = payload["model"]["features"]
    if any(telemetry_dependent(n) or metadata_dependent(n) for n in names):
        raise ValueError("Fallback depends on unavailable optional inputs.")
    return {
        "blocks": split_fitted_blocks(payload["model"]),
        "routes": json.loads(
            (root / "artifacts/representation/development/routes.json").read_text()
        ),
        "source_signature": source,
        "fit_sha256": sha256(path),
        "reason": "Optional telemetry missing; positional and landing features only.",
    }


def research_bundle(root: Path) -> dict[str, Any]:
    """Resolve current verified experiment lineage; never substitute an older model."""
    import json

    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.research_evidence import joint_evidence
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
    fallbacks = {}
    if chosen_stage == "context" and chosen_name == "plus_context":
        fallbacks["without_metadata"] = {
            "blocks": blocks[:-1]
            + [{"kind": "context", "model": models["additions"]["context_without_metadata"]}],
            "routes": routes,
            "reason": "Optional player metadata missing; independently refitted ablation.",
        }
    robust = robust_fallback(root, snapshot["source_signatures"])
    if robust is not None:
        fallbacks["without_telemetry"] = robust
    first_input = next(iter(sorted((root / "data/raw/train").glob("input_*.csv"))), None)
    if first_input is None:
        raise ValueError("The verified training input schema is required to export research.")
    training_columns = set(pd.read_csv(first_input, nrows=0).columns)
    selected_metrics = snapshot["selected_metrics"]
    evaluation_games = core["fold"]["evaluation_games"]
    joint_sources = None
    joint = joint_evidence(root, snapshot["source_signatures"])
    earlier_score = min(v for scores in snapshot["inner_scores"].values() for v in scores.values())
    if joint is not None and min(joint["inner_scores"].values()) < earlier_score:
        chosen_stage, chosen_name = "joint_linear", joint["selected_model"]
        folder = root / "artifacts/joint_linear/development"
        blocks = split_fitted_blocks(json.loads((folder / (chosen_name + ".json")).read_text()))
        routes = json.loads((root / "artifacts/representation/development/routes.json").read_text())
        fallbacks = {
            variant: {
                "blocks": split_fitted_blocks(json.loads((folder / (name + ".json")).read_text())),
                "routes": routes,
                "reason": "Joint linear omission fit; same baseline and regularization.",
            }
            for variant, name in (
                ("without_metadata", "joint_without_metadata"),
                ("without_telemetry", "joint_positional"),
            )
        }
        selected_metrics = next(r for r in joint["models"] if r["model"] == chosen_name)
        evaluation_games = joint["fold"]["evaluation_games"]
        joint_sources = joint["joint_sources"]
    return {
        "format": 1,
        "kind": "feature_research",
        "selected_stage": chosen_stage,
        "selected_model": chosen_name,
        "baseline": core["baseline"],
        "blocks": blocks,
        "history": core["history"],
        "routes": routes,
        "fallbacks": fallbacks,
        "input_contract": "Any missing required optional field switches the entire request.",
        "trained_optional_inputs": {
            "without_telemetry": set(TELEMETRY_COLUMNS) <= training_columns,
            "without_metadata": set(METADATA_COLUMNS) <= training_columns,
        },
        "training_games": core["fold"]["training_games"],
        "evaluation_games": evaluation_games,
        "validation_coordinate_rmse_yards": selected_metrics["coordinate_rmse_yards"],
        "retained_features": sum(len(b["model"]["features"]) for b in blocks),
        "joint_sources": joint_sources,
        "source_signatures": snapshot["source_signatures"],
        "inference_sources": {
            name: sha256(Path(__file__).with_name(name + ".py")) for name in INFERENCE_MODULES
        },
        "selection": "pooled chronological inner-fold RMSE; development is reporting only",
        "holdout_evaluation": "not_run",
        "final_model": False,
    }


def input_variant(observed: pd.DataFrame, bundle: dict[str, Any]) -> str:
    """Refuse silent zero-fill extrapolation outside the trained input contract."""
    names = [n for block in bundle["blocks"] for n in block["model"]["features"]]
    for variant, columns, dependency in (
        ("without_telemetry", TELEMETRY_COLUMNS, telemetry_dependent),
        ("without_metadata", METADATA_COLUMNS, metadata_dependent),
    ):
        missing = family_missing(observed, columns) or observed[list(columns)].isna().any().any()
        trained_available = bundle.get("trained_optional_inputs", {}).get(variant, True)
        if missing and trained_available and any(dependency(n) for n in names):
            if variant not in bundle.get("fallbacks", {}):
                raise ValueError(f"Required inputs are missing and {variant} is not fitted.")
            return variant
    return "complete_inputs"


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
    variant = input_variant(observed, bundle)
    if variant != "complete_inputs":
        bundle = {**bundle, **bundle["fallbacks"][variant]}
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
