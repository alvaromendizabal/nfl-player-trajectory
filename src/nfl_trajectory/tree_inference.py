"""Portable inference from verified fixed-capacity feature trees.

Training stays in the separately locked diagnostic environment. The predictor
uses explicit numeric tree arrays and the canonical raw feature constructors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def tree_correction(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Match raw-value histogram-tree predictions without a sklearn runtime."""
    data = np.asarray(x, dtype=np.float64)
    width = len(model["features"])
    if data.ndim != 2 or data.shape[1] != width or not np.isfinite(data).all():
        raise ValueError("Tree input must be finite with the fitted feature schema.")
    initial = np.asarray(model["initial"], dtype=float)
    if initial.shape != (2,) or not np.isfinite(initial).all() or len(model["axes"]) != 2:
        raise ValueError("Invalid two-coordinate tree intercept.")
    result = np.broadcast_to(initial, (len(data), 2)).copy()
    for axis, trees in enumerate(model["axes"]):
        for tree in trees:
            value = np.asarray(tree["value"], dtype=float)
            feature = np.asarray(tree["feature"], dtype=np.int64)
            threshold = np.asarray(tree["threshold"], dtype=float)
            left = np.asarray(tree["left"], dtype=np.int64)
            right = np.asarray(tree["right"], dtype=np.int64)
            leaf = np.asarray(tree["leaf"], dtype=bool)
            count = len(value)
            arrays = (feature, threshold, left, right, leaf)
            if not count or any(a.shape != (count,) for a in arrays):
                raise ValueError("Tree arrays have inconsistent node counts.")
            if not np.isfinite(value).all() or not np.isfinite(threshold).all():
                raise ValueError("Tree parameters must be finite.")
            branches = np.flatnonzero(~leaf)
            if (
                (feature[branches] < 0).any() or (feature[branches] >= width).any()
                or (left[branches] <= branches).any() or (right[branches] <= branches).any()
                or (left[branches] >= count).any() or (right[branches] >= count).any()
            ):
                raise ValueError("Invalid tree feature or child indices.")
            nodes = np.zeros(len(data), dtype=np.int64)
            for _ in range(count + 1):
                active = np.flatnonzero(~leaf[nodes])
                if not len(active):
                    break
                current = nodes[active]
                choose_left = data[active, feature[current]] <= threshold[current]
                nodes[active] = np.where(choose_left, left[current], right[current])
            else:
                raise ValueError("Tree traversal did not terminate.")
            result[:, axis] += value[nodes]
    if not np.isfinite(result).all():
        raise ValueError("Tree correction must be finite.")
    return result


def load_tree_bundle(root: Path, parent: dict[str, Any]) -> dict[str, Any]:
    """Load only current-source, completed tree conversion with verified fit lineage."""
    from nfl_trajectory.runtime import sha256

    path = root / "artifacts/research/tree/bundle.json"
    if not path.exists():
        return parent
    payload = json.loads(path.read_text())
    provenance = payload["provenance"]
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    receipt = json.loads((root / ".state/research-tree-bundle.json").read_text())
    if (
        signature != payload["source_signature"]
        or receipt.get("signature") != signature or receipt.get("status") != "completed"
        or receipt.get("outputs", {}).get(str(path.relative_to(root))) != sha256(path)
        or provenance["parent_bundle_sha256"]
        != hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("The portable tree bundle is stale or unverified.")
    for relative, expected in provenance["inputs"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Tree fit, input, or source lineage changed.")
    fitted = payload["bundle"]
    if fitted["source_signatures"] != parent["source_signatures"]:
        raise ValueError("Tree bundle uses another feature bank.")
    if fitted["inference_sources"] != parent["inference_sources"]:
        raise ValueError("Tree bundle uses another inference implementation.")
    return dict(fitted)


def predict_tree(
    observed: pd.DataFrame, targets: pd.DataFrame, bundle: dict[str, Any]
) -> pd.DataFrame:
    """Construct exactly the retained columns from observed tracking in bounded batches."""
    from nfl_trajectory.context_features import build_context_features, context_catalog
    from nfl_trajectory.feature_candidates import candidate_matrix, research_catalog
    from nfl_trajectory.features import build_player_features
    from nfl_trajectory.models import predict
    from nfl_trajectory.motion import ENTITY, KEYS
    from nfl_trajectory.representation_features import build_representation, representation_catalog
    from nfl_trajectory.research_inference import frozen_history

    bank = build_player_features(observed, targets[ENTITY])
    state = targets[ENTITY].merge(
        bank.state, on=ENTITY, how="left", sort=False, validate="many_to_one"
    )
    history = frozen_history(state, bundle["history"])
    sign = state.sign.to_numpy(float)[:, None]
    names = bundle["tree"]["features"]
    catalogs = {
        "research": set(research_catalog().feature),
        "context": set(context_catalog().feature),
        "representation": set(representation_catalog().feature),
    }
    if len(names) != len(set(names)) or not set(names).issubset(set().union(*catalogs.values())):
        raise ValueError("Unknown or duplicate tree features.")
    requested = {kind: [name for name in names if name in group]
                 for kind, group in catalogs.items()}
    positions = {name: i for i, name in enumerate(names)}
    banks: dict[str, Any] = {}
    if requested["context"]:
        banks["context"] = build_context_features(observed, bank.state)
    if requested["representation"]:
        banks["representation"] = build_representation(bank, bundle["routes"])
    prediction = predict(observed, targets[KEYS], "role_ridge", bundle["baseline"])
    values = prediction[["x", "y"]].to_numpy()
    for start in range(0, len(targets), 1024):
        end = min(start + 1024, len(targets))
        matrix = np.empty((end - start, len(names)), dtype=np.float32)
        request = targets.iloc[start:end]
        for kind, columns in requested.items():
            if not columns:
                continue
            current = (
                candidate_matrix(bank, request, history[start:end], columns)
                if kind == "research" else banks[kind].matrix(request, columns)
            )
            matrix[:, [positions[name] for name in columns]] = current
        values[start:end] += sign[start:end] * tree_correction(matrix, bundle["tree"])
    if values.shape != (len(targets), 2) or not np.isfinite(values).all():
        raise ValueError("Tree predictions must be finite and aligned.")
    prediction[["x", "y"]] = values
    return prediction
