# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Convert verified feature trees to portable numeric arrays; never refit a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ablate_features import error_rows, probe_module

FOLDS = ("inner_1", "inner_2", "inner_3", "development")
VARIANTS = ("without_metadata", "without_optional_inputs")


def portable(pair: list[Any], names: list[str]) -> dict[str, Any]:
    used = sorted(
        {
            int(node["feature_idx"])
            for model in pair
            for predictors in model._predictors
            for node in predictors[0].nodes
            if not node["is_leaf"]
        }
    )
    positions = {old: new for new, old in enumerate(used)}
    axes: list[list[dict[str, Any]]] = []
    initial: list[float] = []
    for model in pair:
        if model.n_features_in_ != len(names) or model.n_trees_per_iteration_ != 1:
            raise ValueError("Expected independent numerical regression trees.")
        initial.append(float(model._baseline_prediction[0, 0]))
        trees = []
        for predictors in model._predictors:
            nodes = predictors[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("Categorical splits need a different export contract.")
            trees.append(
                {
                    "value": nodes["value"].tolist(),
                    "feature": [
                        positions[int(n["feature_idx"])] if not n["is_leaf"] else 0 for n in nodes
                    ],
                    "threshold": nodes["num_threshold"].tolist(),
                    "left": nodes["left"].tolist(),
                    "right": nodes["right"].tolist(),
                    "leaf": nodes["is_leaf"].astype(bool).tolist(),
                }
            )
        axes.append(trees)
    return {
        "features": [names[i] for i in used],
        "initial": initial,
        "axes": axes,
        "screened_feature_count": len(names),
    }


def main(root: Path) -> None:
    from nfl_trajectory.benchmark import error_metrics
    from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.research_evidence import verified_checkpoint
    from nfl_trajectory.research_inference import research_bundle
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage
    from nfl_trajectory.tree_inference import tree_correction

    parent = research_bundle(root, include_tree=False)
    folder = root / "artifacts/feature_attribution"
    report = json.loads((folder / "summary.json").read_text())
    signature = hashlib.sha256(
        json.dumps(report["provenance"], sort_keys=True).encode()
    ).hexdigest()
    if (
        signature != report["source_signature"]
        or report["source_signatures"] != parent["source_signatures"]
    ):
        raise ValueError("Wide attribution belongs to another feature lineage.")
    verified_checkpoint(root, "feature-attribution-report", signature, folder / "summary.json")
    for relative, expected in report["provenance"]["inputs"].items():
        if sha256(root / relative) != expected:
            raise ValueError("Wide attribution source or fitted artifacts changed.")
    folds = [*report["inner_folds"], report["development"]]
    scores = {
        name: float(
            np.sqrt(
                sum(
                    next(
                        row["coordinate_rmse_yards"]
                        for row in fold["omissions"]
                        if row["model"] == name
                    )
                    ** 2
                    * fold["rows"]
                    for fold in folds[:3]
                )
                / sum(fold["rows"] for fold in folds[:3])
            )
        )
        for name in VARIANTS
    }
    selected = min(scores, key=lambda name: (scores[name], name))
    paths = [Path(__file__), Path(__file__).with_suffix(".py.lock"), folder / "summary.json"]
    for fold in FOLDS:
        for name in VARIANTS:
            for extension in (".pkl", ".csv"):
                path = folder / fold / (name + extension)
                verified_checkpoint(root, "wide-attribution-" + fold, signature, path)
                paths.append(path)
    for name in parent["inference_sources"]:
        paths.append(root / "src/nfl_trajectory" / (name + ".py"))
    provenance = {
        "inputs": {str(path.relative_to(root)): sha256(path) for path in sorted(set(paths))},
        "parent_bundle_sha256": hashlib.sha256(
            json.dumps(parent, sort_keys=True).encode()
        ).hexdigest(),
        "wide_attribution_source": signature,
        "inner_scores": scores,
        "selected_variant": selected,
        "policy": (
            "Deployment excludes body/position metadata. Historical priors remain eligible; "
            "cold-history stress is checked separately. Choose between the already-fitted "
            "availability profiles by inner-fold RMSE; development is excluded."
        ),
    }
    source = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output = root / "artifacts/research/tree"
    caches, _ = verify_inputs(root)
    probe = probe_module(root)
    plan = json.loads((root / "artifacts/nonlinear_probe/plan.json").read_text())

    def action() -> None:
        parity_rows = []
        deployment: dict[str, dict[str, Any]] = {}
        selected_names: list[str] = []
        for fold in folds:
            name = fold["fold"]
            split = next(item for item in plan["folds"] if item["name"] == name)
            names = fold["features"]
            evaluation = probe.materialize(root, caches, split, names, False)
            for variant in VARIANTS:
                indices = [
                    i
                    for i, feature in enumerate(names)
                    if not metadata_dependent(feature)
                    and (variant != "without_optional_inputs" or not telemetry_dependent(feature))
                ]
                retained = [names[i] for i in indices]
                pair = pickle.loads((folder / name / (variant + ".pkl")).read_bytes())
                converted = portable(pair, retained)
                matrix = evaluation[0][:, indices]
                expected = np.column_stack([model.predict(matrix) for model in pair])
                lookup = {feature: i for i, feature in enumerate(retained)}
                used = [lookup[feature] for feature in converted["features"]]
                actual = tree_correction(matrix[:, used], converted)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                errors = error_rows(evaluation, actual, variant)
                score = error_metrics(errors)["coordinate_rmse_yards"]
                recorded = next(row for row in fold["omissions"] if row["model"] == variant)
                if not np.isclose(score, recorded["coordinate_rmse_yards"], rtol=1e-10, atol=1e-10):
                    raise ValueError("Portable predictor does not reproduce the experiment metric.")
                parity_rows.append(
                    {
                        "fold": name,
                        "variant": variant,
                        "rows": len(matrix),
                        "screened_features": len(retained),
                        "features": len(converted["features"]),
                        "coordinate_rmse_yards": score,
                        "max_absolute_prediction_difference": float(
                            np.max(np.abs(actual - expected))
                        ),
                    }
                )
                if name == "development":
                    deployment[variant] = converted
                    if variant == selected:
                        selected_names = converted["features"]
        bundle = copy.deepcopy(parent)
        bundle.update(
            selected_stage="fixed_tree",
            selected_model=report["selected_model"] + "__" + selected,
            tree=deployment[selected],
            blocks=[{"kind": "tree", "model": {"features": selected_names}}],
            retained_features=len(selected_names),
            validation_coordinate_rmse_yards=next(
                row["coordinate_rmse_yards"]
                for row in parity_rows
                if row["fold"] == "development" and row["variant"] == selected
            ),
            selection=provenance["policy"],
            tree_source=source,
            fallbacks={
                alias: {
                    "tree": deployment[variant],
                    "blocks": [
                        {"kind": "tree", "model": {"features": deployment[variant]["features"]}}
                    ],
                    "reason": "Independently fitted fixed-tree availability profile.",
                }
                for alias, variant in (
                    ("without_metadata", selected),
                    ("without_telemetry", "without_optional_inputs"),
                )
            },
        )
        atomic_json(
            output / "bundle.json",
            {
                "bundle": bundle,
                "provenance": provenance,
                "source_signature": source,
            },
        )
        atomic_bytes(output / "parity.csv", pd.DataFrame(parity_rows).to_csv(index=False).encode())
        atomic_json(
            output / "summary.json",
            {
                "status": "passed",
                "source_signature": source,
                "provenance": provenance,
                "source_signatures": parent["source_signatures"],
                "selected_model": bundle["selected_model"],
                "selected_stage": "fixed_tree",
                "retained_features": len(selected_names),
                "inner_scores": scores,
                "parity": parity_rows,
                "validation_coordinate_rmse_yards": bundle["validation_coordinate_rmse_yards"],
                "training": "No refit; verified diagnostic trees converted to numeric arrays.",
                "raw_inference_validation": "required separately",
                "holdout_evaluation": "not_run",
                "final_model": False,
            },
        )

    with Run(root, "portable-feature-tree") as run:
        stage(
            root,
            "research-tree-bundle",
            source,
            [output / "bundle.json", output / "summary.json", output / "parity.csv"],
            action,
            run,
        )


def self_test(root: Path) -> None:
    from nfl_trajectory.tree_inference import tree_correction

    rng = np.random.default_rng(2026)
    x = rng.normal(size=(1000, 5)).astype(np.float32)
    y = np.column_stack([x[:, 0] * x[:, 1], np.sin(x[:, 2])])
    pair = probe_module(root).train_pair(x[:750], y[:750])
    converted = portable(pair, [str(i) for i in range(5)])
    expected = np.column_stack([model.predict(x[750:]) for model in pair])
    actual = tree_correction(x[750:, [int(name) for name in converted["features"]]], converted)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    print(json.dumps({"portable_tree_self_test": "passed", "rows": 250}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    if args.self_test:
        self_test(root)
    else:
        main(root)
