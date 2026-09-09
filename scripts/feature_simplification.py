# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Test the predeclared combination of two consistently weak direct feature groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ablate_features import error_rows, probe_module


def main(root: Path, fold_name: str) -> None:
    from filelock import FileLock

    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.research_evidence import read, verified_checkpoint, verified_plan
    from nfl_trajectory.research_inference import research_bundle
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage
    from nfl_trajectory.simplification import FOLDS, MODEL, REMOVED_FAMILIES
    from nfl_trajectory.wide_ablation import retained_indices

    if fold_name not in FOLDS:
        raise ValueError("Only the three inner folds and development are authorized.")
    bundle = research_bundle(root)
    tree_path = root / "artifacts/research/tree/summary.json"
    tree = read(tree_path)
    parent_path = root / "artifacts/feature_attribution/summary.json"
    parent = read(parent_path)
    parent_signature = hashlib.sha256(
        json.dumps(parent["provenance"], sort_keys=True).encode()
    ).hexdigest()
    if parent_signature != parent["source_signature"]:
        raise ValueError("The parent attribution provenance is inconsistent.")
    verified_checkpoint(root, "feature-attribution-report", parent_signature, parent_path)
    for path, expected in parent["provenance"]["inputs"].items():
        if sha256(root / path) != expected:
            raise ValueError("The parent fit or numerical source changed.")
    if (
        bundle["selected_stage"] != "fixed_tree"
        or tree["selected_model"] != bundle["selected_model"]
        or tree["provenance"]["selected_variant"] != "without_metadata"
    ):
        raise ValueError("The recorded study requires the selected metadata-free wide parent.")
    fold = next(
        f for f in [*parent["inner_folds"], parent["development"]] if f["fold"] == fold_name
    )
    parent_folder = root / "artifacts/feature_attribution" / fold_name
    parent_model, parent_errors = [
        parent_folder / ("without_metadata" + ext) for ext in (".pkl", ".csv")
    ]
    for path in (parent_model, parent_errors):
        verified_checkpoint(root, "wide-attribution-" + fold_name, parent_signature, path)
    plan_path = root / "artifacts/feature_budget" / fold_name / "plan.json"
    plan = verified_plan(root, plan_path.parent, "feature_budget.py")
    catalog = pd.concat(
        [research_catalog(), context_catalog(), representation_catalog()], ignore_index=True
    )
    families = dict(zip(catalog.feature, catalog.family, strict=True))
    names = fold["features"]
    parent_features = [
        names[i] for i in retained_indices(names, families, "without_metadata", set())
    ]
    keep = [i for i, name in enumerate(parent_features) if families[name] not in REMOVED_FAMILIES]
    features = [parent_features[i] for i in keep]
    probe = probe_module(root)
    paths = [
        Path(__file__),
        Path(__file__).with_suffix(".py.lock"),
        tree_path,
        parent_path,
        parent_model,
        parent_errors,
        plan_path,
        root / "src/nfl_trajectory/simplification.py",
        root / "src/nfl_trajectory/wide_ablation.py",
        root / "src/nfl_trajectory/feature_contracts.py",
        root / "docs/SIMPLIFICATION_PROTOCOL.md",
        root / "scripts/nonlinear_probe.py",
        root / "scripts/nonlinear_probe.py.lock",
        root / "scripts/ablate_features.py",
    ]
    provenance = {
        "inputs": {str(p.relative_to(root)): sha256(p) for p in sorted(set(paths))},
        "source_signatures": bundle["source_signatures"],
        "bundle_sha256": hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
        "fold": fold_name,
        "parent_model": bundle["selected_model"],
        "removed_families": sorted(REMOVED_FAMILIES),
        "settings": probe.SETTINGS,
        "features": features,
        "parent_features": parent_features,
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    folder = root / "artifacts/simplification" / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    model_path, errors_path, output = [
        folder / name for name in ("model.pkl", "errors.csv", "summary.json")
    ]
    with (
        FileLock(str(folder / "pipeline.lock"), timeout=1),
        Run(root, "simplification-" + fold_name) as run,
    ):
        try:
            for suffix, paths_to_check in (
                ("fit", [model_path]),
                ("evaluate", [errors_path, output]),
            ):
                for path in paths_to_check:
                    verified_checkpoint(
                        root, f"simplification-{fold_name}-{suffix}", signature, path
                    )
        except (FileNotFoundError, ValueError):
            pass
        else:
            run.event("complete_fold_reused", fold=fold_name)
            return
        caches, _ = verify_inputs(root)
        evaluation = probe.materialize(root, caches, plan["fold"], parent_features, False)
        pair = pickle.loads(parent_model.read_bytes())
        if any(model.n_features_in_ != len(parent_features) for model in pair):
            raise ValueError("The parent feature width changed.")
        reference = error_rows(
            evaluation,
            np.column_stack([model.predict(evaluation[0]) for model in pair]),
            "reference",
        )
        recorded = pd.read_csv(parent_errors)
        keys = ["game_id", "play_id", "nfl_id", "frame_id"]
        pd.testing.assert_frame_equal(reference[keys], recorded[keys], check_dtype=False)
        np.testing.assert_allclose(
            reference[["dx", "dy"]], recorded[["dx", "dy"]], rtol=1e-9, atol=1e-9
        )
        evaluation = (evaluation[0][:, keep], *evaluation[1:])

        def fit() -> None:
            training = probe.materialize(root, caches, plan["fold"], features, True)
            fitted = probe.train_pair(training[0], training[1])
            atomic_bytes(model_path, pickle.dumps(fitted, protocol=5))

        stage(root, f"simplification-{fold_name}-fit", signature, [model_path], fit, run)

        def evaluate() -> None:
            fitted = pickle.loads(model_path.read_bytes())
            correction = np.column_stack([model.predict(evaluation[0]) for model in fitted])
            errors = error_rows(evaluation, correction, MODEL)
            atomic_bytes(errors_path, errors.to_csv(index=False).encode())
            atomic_json(
                output,
                {
                    "status": "passed",
                    "source_signature": signature,
                    "provenance": provenance,
                    "source_signatures": bundle["source_signatures"],
                    "fold": fold_name,
                    "model": MODEL,
                    "rows": len(errors),
                    "feature_count": len(features),
                    "parent_feature_count": len(parent_features),
                    "reference": error_metrics(reference),
                    "metrics": error_metrics(errors),
                    "paired_game_ci95": np.quantile(
                        bootstrap_scores(errors) - bootstrap_scores(reference), [0.025, 0.975]
                    ).tolist(),
                    "parent_raw_parity": "passed for every evaluation row",
                    "holdout_evaluation": "not_run",
                },
            )

        stage(
            root,
            f"simplification-{fold_name}-evaluate",
            signature,
            [errors_path, output],
            evaluate,
            run,
        )
        result = read(output)
        run.event("combined_omission_completed", fold=fold_name, **result["metrics"])


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    main(root, parser.parse_args().fold)
