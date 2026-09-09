# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Strict group refits at the selected wide availability profile, before final training."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

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
    from nfl_trajectory.research_evidence import FOLDS, read, verified_checkpoint, verified_plan
    from nfl_trajectory.research_inference import research_bundle
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage
    from nfl_trajectory.wide_ablation import GROUPS, retained_indices

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
        raise ValueError("The attribution report has inconsistent provenance.")
    verified_checkpoint(root, "feature-attribution-report", parent_signature, parent_path)
    for path, expected in parent["provenance"]["inputs"].items():
        if sha256(root / path) != expected:
            raise ValueError("The wide parent source or fitted artifacts changed.")
    if (
        tree["selected_model"] != bundle["selected_model"]
        or bundle["selected_stage"] != "fixed_tree"
    ):
        raise ValueError("The ablation must explain the current portable predictor.")
    profile = tree["provenance"]["selected_variant"]
    fold = next(
        f for f in [*parent["inner_folds"], parent["development"]] if f["fold"] == fold_name
    )
    parent_folder = root / "artifacts/feature_attribution" / fold_name
    parent_model, parent_errors = [parent_folder / (profile + ext) for ext in (".pkl", ".csv")]
    for path in (parent_model, parent_errors):
        verified_checkpoint(root, "wide-attribution-" + fold_name, parent_signature, path)
    plan = verified_plan(root, root / "artifacts/feature_budget" / fold_name, "feature_budget.py")
    catalog = pd.concat(
        [research_catalog(), context_catalog(), representation_catalog()], ignore_index=True
    )
    families = dict(zip(catalog.feature, catalog.family, strict=True))
    covered = [family for group in GROUPS.values() for family in group]
    if len(covered) != len(set(covered)) or set(covered) != set(catalog.family):
        raise ValueError("The predeclared groups must partition the complete family catalog.")
    names = fold["features"]
    indices = retained_indices(names, families, profile, set())
    retained = [names[i] for i in indices]
    probe = probe_module(root)
    paths = [
        Path(__file__),
        Path(__file__).with_suffix(".py.lock"),
        tree_path,
        parent_path,
        parent_model,
        parent_errors,
        root / "src/nfl_trajectory/wide_ablation.py",
        root / "src/nfl_trajectory/feature_contracts.py",
        root / "docs/WIDE_ABLATION_PROTOCOL.md",
        root / "scripts/nonlinear_probe.py",
        root / "scripts/nonlinear_probe.py.lock",
        root / "scripts/ablate_features.py",
        root / "artifacts/feature_budget" / fold_name / "plan.json",
    ]
    provenance = {
        "inputs": {str(p.relative_to(root)): sha256(p) for p in sorted(set(paths))},
        "source_signatures": bundle["source_signatures"],
        "bundle_sha256": hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
        "fold": fold_name,
        "profile": profile,
        "selected_model": bundle["selected_model"],
        "groups": {name: sorted(group) for name, group in GROUPS.items()},
        "settings": probe.SETTINGS,
        "features": retained,
        "selection": "Width and availability profile selected by pooled inner folds only.",
    }
    source = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    folder = root / "artifacts/wide_ablation" / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(folder / "pipeline.lock"), timeout=1),
        Run(root, "wide-ablation-" + fold_name) as run,
    ):
        output = folder / "summary.json"
        try:
            verified_checkpoint(root, "wide-group-" + fold_name, source, output)
            previous = read(output)
            for row in previous["models"]:
                if row["refitted"]:
                    for phase, extension in (("fit", ".pkl"), ("evaluate", ".csv")):
                        verified_checkpoint(
                            root,
                            f"wide-group-{fold_name}-{row['group']}-{phase}",
                            source,
                            folder / (row["model"] + extension),
                        )
        except (FileNotFoundError, ValueError):
            pass
        else:
            run.event("complete_fold_reused", fold=fold_name)
            return
        caches, _ = verify_inputs(root)
        evaluation = probe.materialize(root, caches, plan["fold"], retained, False)
        fitted = pickle.loads(parent_model.read_bytes())
        if any(model.n_features_in_ != len(retained) for model in fitted):
            raise ValueError("Parent feature order/width differs from the availability profile.")
        reference = error_rows(
            evaluation, np.column_stack([m.predict(evaluation[0]) for m in fitted]), "reference"
        )
        recorded = pd.read_csv(parent_errors)
        keys = ["game_id", "play_id", "nfl_id", "frame_id"]
        pd.testing.assert_frame_equal(reference[keys], recorded[keys], check_dtype=False)
        np.testing.assert_allclose(
            reference[["dx", "dy"]], recorded[["dx", "dy"]], rtol=1e-9, atol=1e-9
        )
        reference_metrics = error_metrics(reference)
        reference_samples = bootstrap_scores(reference)
        training = probe.materialize(root, caches, plan["fold"], retained, True)
        rows: list[dict[str, Any]] = []
        outputs = [output]
        for group, removed in GROUPS.items():
            keep = [i for i, feature in enumerate(retained) if families[feature] not in removed]
            name = "without_" + group
            fitted_path, errors_path = [folder / (name + ext) for ext in (".pkl", ".csv")]
            removed_count = len(retained) - len(keep)
            if removed_count:

                def fit(keep: list[int] = keep, fitted_path: Path = fitted_path) -> None:
                    pair = probe.train_pair(training[0][:, keep], training[1])
                    atomic_bytes(fitted_path, pickle.dumps(pair, protocol=5))

                stage(root, f"wide-group-{fold_name}-{group}-fit", source, [fitted_path], fit, run)

                def evaluate(
                    keep: list[int] = keep,
                    fitted_path: Path = fitted_path,
                    errors_path: Path = errors_path,
                    name: str = name,
                ) -> None:
                    pair = pickle.loads(fitted_path.read_bytes())
                    correction = np.column_stack([m.predict(evaluation[0][:, keep]) for m in pair])
                    errors = error_rows(evaluation, correction, name)
                    atomic_bytes(errors_path, errors.to_csv(index=False).encode())

                stage(
                    root,
                    f"wide-group-{fold_name}-{group}-evaluate",
                    source,
                    [errors_path],
                    evaluate,
                    run,
                )
                errors = pd.read_csv(errors_path)
                outputs.extend([fitted_path, errors_path])
            else:
                # An absent family is an explicit structural control, not a fabricated refit.
                errors = reference
            metrics = error_metrics(errors)
            rows.append(
                {
                    "model": name,
                    "group": group,
                    "removed_families": sorted(removed),
                    "removed_features": removed_count,
                    "feature_count": len(keep),
                    "refitted": bool(removed_count),
                    **metrics,
                    "rmse_change": metrics["coordinate_rmse_yards"]
                    - reference_metrics["coordinate_rmse_yards"],
                    "paired_game_ci95": np.quantile(
                        bootstrap_scores(errors) - reference_samples, [0.025, 0.975]
                    ).tolist(),
                }
            )
            run.event("group_completed", fold=fold_name, group=group, **metrics)

        def publish() -> None:
            atomic_json(
                output,
                {
                    "status": "passed",
                    "source_signature": source,
                    "provenance": provenance,
                    "source_signatures": bundle["source_signatures"],
                    "fold": fold_name,
                    "selected_model": bundle["selected_model"],
                    "profile": profile,
                    "rows": len(reference),
                    "reference": reference_metrics,
                    "feature_count": len(retained),
                    "models": rows,
                    "holdout_evaluation": "not_run",
                    "interpretation": (
                        "Strict group removal; fixed estimator and physical baseline; "
                        "no replacement columns. Paired game intervals are descriptive, "
                        "without multiple-comparison adjustment."
                    ),
                },
            )

        stage(root, "wide-group-" + fold_name, source, outputs, publish, run)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", required=True)
    main(root, parser.parse_args().fold)
