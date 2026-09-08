# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Attribute the training-selected wide representation at fixed estimator capacity."""

from __future__ import annotations

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
SEEDS = (2026, 2027, 2028, 2029, 2030)


def main(root: Path) -> None:
    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.research import feature_research_snapshot, trajectory_permutation
    from nfl_trajectory.research_evidence import (
        pooled_scores,
        verified_checkpoint,
        verified_plan,
        verify_error_metric,
    )
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    destination = root / "artifacts/feature_attribution"
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = feature_research_snapshot(root)
    probe = probe_module(root)
    summaries, plans = [], []
    paths = [
        Path(__file__), Path(__file__).with_suffix(".py.lock"),
        root / "scripts/nonlinear_probe.py", root / "scripts/nonlinear_probe.py.lock",
        root / "scripts/ablate_features.py", root / "scripts/feature_budget.py",
        root / "scripts/feature_budget.py.lock",
        root / "src/nfl_trajectory/research.py",
        root / "src/nfl_trajectory/feature_contracts.py",
    ]
    for fold in FOLDS:
        folder = root / "artifacts/feature_budget" / fold
        plan = verified_plan(root, folder, "feature_budget.py")
        if plan["representation_sources"] != snapshot["source_signatures"]:
            raise ValueError("Wide feature source lineage changed.")
        summary = json.loads((folder / "summary.json").read_text())
        if summary["source_signature"] != plan["source_signature"]:
            raise ValueError("Budget summary differs from its verified plan.")
        for row in summary["models"]:
            if row["model"] == "all_engineered":
                continue
            name = row["model"]
            for phase, extension in (("fit", ".pkl"), ("evaluate", ".csv")):
                output = folder / (name + extension)
                verified_checkpoint(root, f"budget-{fold}-{name}-{phase}",
                                    plan["source_signature"], output)
                paths.append(output)
            verify_error_metric(folder / (name + ".csv"), row)
        paths.extend([folder / "plan.json", folder / "summary.json"])
        summaries.append(summary)
        plans.append(plan)
    inner_scores = pooled_scores(summaries[:3], "rows")
    selected = min(inner_scores, key=lambda name: (inner_scores[name], name))
    catalog = pd.concat([research_catalog(), context_catalog(), representation_catalog()],
                        ignore_index=True)
    families = dict(zip(catalog.feature, catalog.family, strict=True))
    provenance = {
        "inputs": {str(path.relative_to(root)): sha256(path) for path in sorted(set(paths))},
        "source_signatures": snapshot["source_signatures"],
        "selected_model": selected,
        "selection": "pooled chronological inner folds; development excluded",
        "inner_scores": inner_scores,
        "permutation_seeds": list(SEEDS),
        "settings": probe.SETTINGS,
        "omissions": ["without_metadata", "without_optional_inputs"],
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    caches, _ = verify_inputs(root)
    results: list[dict[str, Any]] = []
    with Run(root, "wide-feature-attribution") as run:
        for fold, summary, plan in zip(FOLDS, summaries, plans, strict=True):
            output = destination / fold / "summary.json"
            fold_outputs = [output]
            for variant in ("without_metadata", "without_optional_inputs"):
                fold_outputs.extend([output.parent / (variant + ".pkl"),
                                     output.parent / (variant + ".csv")])

            def action(fold: str = fold, summary: dict[str, Any] = summary,
                       plan: dict[str, Any] = plan, output: Path = output) -> None:
                row = next(r for r in summary["models"] if r["model"] == selected)
                if selected == "all_engineered":
                    bundles = [
                        json.loads((root / "artifacts" / label / fold / "models.json").read_text())
                        for label in ("research", "context", "representation")
                    ]
                    names = probe.feature_sets(*bundles)[selected]
                    model_folder = root / "artifacts/nonlinear_probe" / fold
                    parent_plan = verified_plan(root, model_folder.parent, "nonlinear_probe.py")
                    for phase, extension in (("fit", ".pkl"), ("evaluate", ".csv")):
                        verified_checkpoint(root, f"nonlinear-{fold}-{selected}-{phase}",
                                            parent_plan["source_signature"],
                                            model_folder / (selected + extension))
                else:
                    names = row["features"]
                    model_folder = root / "artifacts/feature_budget" / fold
                fitted = pickle.loads((model_folder / (selected + ".pkl")).read_bytes())
                evaluation = probe.materialize(root, caches, plan["fold"], names, False)
                reference = error_rows(evaluation, np.column_stack(
                    [m.predict(evaluation[0]) for m in fitted]), selected)
                score = error_metrics(reference)["coordinate_rmse_yards"]
                if not np.isclose(score, row["coordinate_rmse_yards"], rtol=1e-10, atol=1e-10):
                    raise ValueError("Fresh feature materialization failed parent prediction parity.")
                parent_errors = pd.read_csv(model_folder / (selected + ".csv"))
                keys = ["game_id", "play_id", "nfl_id", "frame_id"]
                pd.testing.assert_frame_equal(reference[keys].reset_index(drop=True),
                                              parent_errors[keys].reset_index(drop=True),
                                              check_dtype=False)
                np.testing.assert_allclose(reference[["dx", "dy"]], parent_errors[["dx", "dy"]],
                                           rtol=1e-9, atol=1e-9)
                reference_bootstrap = bootstrap_scores(reference)
                permutation_rows = []
                assignments = {seed: trajectory_permutation(evaluation[5], seed) for seed in SEEDS}
                for family in sorted({families[name] for name in names}):
                    indices = [i for i, name in enumerate(names) if families[name] == family]
                    changes = []
                    for seed in SEEDS:
                        permuted = evaluation[0].copy()
                        permuted[:, indices] = evaluation[0][assignments[seed]][:, indices]
                        correction = np.column_stack([m.predict(permuted) for m in fitted])
                        errors = error_rows(evaluation, correction, family)
                        changes.append(error_metrics(errors)["coordinate_rmse_yards"] - score)
                    permutation_rows.append({
                        "family": family, "feature_count": len(indices),
                        "rmse_increase_mean": float(np.mean(changes)),
                        "rmse_increase_min": float(np.min(changes)),
                        "rmse_increase_max": float(np.max(changes)),
                        "seed_changes": changes,
                    })
                    run.event("family_permuted", fold=fold, family=family,
                              rmse_increase=float(np.mean(changes)))
                training = probe.materialize(root, caches, plan["fold"], names, True)
                omissions = []
                masks = {
                    "without_metadata": [
                        i for i, name in enumerate(names) if not metadata_dependent(name)
                    ],
                    "without_optional_inputs": [
                        i for i, name in enumerate(names)
                        if not metadata_dependent(name) and not telemetry_dependent(name)
                    ],
                }
                for variant, indices in masks.items():
                    fitted_path, errors_path = (output.parent / (variant + extension)
                                                for extension in (".pkl", ".csv"))

                    def fit(indices: list[int] = indices, fitted_path: Path = fitted_path) -> None:
                        pair = probe.train_pair(training[0][:, indices], training[1])
                        atomic_bytes(fitted_path, pickle.dumps(pair, protocol=5))

                    stage(root, f"wide-attribution-{fold}-{variant}-fit", signature,
                          [fitted_path], fit, run)
                    pair = pickle.loads(fitted_path.read_bytes())
                    correction = np.column_stack([m.predict(evaluation[0][:, indices]) for m in pair])
                    errors = error_rows(evaluation, correction, variant)
                    atomic_bytes(errors_path, errors.to_csv(index=False).encode())
                    omissions.append({
                        "model": variant, "feature_count": len(indices), **error_metrics(errors),
                        "delta_vs_full_ci95": np.quantile(
                            bootstrap_scores(errors) - reference_bootstrap, [0.025, 0.975]
                        ).tolist(),
                    })
                # Paired incremental width comparisons use matching games and frame keys.
                width_changes = []
                rows = [r for r in summary["models"] if r["model"] != "all_engineered"]
                for previous, current in zip(rows, rows[1:], strict=False):
                    base = root / "artifacts/feature_budget" / fold
                    earlier = pd.read_csv(base / (previous["model"] + ".csv"))
                    later = pd.read_csv(base / (current["model"] + ".csv"))
                    pd.testing.assert_frame_equal(earlier[keys], later[keys])
                    width_changes.append({
                        "from": previous["model"], "to": current["model"],
                        "rmse_change": current["coordinate_rmse_yards"]
                                       - previous["coordinate_rmse_yards"],
                        "paired_game_ci95": np.quantile(
                            bootstrap_scores(later) - bootstrap_scores(earlier), [0.025, 0.975]
                        ).tolist(),
                    })
                atomic_json(output, {
                    "fold": fold, "status": "passed", "source_signature": signature,
                    "rows": len(reference), "selected_model": selected,
                    "feature_count": len(names), "features": names,
                    "coordinate_rmse_yards": score, "permutation": permutation_rows,
                    "omissions": omissions, "incremental_width": width_changes,
                    "holdout_evaluation": "not_run",
                })

            stage(root, "wide-attribution-" + fold, signature, fold_outputs, action, run)
            results.append(json.loads(output.read_text()))

        def report() -> None:
            import matplotlib.pyplot as plt

            counts = {}
            for result in results[:3]:
                for name in result["features"]:
                    counts[name] = counts.get(name, 0) + 1
            table = pd.DataFrame(results[-1]["permutation"]).sort_values("rmse_increase_mean")
            fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
            ax.barh(table.family.str.replace("_", " "), table.rmse_increase_mean, color="#247A89")
            ax.axvline(0, color="#444444", linewidth=0.8)
            ax.set_xlabel("Increase in coordinate RMSE (yards); five trajectory permutations")
            ax.set_title(f"Where the {results[-1]['feature_count']}-feature signal comes from")
            fig.savefig(destination / "figure.png", dpi=160)
            plt.close(fig)
            atomic_json(destination / "summary.json", {
                "status": "passed", "source_signature": signature,
                "source_signatures": snapshot["source_signatures"], "provenance": provenance,
                "selected_model": selected, "inner_scores": inner_scores,
                "development": results[-1], "inner_folds": results[:3],
                "selected_in_all_inner_folds": sorted(n for n, count in counts.items() if count == 3),
                "stability_caveat": "PCA axes can rotate between folds.",
                "permutation_interpretation": (
                    "Whole trajectories shuffled within role and forecast horizon; exact frame "
                    "alignment; physical baseline held fixed. Conditional reliance, not causal "
                    "importance. Seed range is not a confidence interval."
                ),
                "omission_interpretation": (
                    "Refit with identical estimator settings and no replacement features; "
                    "paired game bootstraps quantify development-game uncertainty."
                ),
                "holdout_evaluation": "not_run", "feature_gate": "open",
            })

        stage(root, "feature-attribution-report", signature,
              [destination / "summary.json", destination / "figure.png"], report, run)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    main(root)
