# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Widen the feature search with fixed model capacity and training-only selection."""

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

BUDGETS = (512, 1024, 2048, 4096, 8192)
POOL = 8192


def candidate_pool(root: Path, fold: str, parent: list[str]) -> list[str]:
    screens = []
    for label in ("research", "context", "representation"):
        file = "conditional.csv" if label == "research" else "screening.csv"
        screens.append(pd.read_csv(root / "artifacts" / label / fold / file))
    table = pd.concat(screens, ignore_index=True)
    table = table[table.screen_status.eq("eligible")].sort_values(
        ["training_association", "feature"], ascending=[False, True], kind="stable"
    )
    families = [part.feature.tolist() for _, part in table.groupby("family", sort=True)]
    selected = list(parent)
    seen = set(selected)
    for rank in range(max(map(len, families))):
        for family in families:
            if rank < len(family) and family[rank] not in seen:
                selected.append(family[rank])
                seen.add(family[rank])
                if len(selected) == POOL:
                    return selected
    return selected


def retain_columns(x: np.ndarray, protected: int) -> list[int]:
    # Only training rows. Full-training variance eligibility was checked upstream.
    sample = np.random.default_rng(2026).choice(len(x), min(8192, len(x)), replace=False)
    small = x[sample].astype(float)
    small -= small.mean(0)
    norms = np.sqrt(np.sum(small**2, axis=0))
    normalized = small / np.maximum(norms, 1e-12)
    correlation = normalized.T @ normalized
    chosen = list(range(protected))
    for index in range(protected, x.shape[1]):
        if norms[index] > 1e-8 and np.max(np.abs(correlation[index, chosen])) < 0.9995:
            chosen.append(index)
            if len(chosen) == max(BUDGETS):
                break
    return chosen


def run_fold(root: Path, fold_name: str) -> None:
    from filelock import FileLock

    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    folder = root / "artifacts/feature_budget" / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(folder / "pipeline.lock"), timeout=1),
        Run(root, "budget-" + fold_name) as run,
    ):
        snapshot = feature_research_snapshot(root)
        probe = probe_module(root)
        plan = json.loads((root / "artifacts/nonlinear_probe/plan.json").read_text())
        if plan["script"] != sha256(root / "scripts/nonlinear_probe.py"):
            raise ValueError("The parent probe is stale.")
        fold = next(f for f in plan["folds"] if f["name"] == fold_name)
        models = [
            json.loads((root / "artifacts" / label / fold_name / "models.json").read_text())
            for label in ("research", "context", "representation")
        ]
        parent = probe.feature_sets(*models)["all_engineered"]
        names = candidate_pool(root, fold_name, parent)
        parameters = {
            "script": sha256(Path(__file__)),
            "lock": sha256(Path(__file__).with_suffix(".py.lock")),
            "helper": sha256(root / "scripts/ablate_features.py"),
            "representation_sources": snapshot["source_signatures"],
            "parent": plan["source_signature"],
            "fold": fold,
            "pool": names,
            "budgets": BUDGETS,
            "settings": probe.SETTINGS,
        }
        source = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        atomic_json(folder / "plan.json", {"source_signature": source, **parameters})
        caches, _ = verify_inputs(root)
        training = probe.materialize(root, caches, fold, names, True)
        indices = retain_columns(training[0], len(parent))
        evaluation = probe.materialize(root, caches, fold, names, False)
        parent_errors = root / "artifacts/nonlinear_probe" / fold_name / "all_engineered.csv"
        receipt = json.loads(
            (root / ".state" / f"nonlinear-{fold_name}-all_engineered-evaluate.json").read_text()
        )
        if (
            receipt.get("status") != "completed"
            or receipt.get("signature") != plan["source_signature"]
            or receipt.get("outputs", {}).get(str(parent_errors.relative_to(root)))
            != sha256(parent_errors)
        ):
            raise ValueError("A verified parent evaluation is required.")
        reference = pd.read_csv(parent_errors)
        bootstrap = bootstrap_scores(reference)
        rows = [
            {"model": "all_engineered", "feature_count": len(parent), **error_metrics(reference)}
        ]
        for budget in BUDGETS:
            chosen = indices[:budget]
            name = "budget_" + str(budget)
            fitted_path, output = folder / (name + ".pkl"), folder / (name + ".csv")

            def fit_action(chosen: list[int] = chosen, fitted_path: Path = fitted_path) -> None:
                model = probe.train_pair(training[0][:, chosen], training[1])
                atomic_bytes(fitted_path, pickle.dumps(model, protocol=5))

            stage(root, f"budget-{fold_name}-{name}-fit", source, [fitted_path], fit_action, run)

            def evaluate_action(
                chosen: list[int] = chosen,
                fitted_path: Path = fitted_path,
                output: Path = output,
                name: str = name,
            ) -> None:
                model = pickle.loads(fitted_path.read_bytes())
                prediction = np.column_stack([m.predict(evaluation[0][:, chosen]) for m in model])
                errors = error_rows(evaluation, prediction, name)
                atomic_bytes(output, errors.to_csv(index=False).encode())

            stage(
                root, f"budget-{fold_name}-{name}-evaluate", source, [output], evaluate_action, run
            )
            errors = pd.read_csv(output)
            rows.append(
                {
                    "model": name,
                    "feature_count": len(chosen),
                    "features": [names[i] for i in chosen],
                    **error_metrics(errors),
                    "delta_vs_full_ci95": np.quantile(
                        bootstrap_scores(errors) - bootstrap, [0.025, 0.975]
                    ).tolist(),
                }
            )
        atomic_json(
            folder / "summary.json",
            {
                "status": "passed",
                "source_signature": source,
                "fold": fold,
                "models": rows,
                "rows": len(reference),
                "holdout_evaluation": "not_run",
                "candidate_pool": len(names),
                "retained_after_redundancy": len(indices),
                "redundancy": "8192 deterministic training rows; 0.9995 correlation threshold",
                "parent_features_preserved": True,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold", required=True, choices=["inner_1", "inner_2", "inner_3", "development"]
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    run_fold(root, args.fold)
