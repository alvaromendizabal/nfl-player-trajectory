# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Matched joint linear refits isolate optional-input dependence from fitting order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ablate_features import error_rows, probe_module


def run_fold(root: Path, fold_name: str) -> None:
    from filelock import FileLock

    from nfl_trajectory.benchmark import error_metrics
    from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
    from nfl_trajectory.feature_research import predict_linear, solve_ridge, verify_inputs
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    folder = root / "artifacts/joint_linear" / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(root, "joint-" + fold_name) as run:
        probe = probe_module(root)
        snapshot = feature_research_snapshot(root)
        parent = json.loads((root / "artifacts/nonlinear_probe/plan.json").read_text())
        if parent["script"] != sha256(root / "scripts/nonlinear_probe.py"):
            raise ValueError("Joint refits require the current representation probe.")
        fold = next(f for f in parent["folds"] if f["name"] == fold_name)
        models = [
            json.loads((root / "artifacts" / label / fold_name / "models.json").read_text())
            for label in ("research", "context", "representation")
        ]
        names = probe.feature_sets(*models)["all_engineered"]
        variants = {
            "joint_all": names,
            "joint_without_metadata": [n for n in names if not metadata_dependent(n)],
            "joint_positional": [
                n for n in names if not telemetry_dependent(n) and not metadata_dependent(n)
            ],
        }
        plan = {
            "script": sha256(Path(__file__)),
            "lock": sha256(Path(__file__).with_suffix(".py.lock")),
            "helper": sha256(root / "scripts/ablate_features.py"),
            "contracts": sha256(root / "src/nfl_trajectory/feature_contracts.py"),
            "representation_sources": snapshot["source_signatures"],
            "parent": parent["source_signature"],
            "fold": fold,
            "variants": variants,
            "alpha": 0.01,
        }
        source = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        atomic_json(folder / "plan.json", {"source_signature": source, **plan})
        caches, _ = verify_inputs(root)
        training = probe.materialize(root, caches, fold, names, True)
        evaluation = probe.materialize(root, caches, fold, names, False)
        rows = []
        for name, columns in variants.items():
            indices = [names.index(n) for n in columns]
            fitted_path, output = folder / (name + ".json"), folder / (name + ".csv")

            def fit_action(
                indices: list[int] = indices,
                columns: list[str] = columns,
                fitted_path: Path = fitted_path,
            ) -> None:
                x, y = training[0][:, indices].astype(float), training[1]
                fitted = solve_ridge(
                    x.T @ x, x.T @ y, x.sum(0), y.sum(0), len(x), columns, len(columns)
                )
                atomic_json(fitted_path, fitted)

            stage(root, f"joint-{fold_name}-{name}-fit", source, [fitted_path], fit_action, run)
            fitted = json.loads(fitted_path.read_text())

            def evaluate_action(
                output: Path = output, name: str = name, fitted: dict[str, Any] = fitted
            ) -> None:
                indices = [names.index(n) for n in fitted["features"]]
                correction = predict_linear(evaluation[0][:, indices].astype(float), fitted)
                errors = error_rows(evaluation, correction, name)
                atomic_bytes(output, errors.to_csv(index=False).encode())

            stage(
                root, f"joint-{fold_name}-{name}-evaluate", source, [output], evaluate_action, run
            )
            import pandas as pd

            errors = pd.read_csv(output)
            rows.append(
                {"model": name, "feature_count": len(fitted["features"]), **error_metrics(errors)}
            )
        atomic_json(
            folder / "summary.json",
            {
                "status": "passed",
                "source_signature": source,
                "fold": fold,
                "models": rows,
                "rows": len(evaluation[0]),
                "holdout_evaluation": "not_run",
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
