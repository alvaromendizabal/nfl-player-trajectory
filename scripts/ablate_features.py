# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Strict nonlinear family removals and a telemetry-independent linear fallback."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MASKS = {
    "without_reachability": {"reachability"},
    "without_role_response": {"role_response"},
    "without_route_representation": {"route_representation"},
    "without_role_destination": {"role_destination"},
    "without_graph_pool": {"graph_pool"},
    "without_matched_history": {"matched_history"},
    "without_player_metadata": {"metadata", "player_history"},
}


def probe_module(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "feature_probe", root / "scripts/nonlinear_probe.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("The fixed nonlinear probe is required.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error_rows(evaluation: tuple[Any, ...], correction: np.ndarray, name: str) -> pd.DataFrame:
    frame = evaluation[5].copy()
    frame[["dx", "dy"]] = evaluation[2] + evaluation[4] * correction - evaluation[3]
    frame["model"] = name
    return frame


def run_fold(root: Path, fold_name: str) -> None:
    from filelock import FileLock

    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
    from nfl_trajectory.feature_research import predict_linear, solve_ridge, verify_inputs
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    destination = root / "artifacts/feature_ablation" / fold_name
    destination.mkdir(parents=True, exist_ok=True)
    with (
        FileLock(str(destination / "pipeline.lock"), timeout=1),
        Run(root, "ablation-" + fold_name) as run,
    ):
        snapshot = feature_research_snapshot(root)
        parent_plan = json.loads((root / "artifacts/nonlinear_probe/plan.json").read_text())
        if parent_plan["script"] != sha256(root / "scripts/nonlinear_probe.py"):
            raise ValueError("Parent nonlinear experiment source is stale.")
        fold = next(f for f in parent_plan["folds"] if f["name"] == fold_name)
        models = {
            name: json.loads((root / "artifacts" / name / fold_name / "models.json").read_text())
            for name in ("research", "context", "representation")
        }
        probe = probe_module(root)
        names = probe.feature_sets(models["research"], models["context"], models["representation"])[
            "all_engineered"
        ]
        catalog = pd.concat(
            [research_catalog(), context_catalog(), representation_catalog()], ignore_index=True
        )
        families = dict(zip(catalog.feature, catalog.family, strict=True))
        parameter = {
            "parent": parent_plan["source_signature"],
            "representations": snapshot["source_signatures"],
            "script": sha256(Path(__file__)),
            "lock": sha256(Path(__file__).with_suffix(".py.lock")),
            "contracts": sha256(root / "src/nfl_trajectory/feature_contracts.py"),
            "fold": fold,
            "masks": {k: sorted(v) for k, v in MASKS.items()},
        }
        source = hashlib.sha256(json.dumps(parameter, sort_keys=True).encode()).hexdigest()
        atomic_json(destination / "plan.json", {"source_signature": source, **parameter})
        caches, _ = verify_inputs(root)
        training = probe.materialize(root, caches, fold, names, True)
        evaluation = probe.materialize(root, caches, fold, names, False)
        reference_path = root / "artifacts/nonlinear_probe" / fold_name / "all_engineered.pkl"
        receipt = json.loads(
            (root / ".state" / f"nonlinear-{fold_name}-all_engineered-fit.json").read_text()
        )
        if receipt.get("signature") != parent_plan["source_signature"]:
            raise ValueError("Parent model checkpoint belongs to another experiment.")
        if receipt.get("status") != "completed" or receipt.get("outputs", {}).get(
            str(reference_path.relative_to(root))
        ) != sha256(reference_path):
            raise ValueError("The fixed parent model has no verified completed checkpoint.")
        parent = pickle.loads(reference_path.read_bytes())
        reference = error_rows(
            evaluation,
            np.column_stack([m.predict(evaluation[0]) for m in parent]),
            "all_engineered",
        )
        reference_metrics = error_metrics(reference)
        reference_bootstrap = bootstrap_scores(reference)
        rows = [
            {
                "model": "all_engineered",
                "feature_count": len(names),
                **reference_metrics,
                "delta_vs_full_ci95": [0.0, 0.0],
            }
        ]
        for name, removed in MASKS.items():
            indices = [i for i, f in enumerate(names) if families[f] not in removed]
            output = destination / (name + ".csv")
            fitted_path = destination / (name + ".pkl")

            def fit_action(indices: list[int] = indices, fitted_path: Path = fitted_path) -> None:
                fitted = probe.train_pair(training[0][:, indices], training[1])
                atomic_bytes(fitted_path, pickle.dumps(fitted, protocol=5))

            stage(root, f"ablation-{fold_name}-{name}-fit", source, [fitted_path], fit_action, run)

            def evaluate_action(
                indices: list[int] = indices,
                fitted_path: Path = fitted_path,
                output: Path = output,
                name: str = name,
            ) -> None:
                fitted = pickle.loads(fitted_path.read_bytes())
                predicted = np.column_stack([m.predict(evaluation[0][:, indices]) for m in fitted])
                frame = error_rows(evaluation, predicted, name)
                atomic_bytes(output, frame.to_csv(index=False).encode())

            stage(
                root,
                f"ablation-{fold_name}-{name}-evaluate",
                source,
                [output],
                evaluate_action,
                run,
            )
            frame = pd.read_csv(output)
            metrics = error_metrics(frame)
            interval = np.quantile(
                bootstrap_scores(frame) - reference_bootstrap, [0.025, 0.975]
            ).tolist()
            rows.append(
                {
                    "model": name,
                    "feature_count": len(indices),
                    **metrics,
                    "delta_vs_full_ci95": interval,
                }
            )
        safe = [
            i
            for i, name in enumerate(names)
            if not telemetry_dependent(name) and not metadata_dependent(name)
        ]
        safe_names = [names[i] for i in safe]
        fallback_path = destination / "robust_linear.json"
        fallback_errors = destination / "robust_linear.csv"

        def fallback_action() -> None:
            x, y = training[0][:, safe].astype(float), training[1]
            fitted = solve_ridge(
                x.T @ x, x.T @ y, x.sum(0), y.sum(0), len(x), safe_names, len(safe_names)
            )
            atomic_json(
                fallback_path,
                {
                    "model": fitted,
                    "fold": fold,
                    "source_signatures": snapshot["source_signatures"],
                    "contract_sha256": parameter["contracts"],
                    "parent_source": parent_plan["source_signature"],
                    "purpose": (
                        "Fallback with no optional telemetry or player-metadata dependencies."
                    ),
                },
            )

        stage(
            root,
            f"ablation-{fold_name}-robust-linear-fit",
            source,
            [fallback_path],
            fallback_action,
            run,
        )
        fallback = json.loads(fallback_path.read_text())["model"]
        positions = {name: i for i, name in enumerate(names)}
        correction = predict_linear(
            evaluation[0][:, [positions[n] for n in fallback["features"]]].astype(float), fallback
        )
        frame = error_rows(evaluation, correction, "robust_linear")
        stage(
            root,
            f"ablation-{fold_name}-robust-linear-evaluate",
            source,
            [fallback_errors],
            lambda: atomic_bytes(fallback_errors, frame.to_csv(index=False).encode()),
            run,
        )
        rows.append(
            {
                "model": "robust_linear",
                "feature_count": len(fallback["features"]),
                **error_metrics(frame),
            }
        )
        atomic_json(
            destination / "summary.json",
            {
                "status": "passed",
                "source_signature": source,
                "fold": fold,
                "models": rows,
                "holdout_evaluation": "not_run",
                "rows": len(reference),
                "settings": probe.SETTINGS,
                "reference": "all_engineered; fixed estimator capacity; no replacement columns",
                "source_signatures": snapshot["source_signatures"],
            },
        )
        run.event("ablation_fold_completed", fold=fold_name, rows=len(reference), models=len(rows))


def summarize_all(root: Path) -> None:
    from nfl_trajectory.runtime import atomic_json

    folder = root / "artifacts/feature_ablation"
    results = [
        json.loads((folder / name / "summary.json").read_text())
        for name in ("inner_1", "inner_2", "inner_3", "development")
    ]
    if any(r["status"] != "passed" or r["holdout_evaluation"] != "not_run" for r in results):
        raise ValueError("All ablation folds must complete before summarization.")
    models = [r["model"] for r in results[0]["models"]]
    scores = {}
    for name in models:
        scores[name] = float(
            np.sqrt(
                sum(
                    next(m["coordinate_rmse_yards"] for m in r["models"] if m["model"] == name) ** 2
                    * r["rows"]
                    for r in results[:3]
                )
                / sum(r["rows"] for r in results[:3])
            )
        )
    summary = {
        **results[-1],
        "inner_folds": results[:3],
        "inner_scores": scores,
        "feature_gate": "open",
        "selection": "diagnostic removals; no automatic final-model promotion",
    }
    atomic_json(folder / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["inner_1", "inner_2", "inner_3", "development"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    if args.fold:
        run_fold(root, args.fold)
    else:
        summarize_all(root)
