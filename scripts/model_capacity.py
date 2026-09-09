# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Run one predeclared capacity comparison; no holdout, final refit, or Kaggle submission."""

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
from nonlinear_probe import SETTINGS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    import sklearn
    from filelock import FileLock
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.feature_experiment import load_week, target_state
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.model_capacity import error_rows, selected_features, validate_partitions
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

    folder = root / "artifacts/model_capacity/development"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(root, "model-capacity") as run:
        caches, inputs = verify_inputs(root)
        splits = pd.read_csv(root / "artifacts/game_splits.csv")
        fitted = json.loads((root / "artifacts/features/model.json").read_text())
        names = selected_features(fitted, splits[splits.split.eq("train")].game_id.tolist())
        settings = {
            "core_shallow": dict(SETTINGS),
            "core_deeper": {**SETTINGS, "max_iter": 400, "max_depth": 8, "max_leaf_nodes": 31},
        }
        parameters = {
            "inputs": inputs,
            "script": sha256(Path(__file__)),
            "lock": sha256(Path(__file__).with_suffix(".py.lock")),
            "helper": sha256(root / "src/nfl_trajectory/model_capacity.py"),
            "parent_settings": sha256(root / "scripts/nonlinear_probe.py"),
            "features": names,
            "settings": settings,
            "threads": 4,
            "environment": {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "scope": (
                "Two predeclared capacities, identical core features, complete development rows."
            ),
        }
        source = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        atomic_json(folder / "plan.json", {"source_signature": source, **parameters})
        design_path = folder / "design.npz"

        def materialize() -> None:
            parts: dict[str, list[np.ndarray]] = {
                k: [] for k in ("x", "truth", "baseline", "sign", "keys", "labels")
            }
            for index, cache in enumerate(caches, 1):
                bank, targets, arrays = load_week(cache)
                validate_partitions(targets, arrays["labels"], splits)
                parts["x"].append(bank.matrix(targets, names))
                parts["truth"].append(arrays["truth"])
                parts["baseline"].append(arrays["ridge"])
                parts["sign"].append(target_state(bank, targets).sign.to_numpy()[:, None])
                parts["keys"].append(targets[KEYS].to_numpy())
                parts["labels"].append(arrays["labels"])
                run.event("materialize_week", completed=index, total=len(caches))
            import io

            buffer = io.BytesIO()
            arrays_to_save: dict[str, Any] = {k: np.concatenate(v) for k, v in parts.items()}
            np.savez_compressed(buffer, **arrays_to_save)
            atomic_bytes(design_path, buffer.getvalue())

        stage(root, "capacity-design", source, [design_path], materialize, run)
        with np.load(design_path, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        train = data["labels"] == "train"
        evaluation = data["labels"] == "validation"
        targets = pd.DataFrame(data["keys"], columns=KEYS)
        validate_partitions(targets, data["labels"], splits)
        x = data["x"][train]
        y = ((data["truth"] - data["baseline"]) * data["sign"])[train]
        evaluation_x = data["x"][evaluation]
        evaluation_keys = targets.loc[evaluation].reset_index(drop=True)
        results: list[dict[str, Any]] = []
        bootstrap: dict[str, np.ndarray] = {}
        with threadpool_limits(limits=4):
            for name, setting in settings.items():
                predictions = []
                for axis in range(2):
                    model_path = folder / f"{name}_{axis}.pkl"

                    def fit(
                        model_path: Path = model_path,
                        axis: int = axis,
                        setting: dict[str, Any] = setting,
                    ) -> None:
                        model = HistGradientBoostingRegressor(**setting).fit(x, y[:, axis])
                        atomic_bytes(model_path, pickle.dumps(model, protocol=5))

                    stage(root, f"capacity-{name}-{axis}", source, [model_path], fit, run)
                    # This path is written by our fit or verified by stage's SHA-256 receipt.
                    model = pickle.loads(model_path.read_bytes())
                    predictions.append(model.predict(evaluation_x))
                errors = error_rows(
                    evaluation_keys,
                    data["baseline"][evaluation],
                    np.column_stack(predictions),
                    data["sign"][evaluation],
                    data["truth"][evaluation],
                )
                atomic_bytes(folder / f"{name}.csv", errors.to_csv(index=False).encode())
                bootstrap[name] = bootstrap_scores(errors)
                metrics = error_metrics(errors)
                results.append({"model": name, "settings": setting, **metrics})
                run.event("model_evaluated", model=name, **metrics)
        delta = bootstrap["core_deeper"] - bootstrap["core_shallow"]
        report = {
            "status": "passed",
            "source_signature": source,
            "run_id": run.run_id,
            "scope": parameters["scope"],
            "feature_count": len(names),
            "training_games": int(targets.loc[train, "game_id"].nunique()),
            "training_rows": int(train.sum()),
            "development_games": int(evaluation_keys.game_id.nunique()),
            "development_rows": int(evaluation.sum()),
            "models": results,
            "paired_game_bootstrap_delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "frozen_full_feature_development_rmse": 0.6880521879583761,
            "comparison_note": (
                "The frozen full-feature model uses a different representation; only the two "
                "core arms isolate capacity. Development has already been inspected; bootstrap "
                "uncertainty is conditional on this experiment."
            ),
            "holdout_evaluation": "not_run",
            "kaggle_submission": "not_run",
            "promotion": "Requires chronological inner-fold confirmation and inference validation.",
            "artifacts": {
                p.relative_to(root).as_posix(): sha256(p)
                for p in sorted(folder.iterdir())
                if p.suffix in {".pkl", ".csv", ".npz", ".json"} and p.name != "summary.json"
            },
        }
        atomic_json(folder / "summary.json", report)
        if args.publish:
            atomic_json(root / "docs/results/model_capacity.json", report)


if __name__ == "__main__":
    main()
