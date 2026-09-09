# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Fit final frozen schemas, or exercise actual sklearn fitting and recovery on a fixture."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nonlinear_probe import SETTINGS
from prepare_tree import portable


def fit_axis(matrix: np.ndarray, target: np.ndarray, settings: dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    if settings != SETTINGS:
        raise ValueError("Final estimator settings differ from the validated fixed capacity.")
    if matrix.ndim != 2 or target.shape != (len(matrix),) or not np.isfinite(target).all():
        raise ValueError("Final estimator requires finite, aligned coordinate targets.")
    return HistGradientBoostingRegressor(**settings).fit(matrix, target)


def self_test(root: Path) -> dict[str, Any]:
    import sklearn

    from nfl_trajectory.final_fit import fit_profile
    from nfl_trajectory.runtime import Run, sha256

    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(1024, 20)).astype(np.float32)
    matrix[:, 16:] = 0
    names = [f"fixture_{i}" for i in range(20)]
    targets = np.column_stack([matrix[:, 0] ** 2 + matrix[:, 1], np.sin(matrix[:, 2])])
    arrays = {
        "residual": targets,
        "baseline": np.zeros_like(targets),
        "sign": np.ones((len(matrix), 1)),
        "keys": np.column_stack([np.arange(len(matrix)), np.ones((len(matrix), 3), dtype=int)]),
    }
    attempts = 0

    def interrupted(x: np.ndarray, y: np.ndarray, settings: dict[str, Any]) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("Controlled y-coordinate interruption")
        return fit_axis(x, y, settings)

    with tempfile.TemporaryDirectory(prefix="nfl-final-fit-") as temporary:
        workspace = Path(temporary)
        with Run(workspace, "final-fit-self-test") as run:
            try:
                fit_profile(
                    workspace,
                    "fixture",
                    names,
                    "fixture",
                    SETTINGS,
                    lambda: (matrix, arrays),
                    interrupted,
                    portable,
                    run,
                )
            except RuntimeError as exc:
                if str(exc) != "Controlled y-coordinate interruption":
                    raise
            else:
                raise AssertionError("The controlled interruption did not occur.")
            path = workspace / "artifacts/final/models/fixture/x.pkl"
            before = sha256(path), path.stat().st_mtime_ns
            result = fit_profile(
                workspace,
                "fixture",
                names,
                "fixture",
                SETTINGS,
                lambda: (matrix, arrays),
                interrupted,
                portable,
                run,
            )
            assert (sha256(path), path.stat().st_mtime_ns) == before
            assert attempts == 3 and result["active_features"] <= 16
            assert result["portable_max_absolute_difference"] <= 1e-12

            def unnecessary() -> Any:
                raise AssertionError("Completed fits must not materialize training data again.")

            assert (
                fit_profile(
                    workspace,
                    "fixture",
                    names,
                    "fixture",
                    SETTINGS,
                    unnecessary,
                    fit_axis,
                    portable,
                    run,
                )
                == result
            )
        events = [json.loads(line) for line in run.log_path.read_text().splitlines()]
        return {
            "status": "passed",
            "scope": "synthetic numerical integration; not competition fitting",
            "rows": len(matrix),
            "schema_features": len(names),
            "active_features": result["active_features"],
            "portable_max_absolute_difference": result["portable_max_absolute_difference"],
            "interrupted_axis_recovered": True,
            "completed_axis_preserved": True,
            "repeat_without_materialization": True,
            "stage_reuses": sum(e["event"] == "stage_reused" for e in events),
            "scikit_learn_version": sklearn.__version__,
            "holdout_evaluation": "not_run",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from nfl_trajectory.final_fit import fit_models
    from nfl_trajectory.runtime import Run, atomic_json

    if args.self_test:
        atomic_json(root / "artifacts/quality/final_fit.json", self_test(root))
    elif args.validate_data:
        from nfl_trajectory.feature_research import verify_inputs
        from nfl_trajectory.final_features import validate_observed_features

        with Run(root, "final-feature-validation") as run:
            caches, _ = verify_inputs(root)
            report = validate_observed_features(root, caches, run)
            if args.publish:
                atomic_json(root / "docs/results/final_feature_validation.json", report)
    else:
        import sklearn

        with Run(root, "final-fit") as run:
            report = fit_models(
                root,
                run,
                fit_axis,
                portable,
                {
                    "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                },
            )
            if args.publish:
                atomic_json(root / "docs/results/final_fit.json", report)


if __name__ == "__main__":
    main()
