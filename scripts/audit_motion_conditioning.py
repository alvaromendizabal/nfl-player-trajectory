"""Measure training-target conditioning without fitting or reading evaluation labels."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from domain_research import specification  # noqa: E402

from nfl_trajectory.runtime import Run, atomic_json, sha256  # noqa: E402


def main() -> None:
    with Run(ROOT, "motion-conditioning") as run:
        parent = specification()
        folds = []
        for fold in parent["folds"]:
            path = ROOT / f"artifacts/temporal/research/{fold['name']}/samples.pkl"
            samples = pickle.loads(path.read_bytes())
            training = [s for s in samples if str(s["split"]) == "train"]
            time = np.concatenate([s["time"] for s in training])
            residual = np.concatenate([s["truth"] for s in training]).astype(float)
            baseline = np.concatenate([s["baseline"] for s in training]).astype(float)
            displacement = baseline + residual
            keys = np.concatenate([s["keys"] for s in training])
            tail = time > 2
            sse = (residual**2).sum(-1)
            index = int(np.argmax(np.abs(baseline).max(-1)))
            folds.append(
                {
                    "fold": fold["name"],
                    "training_plays": len(training),
                    "training_rows": len(time),
                    "maximum_horizon_seconds": float(time.max()),
                    "ridge_residual_rmse": float(np.sqrt((residual**2).mean())),
                    "raw_displacement_rms": float(np.sqrt((displacement**2).mean())),
                    "maximum_abs_ridge_displacement": float(np.abs(baseline).max()),
                    "maximum_abs_observed_future_displacement": float(np.abs(displacement).max()),
                    "after_two_seconds_row_share": float(tail.mean()),
                    "after_two_seconds_ridge_error_share": float(sse[tail].sum() / sse.sum()),
                    "largest_ridge_displacement_request": keys[index].tolist(),
                    "ridge_displacement_at_request": baseline[index].tolist(),
                    "truth_displacement_at_request": displacement[index].tolist(),
                }
            )
        report = {
            "status": "completed",
            "run_id": run.run_id,
            "scope": "training targets only; no fit or change to any scientific arm",
            "input_hashes": {
                k: v for k, v in parent["inputs"].items() if k.endswith("samples.pkl")
            },
            "sources": {"scripts/audit_motion_conditioning.py": sha256(Path(__file__))},
            "folds": folds,
        }
        atomic_json(ROOT / "docs/results/motion_conditioning.json", report)
        run.event("motion_conditioning_completed", folds=folds)


if __name__ == "__main__":
    main()
