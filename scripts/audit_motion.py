"""Inspect the largest training-error play without dropping any training or test row."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from nfl_trajectory.runtime import Run, atomic_json, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with Run(root, "motion-training-audit") as run:
        path = root / "artifacts/temporal/research/inner_1/samples.pkl"
        plan = json.loads((root / "artifacts/domain_research/plan.json").read_text())
        expected = plan["inputs"][str(path.relative_to(root))]
        if sha256(path) != expected:
            raise ValueError("Training-audit input differs from the verified fold cache.")
        samples = pickle.loads(path.read_bytes())
        sample = next(s for s in samples if tuple(s["keys"][0, :2]) == (2023091100, 3167))
        if str(sample["split"]) != "train":
            raise ValueError("This diagnostic must use a training play.")
        truth = sample["xy"][sample["player"]] + sample["baseline"] + sample["truth"]
        cv = (
            sample["xy"][sample["player"]]
            + sample["velocity"][sample["player"]] * sample["time"][:, None]
        )
        ridge = sample["xy"][sample["player"]] + sample["baseline"]
        step_speeds = []
        for player in np.unique(sample["player"]):
            mask = sample["player"] == player
            times = sample["time"][mask]
            order = np.argsort(times)
            step_speeds.extend(
                np.linalg.norm(np.diff(truth[mask][order], axis=0), axis=1) / np.diff(times[order])
            )
        record = {
            "game_id": 2023091100,
            "play_id": 3167,
            "selection": "Largest frozen attention training-error play; descriptive audit only",
            "partition": str(sample["split"]),
            "maximum_seconds": float(sample["time"].max()),
            "requested_rows": len(truth),
            "scored_players": int(len(np.unique(sample["player"]))),
            "input_players": len(sample["ids"]),
            "target_x_range": [float(truth[:, 0].min()), float(truth[:, 0].max())],
            "target_y_range": [float(truth[:, 1].min()), float(truth[:, 1].max())],
            "cv_rmse": float(np.sqrt(np.mean((cv - truth) ** 2))),
            "ridge_rmse": float(np.sqrt(np.mean((ridge - truth) ** 2))),
            "maximum_position_step_speed": float(max(step_speeds)),
            "landing": (sample["xy"][0] + sample["static"][0, 2:4] * 20).tolist(),
            "source_sha256": expected,
            "evaluator_sha256": sha256(Path(__file__)),
            "no_training_or_evaluation_row_removed": True,
        }
        atomic_json(root / "docs/results/motion_training_audit.json", record)
        run.event("training_play_audited", **record)


if __name__ == "__main__":
    main()
