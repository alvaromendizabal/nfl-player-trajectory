"""Review the feature completion gate from current, verified experiment artifacts."""

from pathlib import Path

from nfl_trajectory.research_gate import review
from nfl_trajectory.runtime import Run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with Run(root, "feature-gate-review") as run:
        report = review(root, run)
        run.event(
            "feature_gate_reviewed", feature_gate=report["feature_gate"], checks=report["checks"]
        )
