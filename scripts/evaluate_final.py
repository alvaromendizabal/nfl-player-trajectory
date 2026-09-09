"""Seal final predictions, then separately score the same frozen holdout evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from nfl_trajectory.final_evaluation import evaluate_sealed, seal_predictions
    from nfl_trajectory.runtime import Run, atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("seal", "evaluate"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    with Run(ROOT, "final-" + args.phase) as run:
        report = seal_predictions(ROOT, run) if args.phase == "seal" else evaluate_sealed(ROOT, run)
        if args.publish and args.phase == "evaluate":
            atomic_json(ROOT / "docs/results/final_evaluation.json", report)


if __name__ == "__main__":
    main()
