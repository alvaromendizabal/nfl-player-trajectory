"""Refit final preprocessing; the frozen tree-profile fits are a subsequent stage."""

import argparse
from pathlib import Path

from nfl_trajectory.final_training import preprocess
from nfl_trajectory.runtime import Run, atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with Run(root, "final-preprocessing") as run:
        summary = preprocess(root, run)
        if args.publish:
            atomic_json(root / "docs/results/final_preprocessing.json", summary)
            run.event("final_preprocessing_published", source_signature=summary["source_signature"])
