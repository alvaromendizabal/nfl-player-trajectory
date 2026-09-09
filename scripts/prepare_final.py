"""Verify and freeze final-training inputs; no fitting or holdout scoring occurs here."""

import argparse
from pathlib import Path

from nfl_trajectory.final_protocol import prepare
from nfl_trajectory.runtime import Run, atomic_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish", action="store_true", help="Publish the verified compact protocol."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with Run(root, "prepare-final") as run:
        summary = prepare(root, run)
        if args.publish:
            atomic_json(root / "docs/results/final_protocol.json", summary)
            run.event("final_protocol_published", source_signature=summary["source_signature"])
