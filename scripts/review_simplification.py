"""Verify and summarize the additional feature study without using holdout data."""

import hashlib
import json
from pathlib import Path

from nfl_trajectory.research_evidence import simplification_evidence
from nfl_trajectory.research_inference import research_bundle

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    bundle = research_bundle(root)
    bundle_hash = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    report = simplification_evidence(root, bundle["source_signatures"], bundle_hash)
    if report is None:
        raise ValueError("The combined omission study has not run.")
    print(json.dumps(report["decision"]))
