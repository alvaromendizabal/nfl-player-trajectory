"""Audit at most 32 existing training plays; no labels, fits or cloud writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from nfl_trajectory.temporal_edges import NAMES, from_sample

SAMPLE_SHA = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"


def audit(cache: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    if not cache.is_file() or cache.is_symlink():
        raise ValueError("The existing regular sample cache is required")
    hasher = hashlib.sha256()
    with cache.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != SAMPLE_SHA:
        raise ValueError("Sample SHA256 mismatch; do not unpickle an unverified cache")
    with cache.open("rb") as stream:
        samples = pickle.load(stream)
    training = [s for s in samples if str(s["split"]) == "train"]
    if len(training) != 4951:
        raise ValueError("Training population differs from the frozen cache")
    training.sort(key=lambda s: tuple(s["keys"][0, :2]))
    valid = np.zeros(len(NAMES), dtype=np.int64)
    support = 0
    maximum_bytes = 0
    output.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(training[:32]):
        if time.monotonic() - started > 120:
            raise TimeoutError("Training-only feature smoke exceeded 120 seconds")
        edges = from_sample(sample)
        valid += edges["valid"].sum(axis=(0, 1, 2))
        support += int(edges["pair_valid"].sum())
        maximum_bytes = max(maximum_bytes, sum(x.nbytes for x in edges.values()))
        if index == 0:
            np.savez_compressed(output / "edges_example.npz", **edges)
        if (index + 1) % 8 == 0:
            print(json.dumps({"event": "feature_smoke", "plays": index + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
    result = {"status": "training_only_feature_smoke_passed", "plays": 32,
              "sample_sha256": SAMPLE_SHA, "channel_names": list(NAMES),
              "example_sha256": hashlib.sha256((output / "edges_example.npz").read_bytes()).hexdigest(),
              "channel_valid_counts": valid.tolist(), "valid_pair_frames": support,
              "max_tensor_bytes": maximum_bytes, "validation_scored": False,
              "scientific_fits": 0, "new_rmse": None, "feature_research": "open",
              "elapsed_seconds": round(time.monotonic() - started, 3)}
    (output / "feature_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.cache, args.output)


if __name__ == "__main__":
    main()
