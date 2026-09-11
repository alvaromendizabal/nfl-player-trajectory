"""Training-only runtime measurement; no validation score or retained fitted model."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nfl_trajectory.motion_supervision import MatchedState, supervised_batch
from nfl_trajectory.runtime import atomic_json, sha256
from nfl_trajectory.supervision_batches import TrainingBatches
from nfl_trajectory.supervision_evidence import runtime_identity

SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"


def profile_training(root: Path) -> dict[str, Any]:
    """Measure both fixed arms on representative and longest training-only batches."""
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cache = root / "artifacts/temporal/research/inner_1/samples.pkl"
    if sha256(cache) != SAMPLE_SHA256:
        raise ValueError("Existing private sample checksum differs; no download or fitting.")
    saved = json.loads(
        (root / "artifacts/motion_supervision/preflight/training_plan.json").read_text()
    )
    if saved["sample_sha256"] != SAMPLE_SHA256:
        raise ValueError("Training plan and sample cache differ.")
    for name, digest in saved["source_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError("Numerical preflight source changed: " + name)
    sources = [
        "src/nfl_trajectory/supervision_profile.py",
        "src/nfl_trajectory/supervision_batches.py",
        "src/nfl_trajectory/motion_supervision.py",
        "src/nfl_trajectory/temporal_model.py",
        "src/nfl_trajectory/temporal_data.py",
        "src/nfl_trajectory/motion_targets.py",
    ]
    identity = {"sources": {p: sha256(root / p) for p in sources}, "runtime": runtime_identity()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = root / "artifacts/motion_supervision/profile/result.json"
    if destination.is_file():
        old = json.loads(destination.read_text())
        if old["signature"] != signature:
            raise ValueError("Prior profile uses different source/runtime; preserve and review.")
        return {**old, "reused": True}
    with cache.open("rb") as stream:
        samples = pickle.load(stream)
    training = [sample for sample in samples if str(sample["split"]) == "train"]
    batches = TrainingBatches(training, 64)
    plan = saved["plan"]
    if (len(training), sum(len(s["keys"]) for s in training)) != (4951, 193452):
        raise ValueError("Training population differs from reviewed preflight.")
    if [list(s["keys"][0, :2].astype(int)) for s in batches.samples] != plan["canonical_play_ids"]:
        raise ValueError("Canonical play identity differs from saved plan.")
    selected = [("first", 0), ("middle", 39), ("short_final", 77)]
    stress = sorted(training, key=lambda s: len(s["keys"]), reverse=True)[:64]
    measurements: list[dict[str, Any]] = []
    for name, weight in (("coordinate", 0.0), ("velocity", 0.1)):
        state = MatchedState(96, 2026, weight, plan["training_only_velocity_rms"])
        for label, cursor in selected + [("largest_requested_rows", -1)]:
            if time.monotonic() - started > 45:
                raise TimeoutError("Training-only profile exceeded its 45-second work budget.")
            begin = time.monotonic()
            batch = stress if cursor == -1 else batches.batch(cursor)
            features, labels = supervised_batch(batch)
            stats = state.step(
                features,
                labels,
                plan["coordinate_loss_denominator"],
                plan["velocity_loss_denominator"],
            )
            elapsed = time.monotonic() - begin
            entry: dict[str, Any] = {
                "arm": name,
                "batch": label,
                "plays": len(batch),
                "rows": int(features["scored"].sum()),
                "rows_after_frame_48": sum(int((s["keys"][:, 3] > 48).sum()) for s in batch),
                "max_requested_frame": max(int(s["keys"][:, 3].max()) for s in batch),
                "seconds": elapsed,
                "finite_loss": bool(np.isfinite(stats["loss"])),
            }
            if not entry["finite_loss"]:
                raise ValueError("Nonfinite profile loss.")
            measurements.append(entry)
            print(json.dumps({"event": "training_profile_batch", **entry}), flush=True)
    representative = [m["seconds"] for m in measurements if m["batch"] in ("first", "middle")]
    result = {
        "status": "training_only_throughput_measured",
        "signature": signature,
        **identity,
        "sample_sha256": SAMPLE_SHA256,
        "training_plays": len(training),
        "batch_plays": 64,
        "batches_per_epoch": batches.batches_per_epoch,
        "measurements": measurements,
        "conservative_epoch_seconds": 1.5 * max(representative) * batches.batches_per_epoch,
        "worst_stress_batch_seconds": max(m["seconds"] for m in measurements),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "python": platform.python_version(),
        "elapsed_seconds": time.monotonic() - started,
        "training_only_optimizer_steps": len(measurements),
        "scientific_fits": 0,
        "new_rmse": None,
        "validation_labels_used": False,
        "training_ready": False,
        "feature_completion_gate": "open",
        "interpretation": "Timing and finite-gradient check, not predictive feature evidence.",
    }
    atomic_json(destination, result)
    return result
