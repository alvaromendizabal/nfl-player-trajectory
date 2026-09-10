"""Bounded preflight for reconstructed motion supervision; performs zero fits."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.motion_targets import fit_motion_scales, motion_targets  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_json, sha256  # noqa: E402

SAMPLE = "artifacts/temporal/research/inner_1/samples.pkl"
EXPECTED = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"


def execute(run: Run) -> None:
    actual = sha256(ROOT / SAMPLE)
    if actual != EXPECTED:
        raise ValueError("Private sample cache differs from the verified parent backup.")
    sources = [
        "scripts/audit_motion_targets.py",
        "src/nfl_trajectory/motion_targets.py",
        "src/nfl_trajectory/temporal_data.py",
        "src/nfl_trajectory/runtime.py",
        "tests/test_motion_targets.py",
        "docs/MOTION_TARGET_AUDIT.md",
    ]
    spec = {
        "input": {SAMPLE: actual},
        "sources": {name: sha256(ROOT / name) for name in sources},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "max_seconds": 300,
    }
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = ROOT / "docs/results/motion_target_audit.json"
    receipt = ROOT / "docs/results/motion_target_audit_seal.json"
    if output.exists() or receipt.exists():
        if not (output.exists() and receipt.exists()):
            raise ValueError("Partial audit publication; preserve it for inspection.")
        old, seal = json.loads(output.read_text()), json.loads(receipt.read_text())
        if old["signature"] != signature or seal["sha256"] != sha256(output):
            raise ValueError("Audit source, runtime, inputs or result changed.")
        run.event("audit_reused", report_sha256=seal["sha256"], new_fits=0)
        return
    started = time.monotonic()
    # Load only the user's original hash-verified private cache.
    samples = pickle.loads((ROOT / SAMPLE).read_bytes())
    splits = {str(s["split"]) for s in samples}
    if splits != {"train", "validation"}:
        raise ValueError("Unexpected sample splits.")
    train_games = {int(s["keys"][0, 0]) for s in samples if str(s["split"]) == "train"}
    validation_games = {int(s["keys"][0, 0]) for s in samples if str(s["split"]) == "validation"}
    if train_games & validation_games or max(train_games) // 100 >= min(validation_games) // 100:
        raise ValueError("Train and validation dates overlap or are out of order.")
    scales = fit_motion_scales(s for s in samples if str(s["split"]) == "train")
    run.event("training_scales_fitted", scales=scales, new_fits=0)
    stats: dict[str, Any] = {}
    magnitudes: dict[str, dict[str, list[np.ndarray]]] = {}
    for split in sorted(splits):
        stats[split] = {"plays": 0, "rows": 0, "velocity_labels": 0, "acceleration_labels": 0}
        magnitudes[split] = {name: [] for name in scales}
    for i, sample in enumerate(samples):
        split = str(sample["split"])
        labels = motion_targets(sample)
        stats[split]["plays"] += 1
        stats[split]["rows"] += len(sample["keys"])
        for name in scales:
            mask = labels[name + "_mask"]
            stats[split][name + "_labels"] += int(mask.sum())
            magnitudes[split][name].append(np.linalg.norm(labels[name][mask], axis=1))
        if (i + 1) % 1000 == 0:
            run.event("plays_audited", completed=i + 1, total=len(samples))
    for split in stats:
        for name in scales:
            values = np.concatenate(magnitudes[split][name])
            stats[split][name + "_vector_magnitude"] = dict(
                zip(
                    ("p50", "p95", "p99", "max"),
                    np.quantile(values, [0.5, 0.95, 0.99, 1]).tolist(),
                    strict=True,
                )
            )
            stats[split][name + "_coverage"] = stats[split][name + "_labels"] / stats[split]["rows"]
    stats["train"]["games"], stats["validation"]["games"] = len(train_games), len(validation_games)
    result = {
        "status": "passed",
        "signature": signature,
        "specification": spec,
        "training_only_rms_scales": scales,
        "support": stats,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "new_model_fits": 0,
        "new_rmse": None,
        "feature_completion_gate": "open",
        "decision": (
            "Label construction verified; predictive utility still requires matched model fits."
        ),
        "private_parent_backup": "docs/results/soft_coverage_backup.json",
    }
    atomic_json(output, result)
    atomic_json(receipt, {"sha256": sha256(output), "signature": signature})
    run.event("audit_saved", plays=len(samples), report_sha256=sha256(output), new_fits=0)


def main() -> None:
    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("Motion target audit exceeded its 300-second bound.")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        with Run(ROOT, "motion-target-audit") as run:
            execute(run)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
