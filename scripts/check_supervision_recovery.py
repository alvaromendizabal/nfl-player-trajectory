"""Prove exact synthetic continuation after an independent private S3 download.

Run prepare and verify in separate processes in the locked motion-supervision
runtime. The transfer phase uses the existing Studio execution role; no credentials
or private competition data belong in this test or its public summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import torch  # noqa: E402
from test_motion_supervision import assert_tree_equal, batch_for_cursor  # noqa: E402

from nfl_trajectory.motion_supervision import MatchedState  # noqa: E402
from nfl_trajectory.runtime import atomic_json, sha256  # noqa: E402
from nfl_trajectory.supervision_evidence import (  # noqa: E402
    load_generation,
    runtime_identity,
    save_generation,
)

SOURCES = [
    "scripts/check_supervision_recovery.py",
    "src/nfl_trajectory/motion_supervision.py",
    "src/nfl_trajectory/supervision_evidence.py",
    "src/nfl_trajectory/temporal_model.py",
    "src/nfl_trajectory/temporal_data.py",
    "src/nfl_trajectory/motion_targets.py",
    "tests/test_motion_supervision.py",
    "tests/test_temporal_data.py",
]


def stop_timeout(*_: Any) -> None:
    raise TimeoutError("Recovery proof exceeded its 90-second process budget.")


def execute(folder: Path, stage: str) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    folder = folder.resolve()
    if not folder.is_relative_to((ROOT / "artifacts/quality").resolve()):
        raise ValueError("Recovery evidence must stay under artifacts/quality.")
    specification = {
        "scope": "synthetic engineering; zero private-data scientific fits",
        "sources": {name: sha256(ROOT / name) for name in SOURCES},
        "runtime": runtime_identity(),
        "width": 96,
        "seed": 2026,
        "steps": 6,
        "interruption_after_steps": 2,
        "velocity_weights": [0.0, 0.1],
        "scale": 3.0,
    }
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    manifest_path = folder / "manifest.json"
    if stage == "prepare":
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["signature"] != signature:
                raise ValueError("Different recovery specification; existing work preserved.")
            for name in ("coordinate", "velocity"):
                load_generation(folder / "upload" / name, signature)
                load_generation(folder / "expected" / name, signature)
            return {**old, "reused": True}
        folder.mkdir(parents=True, exist_ok=True)
        arms = {}
        for name, weight in (("coordinate", 0.0), ("velocity", 0.1)):
            state = MatchedState(96, 2026, weight, 3.0)
            for cursor in range(2):
                state.step(*batch_for_cursor(cursor))
            checkpoint = save_generation(folder / "upload" / name, state.payload(), signature)
            for cursor in range(2, 6):
                state.step(*batch_for_cursor(cursor))
            expected = save_generation(folder / "expected" / name, state.payload(), signature)
            arms[name] = {"checkpoint": checkpoint, "expected": expected}
        result = {
            "status": "prepared_not_remotely_verified",
            "signature": signature,
            "specification": specification,
            "arms": arms,
            "scientific_fits": 0,
            "new_rmse": None,
        }
        atomic_json(manifest_path, result)
        return result
    manifest = json.loads(manifest_path.read_text())
    if manifest["signature"] != signature:
        raise ValueError("Source or runtime changed before independent replay.")
    transfer = json.loads((folder / "transfer.json").read_text())
    if (
        transfer.get("signature") != signature
        or not transfer.get("all_readback_verified")
        or transfer.get("backend") not in ("s3", "synthetic_local_transport")
    ):
        raise ValueError("Independent S3 transfer receipt is required.")
    checked = {}
    for name in ("coordinate", "velocity"):
        original_receipt = manifest["arms"][name]["checkpoint"]
        for filename in ("checkpoint.json", original_receipt["sha256"] + ".pt"):
            relative = name + "/" + filename
            entry = transfer["files"][relative]
            local = folder / "download" / relative
            if local.stat().st_size != entry["bytes"] or sha256(local) != entry["sha256"]:
                raise ValueError("Downloaded recovery file changed: " + relative)
        downloaded = load_generation(folder / "download" / name, signature)
        if downloaded["steps"] != 2:
            raise ValueError("Independent download is not the interrupted checkpoint.")
        state = MatchedState.restore(downloaded)
        for cursor in range(2, 6):
            state.step(*batch_for_cursor(cursor))
        expected = load_generation(folder / "expected" / name, signature)
        assert_tree_equal(expected, state.payload())
        save_generation(folder / "resumed" / name, state.payload(), signature)
        checked[name] = {"exact_state_match": True, "restored_step": 2, "final_step": 6}
    result = {
        "status": (
            "independent_s3_model_recovery_verified"
            if transfer["backend"] == "s3"
            else "local_transport_recovery_verified"
        ),
        "backend": transfer["backend"],
        "signature": signature,
        "runtime": runtime_identity(),
        "arms": checked,
        "matched_state": ["model", "ema", "optimizer", "rng", "steps", "loss_counters"],
        "independent_download": True,
        "fresh_verification_process": True,
        "scientific_fits": 0,
        "synthetic_optimizer_steps": 20,
        "new_rmse": None,
        "feature_completion_gate": "open",
        "training_ready": False,
        "limitation": "Synthetic recovery proof, not a completed scientific experiment.",
    }
    atomic_json(folder / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "verify"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, stop_timeout)
    signal.alarm(90)
    try:
        result = execute(args.directory, args.stage)
        print(json.dumps(result, sort_keys=True))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
