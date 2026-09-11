# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "boto3==1.43.89", "filelock==3.32.5", "numpy==2.4.6", "pandas==3.0.5",
#   "torch==2.8.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Private real-data recovery check using the canonical matched training core.

This script never scores validation. It advances one named arm only to a tiny
engineering cursor, publishing every durable generation through the canonical
RemoteStore. It is intentionally separate from the scientific runner so a
fresh-process resume can be compared with a clean execution before scientific
execution is enabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_motion_experiment import RemoteStore, source_signature  # noqa: E402
from nfl_trajectory.motion_supervision import MatchedState  # noqa: E402
from nfl_trajectory.runtime import atomic_bytes, atomic_json, sha256  # noqa: E402
from nfl_trajectory.supervision_experiment import (  # noqa: E402
    TrainingSettings,
    experiment_signature,
    train_arm,
)

SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
SAMPLE_PATH = "artifacts/temporal/research/inner_1/samples.pkl"
PLAN_PATH = "artifacts/motion_supervision/preflight/training_plan.json"
CONFIG_PATH = "configs/motion_supervision.json"


def settings(config: dict[str, Any]) -> TrainingSettings:
    return TrainingSettings(
        epochs=int(config["scientific_epochs"]),
        batch_plays=int(config["scientific_batch_plays"]),
        warmup_steps=int(config["scientific_lr_schedule"]["warmup_steps"]),
        seed=int(config["matching"]["seed"]),
        width=int(config["model"]["width"]),
        peak_learning_rate=float(config["scientific_lr_schedule"]["peak"]),
        floor_learning_rate=float(config["scientific_lr_schedule"]["floor"]),
    )


def recovery_source_signature() -> str:
    base, _ = source_signature()
    payload = {
        "canonical_runner": base,
        "recovery_check": sha256(Path(__file__)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if not re.fullmatch(r"[a-z0-9_-]+", args.variant):
        raise ValueError("Engineering variant must be a safe lower-case token.")
    if args.steps < 1 or args.steps > 6:
        raise ValueError("Engineering recovery is intentionally capped at six optimizer steps.")
    cache = ROOT / SAMPLE_PATH
    if not cache.is_file() or cache.is_symlink() or sha256(cache) != SAMPLE_SHA256:
        raise ValueError("Verified private inner_1 sample is missing or changed.")
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    plan = json.loads((ROOT / PLAN_PATH).read_text())
    if plan.get("sample_sha256") != SAMPLE_SHA256:
        raise ValueError("Training-only plan does not match the verified sample.")
    with cache.open("rb") as stream:
        samples = pickle.load(stream)
    training = [sample for sample in samples if str(sample["split"]) == "train"]
    validation = [sample for sample in samples if str(sample["split"]) == "validation"]
    if (len(training), len(validation)) != (4951, 2103):
        raise ValueError("Private play counts differ from the reviewed contract.")
    config_sha = sha256(ROOT / CONFIG_PATH)
    source_sha = recovery_source_signature()
    signature = experiment_signature(source_sha, SAMPLE_SHA256, config_sha, args.arm)
    prefix = (
        "experiments/velocity_isolation/engineering/"
        + source_sha
        + "/"
        + args.variant
    )
    store = RemoteStore(args.bucket, prefix)
    local_tag = hashlib.sha256(prefix.encode()).hexdigest()[:16]
    folder = ROOT / "artifacts/motion_supervision/engineering" / local_tag / args.arm
    restore_folder = ROOT / "artifacts/motion_supervision/engineering_readback" / local_tag / args.arm
    restored = store.restore(args.arm, restore_folder, signature)
    if restored is not None:
        receipt = restored["receipt"]
        folder.mkdir(parents=True, exist_ok=True)
        atomic_bytes(
            folder / "checkpoint.json",
            (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
        )
        digest = str(receipt["sha256"])
        atomic_bytes(
            folder / (digest + ".pt"),
            (restore_folder / (digest + ".pt")).read_bytes(),
        )
    before = int(restored["state"]["steps"]) if restored is not None else 0
    publications: list[dict[str, Any]] = []

    def publisher(path: Path, receipt: dict[str, Any]) -> None:
        remote = store.publish(args.arm, path, receipt)
        publications.append(remote)
        print(
            json.dumps(
                {
                    "event": "engineering_checkpoint_verified",
                    "arm": args.arm,
                    "variant": args.variant,
                    "step": receipt["step"],
                    "sha256": receipt["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    state, curve = train_arm(
        samples,
        args.arm,
        float(plan["plan"]["training_only_velocity_rms"]),
        float(plan["plan"]["coordinate_loss_denominator"]),
        float(plan["plan"]["velocity_loss_denominator"]),
        settings(config),
        folder,
        signature,
        publisher,
        stop_after_steps=args.steps,
        max_seconds=120,
    )
    final_folder = ROOT / "artifacts/motion_supervision/engineering_final" / local_tag / args.arm
    final = store.restore(args.arm, final_folder, signature)
    if final is None or int(final["state"]["steps"]) != state.steps:
        raise ValueError("Engineering checkpoint could not be independently restored.")
    remote_state = MatchedState.restore(final["state"])
    if remote_state.steps != state.steps:
        raise ValueError("Remote restored cursor differs from local cursor.")
    result = {
        "status": "private_engineering_checkpoint_verified",
        "arm": args.arm,
        "variant": args.variant,
        "steps": state.steps,
        "initial_remote_step": before,
        "new_optimizer_steps": state.steps - before,
        "signature": signature,
        "checkpoint_sha256": final["receipt"]["sha256"],
        "checkpoint_bytes": final["receipt"]["bytes"],
        "remote_readback_verified": True,
        "checkpoint_publications_this_process": len(publications),
        "curve_tail": curve[-6:],
        "scientific_fits": 0,
        "validation_scored": False,
        "new_rmse": None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    output = folder / "engineering_result.json"
    atomic_json(output, result)
    receipts = ROOT / "artifacts/motion_supervision/engineering_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    atomic_json(receipts / f"{args.variant}-{args.arm}.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("coordinate", "velocity"))
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
