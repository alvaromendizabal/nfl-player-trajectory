# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "boto3==1.43.89", "numpy==2.4.6", "pandas==3.0.5", "torch==2.8.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run one bounded matched supervision arm with independently verified S3 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.runtime import atomic_bytes, atomic_json, sha256  # noqa: E402
from nfl_trajectory.supervision_evidence import load_generation  # noqa: E402
from nfl_trajectory.supervision_experiment import (  # noqa: E402
    TrainingSettings,
    diagnostic_slices,
    evaluate_ema,
    experiment_signature,
    summarize_pair,
    train_arm,
)

SAMPLE_PATH = "artifacts/temporal/research/inner_1/samples.pkl"
SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
PLAN_PATH = "artifacts/motion_supervision/preflight/training_plan.json"
CONFIG_PATH = "configs/motion_supervision.json"
SOURCE_PATHS = [
    "scripts/motion_supervision_experiment.py",
    "src/nfl_trajectory/supervision_experiment.py",
    "src/nfl_trajectory/motion_supervision.py",
    "src/nfl_trajectory/supervision_batches.py",
    "src/nfl_trajectory/supervision_evidence.py",
    "src/nfl_trajectory/supervision_plan.py",
    "src/nfl_trajectory/motion_targets.py",
    "src/nfl_trajectory/temporal_data.py",
    "src/nfl_trajectory/temporal_model.py",
]


def _body_bytes(body: Any) -> bytes:
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    if isinstance(body, str):
        return body.encode()
    if hasattr(body, "read"):
        return body.read()
    raise TypeError("Unsupported S3 response body.")


def upload_verified(client: Any, bucket: str, key: str, path: Path) -> dict[str, Any]:
    """Upload exact bytes then independently GET and compare before returning."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ServerSideEncryption="AES256",
        ExpectedBucketOwner="560403859723",
    )
    response = client.get_object(
        Bucket=bucket,
        Key=key,
        ExpectedBucketOwner="560403859723",
    )
    remote = _body_bytes(response["Body"])
    if remote != data or hashlib.sha256(remote).hexdigest() != digest:
        raise ValueError("Independent S3 read-back differs: " + key)
    return {
        "key": key,
        "bytes": len(data),
        "sha256": digest,
        "version_id": response.get("VersionId"),
    }


def publish_checkpoint(
    client: Any,
    bucket: str,
    prefix: str,
    arm: str,
    folder: Path,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Publish immutable tensor first and checkpoint pointer last; fail closed."""
    base = prefix.rstrip("/") + "/" + arm + "/checkpoints/"
    blob = folder / (receipt["sha256"] + ".pt")
    pointer = folder / "checkpoint.json"
    blob_receipt = upload_verified(client, bucket, base + blob.name, blob)
    pointer_receipt = upload_verified(client, bucket, base + pointer.name, pointer)
    if json.loads(pointer.read_text()) != receipt:
        raise ValueError("Local checkpoint pointer changed during S3 publication.")
    return {"blob": blob_receipt, "pointer": pointer_receipt, "step": receipt["step"]}


def restore_remote_checkpoint(
    client: Any,
    bucket: str,
    prefix: str,
    arm: str,
    folder: Path,
    signature: str,
) -> bool:
    """Restore the remote pointer/blob into an empty local generation directory."""
    base = prefix.rstrip("/") + "/" + arm + "/checkpoints/"
    try:
        pointer_response = client.get_object(
            Bucket=bucket,
            Key=base + "checkpoint.json",
            ExpectedBucketOwner="560403859723",
        )
    except client.exceptions.NoSuchKey:
        return False
    pointer = _body_bytes(pointer_response["Body"])
    receipt = json.loads(pointer)
    if receipt.get("signature") != signature:
        raise ValueError("Remote checkpoint belongs to a different experiment signature.")
    digest = receipt.get("sha256", "")
    blob_response = client.get_object(
        Bucket=bucket,
        Key=base + digest + ".pt",
        ExpectedBucketOwner="560403859723",
    )
    blob = _body_bytes(blob_response["Body"])
    if len(blob) != receipt.get("bytes") or hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError("Remote checkpoint blob failed size/hash verification.")
    folder.mkdir(parents=True, exist_ok=True)
    atomic_bytes(folder / (digest + ".pt"), blob)
    atomic_bytes(folder / "checkpoint.json", pointer)
    load_generation(folder, signature)
    return True


def source_signature() -> str:
    values = {name: sha256(ROOT / name) for name in SOURCE_PATHS}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    sample = ROOT / SAMPLE_PATH
    if not sample.is_file() or sample.is_symlink() or sha256(sample) != SAMPLE_SHA256:
        raise ValueError("Verified inner_1 sample cache is missing or changed.")
    plan = json.loads((ROOT / PLAN_PATH).read_text())
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    if plan.get("sample_sha256") != SAMPLE_SHA256:
        raise ValueError("Training plan does not match the verified sample cache.")
    with sample.open("rb") as stream:
        samples = pickle.load(stream)
    if len(samples) != 7054:
        raise ValueError("Private sample play count changed.")
    return samples, plan, config


def settings_from_config(config: dict[str, Any]) -> TrainingSettings:
    return TrainingSettings(
        epochs=int(config["scientific_epochs"]),
        batch_plays=int(config["scientific_batch_plays"]),
        warmup_steps=int(config["scientific_lr_schedule"]["warmup_steps"]),
        seed=int(config["matching"]["seed"]),
        width=int(config["model"]["width"]),
        peak_learning_rate=float(config["scientific_lr_schedule"]["peak"]),
        floor_learning_rate=float(config["scientific_lr_schedule"]["floor"]),
    )


def client_for_region(region: str) -> Any:
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "max_attempts": 4},
        ),
    )


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    samples, plan, config = load_inputs()
    if args.engineering_steps is None and not bool(config.get("scientific_execution_enabled")):
        raise ValueError("Scientific execution is still disabled by the frozen config.")
    config_sha = sha256(ROOT / CONFIG_PATH)
    signature = experiment_signature(source_signature(), SAMPLE_SHA256, config_sha, args.arm)
    folder = ROOT / "artifacts/motion_supervision/inner_1" / args.arm
    client = client_for_region(args.region)
    restored = restore_remote_checkpoint(
        client, args.bucket, args.prefix, args.arm, folder, signature
    )
    publications: list[dict[str, Any]] = []

    def publisher(checkpoint_folder: Path, receipt: dict[str, Any]) -> None:
        remote = publish_checkpoint(
            client,
            args.bucket,
            args.prefix,
            args.arm,
            checkpoint_folder,
            receipt,
        )
        publications.append(remote)
        print(
            json.dumps(
                {
                    "event": "checkpoint_readback_verified",
                    "arm": args.arm,
                    "step": receipt["step"],
                    "sha256": receipt["sha256"],
                }
            ),
            flush=True,
        )

    def progress(row: dict[str, float | int]) -> None:
        step = int(row["step"])
        if step == 1 or step % 20 == 0:
            print(
                json.dumps(
                    {
                        "event": "training_heartbeat",
                        "arm": args.arm,
                        "step": step,
                        "epoch": row["epoch"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )

    state, curve = train_arm(
        samples,
        args.arm,
        float(plan["plan"]["training_only_velocity_rms"]),
        float(plan["plan"]["coordinate_loss_denominator"]),
        float(plan["plan"]["velocity_loss_denominator"]),
        settings_from_config(config),
        folder,
        signature,
        publisher,
        stop_after_steps=args.engineering_steps,
        progress=progress,
        max_seconds=float(config["budgets"]["per_arm_training_seconds"]),
    )
    total_steps = int(config["scientific_total_steps"])
    complete = state.steps == total_steps
    result: dict[str, Any] = {
        "status": "arm_completed" if complete else "engineering_checkpoint_completed",
        "arm": args.arm,
        "signature": signature,
        "steps": state.steps,
        "total_steps": total_steps,
        "remote_checkpoint_restored": restored,
        "checkpoint_publications": publications,
        "elapsed_seconds": time.monotonic() - started,
        "scientific_fit": bool(complete),
        "validation_scored": False,
        "new_rmse": None,
    }
    if complete:
        errors = evaluate_ema(state, samples, int(config["scientific_batch_plays"]))
        errors_path = folder / "errors.csv"
        atomic_bytes(errors_path, errors.to_csv(index=False).encode())
        result.update(
            validation_scored=True,
            metrics=__import__("nfl_trajectory.benchmark", fromlist=["error_metrics"]).error_metrics(
                errors
            ),
            diagnostic_slices=diagnostic_slices(errors),
            evaluation_rows=len(errors),
            evaluation_games=int(errors.game_id.nunique()),
        )
        result["errors_remote"] = upload_verified(
            client,
            args.bucket,
            args.prefix.rstrip("/") + "/" + args.arm + "/errors.csv",
            errors_path,
        )
        result["new_rmse"] = result["metrics"]["coordinate_rmse_yards"]
    atomic_json(folder / "summary.json", result)
    result["summary_remote"] = upload_verified(
        client,
        args.bucket,
        args.prefix.rstrip("/") + "/" + args.arm + "/summary.json",
        folder / "summary.json",
    )
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    client = client_for_region(args.region)
    frames = {}
    for arm in ("coordinate", "velocity"):
        key = args.prefix.rstrip("/") + "/" + arm + "/errors.csv"
        response = client.get_object(
            Bucket=args.bucket,
            Key=key,
            ExpectedBucketOwner="560403859723",
        )
        frames[arm] = pd.read_csv(io.BytesIO(_body_bytes(response["Body"])))
    config = json.loads((ROOT / CONFIG_PATH).read_text())
    result = summarize_pair(
        frames["coordinate"],
        frames["velocity"],
        float(config["continuation"]["minimum_relative_coordinate_rmse_gain"]),
        int(config["evaluation"]["bootstrap_resamples"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    result["feature_completion_gate"] = "open"
    destination = ROOT / "artifacts/motion_supervision/inner_1/comparison.json"
    atomic_json(destination, result)
    result["remote"] = upload_verified(
        client,
        args.bucket,
        args.prefix.rstrip("/") + "/comparison.json",
        destination,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("coordinate", "velocity"))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--engineering-steps", type=int)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()
    if args.compare == (args.arm is not None):
        raise SystemExit("Choose exactly one arm or --compare.")
    if args.engineering_steps is not None and args.engineering_steps < 1:
        raise SystemExit("--engineering-steps must be positive.")
    result = compare(args) if args.compare else run_arm(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
