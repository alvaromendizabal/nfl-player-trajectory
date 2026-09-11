"""Run the frozen inner_1 coordinate-versus-velocity comparison with private S3 recovery."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from nfl_trajectory.motion_supervision import MatchedState
from nfl_trajectory.runtime import atomic_bytes, atomic_json, sha256
from nfl_trajectory.supervision_evidence import load_generation
from nfl_trajectory.supervision_experiment import (
    TrainingSettings,
    evaluate_ema,
    experiment_signature,
    summarize_pair,
    train_arm,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
SOURCE_PATHS = [
    "scripts/run_motion_experiment.py",
    "scripts/motion_supervision.py",
    "src/nfl_trajectory/supervision_experiment.py",
    "src/nfl_trajectory/motion_supervision.py",
    "src/nfl_trajectory/supervision_batches.py",
    "src/nfl_trajectory/supervision_evidence.py",
    "src/nfl_trajectory/supervision_plan.py",
    "src/nfl_trajectory/temporal_model.py",
    "src/nfl_trajectory/temporal_data.py",
    "src/nfl_trajectory/motion_targets.py",
    "src/nfl_trajectory/benchmark.py",
    "configs/motion_supervision.json",
    "uv.lock",
]


class RemoteStore:
    """Content-addressed private S3 checkpoints with independent read-back verification."""

    def __init__(self, bucket: str, prefix: str, client: Any | None = None) -> None:
        if not bucket or not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
            raise ValueError("Safe bucket and relative experiment prefix are required.")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.config = Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "max_attempts": 4},
        )
        self.client = client or boto3.client("s3", region_name="us-west-2", config=self.config)

    def _key(self, arm: str, name: str) -> str:
        if arm not in {"coordinate", "velocity"} or "/" in name or ".." in name:
            raise ValueError("Unsafe checkpoint key component.")
        return f"{self.prefix}/{arm}/{name}"

    @staticmethod
    def _body_bytes(response: dict[str, Any]) -> bytes:
        body = response["Body"]
        data = body.read() if hasattr(body, "read") else body
        if not isinstance(data, bytes):
            raise ValueError("S3 checkpoint body is not bytes.")
        return data

    def _read(self, key: str, client: Any | None = None) -> bytes | None:
        use = client or self.client
        try:
            response = use.get_object(
                Bucket=self.bucket,
                Key=key,
                ExpectedBucketOwner="560403859723",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                return None
            raise
        return self._body_bytes(response)

    def _fresh_client(self) -> Any:
        return boto3.session.Session().client("s3", region_name="us-west-2", config=self.config)

    def put_immutable(self, key: str, payload: bytes) -> dict[str, Any]:
        """Create a content object once, then verify exact bytes with a fresh S3 client."""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=payload,
                ServerSideEncryption="AES256",
                IfNoneMatch="*",
                ExpectedBucketOwner="560403859723",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
                raise
        downloaded = self._read(key, self._fresh_client())
        if downloaded != payload:
            raise ValueError("Independent immutable S3 read-back differs: " + key)
        return {
            "key": key,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "readback_verified": True,
        }

    def publish(self, arm: str, folder: Path, receipt: dict[str, Any]) -> dict[str, Any]:
        """Upload blob before pointer, then independently download and verify exact bytes."""
        pointer = folder / "checkpoint.json"
        blob = folder / (str(receipt["sha256"]) + ".pt")
        pointer_bytes = pointer.read_bytes()
        blob_bytes = blob.read_bytes()
        if hashlib.sha256(blob_bytes).hexdigest() != receipt["sha256"]:
            raise ValueError("Local checkpoint blob differs from its receipt.")
        remote_pointer_key = self._key(arm, "checkpoint.json")
        current_bytes = self._read(remote_pointer_key)
        if current_bytes is not None:
            current = json.loads(current_bytes)
            if current.get("signature") != receipt["signature"]:
                raise ValueError("Remote arm belongs to a different experiment signature.")
            if int(current.get("step", -1)) > int(receipt["step"]):
                raise ValueError("Refusing to replace a newer remote checkpoint.")
        blob_key = self._key(arm, str(receipt["sha256"]) + ".pt")
        self.put_immutable(blob_key, blob_bytes)
        self.client.put_object(
            Bucket=self.bucket,
            Key=remote_pointer_key,
            Body=pointer_bytes,
            ServerSideEncryption="AES256",
            ExpectedBucketOwner="560403859723",
        )
        fresh = self._fresh_client()
        downloaded_blob = self._read(blob_key, fresh)
        downloaded_pointer = self._read(remote_pointer_key, fresh)
        if downloaded_blob != blob_bytes or downloaded_pointer != pointer_bytes:
            raise ValueError("Independent S3 checkpoint read-back differs from local bytes.")
        return {
            "arm": arm,
            "step": int(receipt["step"]),
            "sha256": receipt["sha256"],
            "bytes": len(blob_bytes),
            "blob_key": blob_key,
            "pointer_key": remote_pointer_key,
            "independent_readback_verified": True,
        }

    def restore(self, arm: str, folder: Path, signature: str) -> dict[str, Any] | None:
        """Download the current remote generation to a new local directory and verify it."""
        pointer_key = self._key(arm, "checkpoint.json")
        fresh = self._fresh_client()
        pointer_bytes = self._read(pointer_key, fresh)
        if pointer_bytes is None:
            return None
        receipt = json.loads(pointer_bytes)
        if receipt.get("signature") != signature:
            raise ValueError("Remote checkpoint signature differs from the requested experiment.")
        digest = receipt.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Remote checkpoint digest is invalid.")
        blob_bytes = self._read(self._key(arm, digest + ".pt"), fresh)
        if blob_bytes is None or hashlib.sha256(blob_bytes).hexdigest() != digest:
            raise ValueError("Remote checkpoint blob is absent or corrupt.")
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Independent restore destination must be empty.")
        folder.mkdir(parents=True, exist_ok=True)
        atomic_bytes(folder / "checkpoint.json", pointer_bytes)
        atomic_bytes(folder / (digest + ".pt"), blob_bytes)
        payload = load_generation(folder, signature)
        return {
            "receipt": receipt,
            "state": payload,
            "independent_readback_verified": True,
        }


def source_signature() -> tuple[str, dict[str, str]]:
    hashes = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return digest, hashes


def write_errors(store: RemoteStore, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    if name not in {"coordinate", "velocity"}:
        raise ValueError("Unknown validation-error arm.")
    payload = frame.to_csv(index=False).encode()
    digest = hashlib.sha256(payload).hexdigest()
    key = f"{store.prefix}/errors/{name}-{digest}.csv"
    return store.put_immutable(key, payload)


def main() -> None:
    started = time.monotonic()
    bucket = os.environ["NFL_BUCKET"]
    job = os.environ["NFL_JOB_NAME"]
    config_path = ROOT / "configs/motion_supervision.json"
    config = json.loads(config_path.read_text())
    if not config.get("scientific_execution_enabled"):
        raise ValueError("Scientific execution is disabled in the committed experiment config.")
    cache = ROOT / config["input"]["sample_path"]
    if sha256(cache) != SAMPLE_SHA256:
        raise ValueError("Private inner_1 sample checksum differs from the frozen experiment.")
    with cache.open("rb") as stream:
        samples = pickle.load(stream)
    training = [sample for sample in samples if str(sample["split"]) == "train"]
    validation = [sample for sample in samples if str(sample["split"]) == "validation"]
    if (len(training), len(validation)) != (4951, 2103):
        raise ValueError("Private play counts differ from the frozen experiment.")
    if sum(len(sample["keys"]) for sample in validation) != 83938:
        raise ValueError("Private validation row count differs from the frozen experiment.")
    plan = json.loads(
        (ROOT / "artifacts/motion_supervision/preflight/training_plan.json").read_text()
    )["plan"]
    settings = TrainingSettings(
        epochs=int(config["scientific_epochs"]),
        batch_plays=int(config["scientific_batch_plays"]),
        warmup_steps=int(config["scientific_lr_schedule"]["warmup_steps"]),
        seed=int(config["matching"]["seed"]),
        width=int(config["model"]["width"]),
        peak_learning_rate=float(config["scientific_lr_schedule"]["peak"]),
        floor_learning_rate=float(config["scientific_lr_schedule"]["floor"]),
    )
    if settings.total_steps(len(training)) != int(config["scientific_total_steps"]):
        raise ValueError("Configured exposure and training population disagree.")
    source_digest, source_hashes = source_signature()
    config_digest = sha256(config_path)
    study_digest = hashlib.sha256(
        json.dumps(
            {
                "source": source_digest,
                "sample": SAMPLE_SHA256,
                "config": config_digest,
                "experiment_id": config["experiment_id"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    prefix = f"experiments/velocity_isolation/inner_1/{study_digest}"
    store = RemoteStore(bucket, prefix)
    status_key = f"cloud-runs/{job}/scientific-status.json"
    last_heartbeat = 0.0

    def event(status: str, **fields: Any) -> None:
        row = {
            "utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": status,
            "job": job,
            "source_commit": os.environ.get("NFL_REPO_REF"),
            "study_signature": study_digest,
            **fields,
        }
        payload = json.dumps(row, sort_keys=True).encode()
        print(payload.decode(), flush=True)
        store.client.put_object(
            Bucket=bucket,
            Key=status_key,
            Body=payload,
            ServerSideEncryption="AES256",
            ExpectedBucketOwner="560403859723",
        )

    def publisher_for(
        arm_name: str,
        published_rows: list[dict[str, Any]],
    ) -> Any:
        def publisher(folder: Path, receipt: dict[str, Any]) -> None:
            remote = store.publish(arm_name, folder, receipt)
            published_rows.append(remote)
            event("epoch_checkpoint_verified", **remote)

        return publisher

    def progress_for(arm_name: str) -> Any:
        def progress(row: dict[str, float | int]) -> None:
            nonlocal last_heartbeat
            now = time.monotonic()
            if now - last_heartbeat >= float(config["budgets"]["heartbeat_seconds"]):
                event(
                    "training_heartbeat",
                    arm=arm_name,
                    step=int(row["step"]),
                    epoch=int(row["epoch"]),
                    loss=float(row["loss"]),
                )
                last_heartbeat = now

        return progress

    results: dict[str, Any] = {}
    errors: dict[str, pd.DataFrame] = {}
    for arm in ("coordinate", "velocity"):
        signature = experiment_signature(source_digest, SAMPLE_SHA256, config_digest, arm)
        work = ROOT / "artifacts/motion_supervision/scientific" / study_digest / arm
        remote_seed = ROOT / "artifacts/motion_supervision/remote_seed" / study_digest / arm
        restored = store.restore(arm, remote_seed, signature)
        if restored is not None:
            receipt = restored["receipt"]
            work.mkdir(parents=True, exist_ok=True)
            pointer = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
            atomic_bytes(work / "checkpoint.json", pointer)
            digest = str(receipt["sha256"])
            remote_blob = (remote_seed / (digest + ".pt")).read_bytes()
            atomic_bytes(work / (digest + ".pt"), remote_blob)
        published: list[dict[str, Any]] = []
        before_steps = int(restored["state"]["steps"]) if restored is not None else 0
        event("arm_started", arm=arm, restored_step=before_steps)
        state, curve = train_arm(
            samples,
            arm,
            float(plan["training_only_velocity_rms"]),
            float(plan["coordinate_loss_denominator"]),
            float(plan["velocity_loss_denominator"]),
            settings,
            work,
            signature,
            publisher_for(arm, published),
            progress=progress_for(arm),
            max_seconds=float(config["budgets"]["per_arm_wall_seconds"]),
        )
        if state.steps != int(config["scientific_total_steps"]):
            raise ValueError("Arm did not finish the frozen optimizer exposure.")
        final_restore = ROOT / "artifacts/motion_supervision/final_readback" / study_digest / arm
        final = store.restore(arm, final_restore, signature)
        if final is None or int(final["state"]["steps"]) != state.steps:
            raise ValueError("Final remote checkpoint could not be independently restored.")
        restored_state = MatchedState.restore(final["state"])
        frame = evaluate_ema(restored_state, samples, settings.batch_plays)
        errors[arm] = frame
        error_receipt = write_errors(store, arm, frame)
        results[arm] = {
            "final_step": restored_state.steps,
            "initial_remote_step": before_steps,
            "new_optimizer_steps": restored_state.steps - before_steps,
            "epochs": settings.epochs,
            "checkpoint_publications_this_job": len(published),
            "final_checkpoint": final["receipt"],
            "final_remote_readback_verified": True,
            "private_errors": error_receipt,
            "curve_tail": curve[-20:],
        }
        event(
            "arm_completed",
            arm=arm,
            final_step=restored_state.steps,
            new_optimizer_steps=restored_state.steps - before_steps,
        )
    summary = summarize_pair(
        errors["coordinate"],
        errors["velocity"],
        float(config["continuation"]["minimum_relative_coordinate_rmse_gain"]),
        int(config["evaluation"]["bootstrap_resamples"]),
        int(config["evaluation"]["bootstrap_seed"]),
    )
    report = {
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "study_signature": study_digest,
        "source_commit": os.environ.get("NFL_REPO_REF"),
        "source_signature": source_digest,
        "source_hashes": source_hashes,
        "config_sha256": config_digest,
        "sample_sha256": SAMPLE_SHA256,
        "settings": {
            "epochs": settings.epochs,
            "batch_plays": settings.batch_plays,
            "total_steps": settings.total_steps(len(training)),
            "warmup_steps": settings.warmup_steps,
            "seed": settings.seed,
            "width": settings.width,
        },
        "arms": results,
        "evaluation": summary,
        "feature_completion_gate": "open",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "private_data_published_to_github": False,
        "scientific_fits_new": sum(
            int(result["new_optimizer_steps"] > 0) for result in results.values()
        ),
    }
    destination = ROOT / "artifacts/motion_supervision/scientific" / study_digest / "summary.json"
    atomic_json(destination, report)
    payload = destination.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    summary_key = f"{prefix}/summaries/{digest}.json"
    store.put_immutable(summary_key, payload)
    event(
        "completed",
        coordinate_rmse=summary["control"]["coordinate_rmse_yards"],
        velocity_rmse=summary["velocity"]["coordinate_rmse_yards"],
        relative_rmse_gain=summary["relative_rmse_gain"],
        continuation_gate_passed=summary["continuation_gate_passed"],
        summary_key=summary_key,
    )


if __name__ == "__main__":
    main()
