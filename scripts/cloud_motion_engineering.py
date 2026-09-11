"""Bounded real-data engineering recovery proof; no validation scoring or scientific fit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
from botocore.config import Config

REGION = "us-west-2"
ARCHIVE_SHA256 = "08b4a3fb6b2a85e4f253632e146ea52f67d55164ac823c720629361a386bd998"
ARCHIVE_KEY = "experiments/soft_coverage/checkpoints/" + ARCHIVE_SHA256 + ".tar.gz"
SAMPLE_PATH = "artifacts/temporal/research/inner_1/samples.pkl"
SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
MAX_RUNNER_SECONDS = 900


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def repository_archive(commit: str, output: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Pinned 40-character Git commit required.")
    url = "https://codeload.github.com/alvaromendizabal/nfl-player-trajectory/tar.gz/" + commit
    with urllib.request.urlopen(url, timeout=60) as response, output.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def unpack_repository(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            path = PurePosixPath(*parts)
            if path.is_absolute() or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe repository archive member.")
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("Repository archive member has no content.")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def extract_verified_sample(archive: Path, destination: Path) -> None:
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Minimal archive checksum mismatch.")
    found = False
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe private archive member.")
            if member.name != SAMPLE_PATH:
                continue
            if not member.isfile():
                raise ValueError("Sample archive entry is not an ordinary file.")
            target = destination / SAMPLE_PATH
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("Sample archive entry has no content.")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            found = True
            break
    if not found or digest(destination / SAMPLE_PATH) != SAMPLE_SHA256:
        raise ValueError("Verified sample cache is absent or changed.")


def body_bytes(response: dict[str, Any]) -> bytes:
    body = response["Body"]
    data = body.read() if hasattr(body, "read") else body
    if not isinstance(data, bytes):
        raise ValueError("S3 response body is not bytes.")
    return data


def main() -> None:
    bucket = os.environ["NFL_BUCKET"]
    job = os.environ["NFL_JOB_NAME"]
    commit = os.environ["NFL_REPO_REF"]
    if os.environ.get("NFL_MODE") != "engineering":
        raise ValueError("Only engineering mode is allowed by this runner.")
    started = time.monotonic()
    root = Path("/opt/ml/processing/project")
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"mode": "standard", "max_attempts": 4},
        ),
    )
    status_prefix = "cloud-runs/" + job
    experiment_prefix = "experiments/velocity_isolation/engineering/" + commit

    def event(status: str, **fields: Any) -> None:
        elapsed = time.monotonic() - started
        if elapsed > MAX_RUNNER_SECONDS:
            raise TimeoutError("Engineering recovery proof exceeded 15 minutes.")
        row = {
            "utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "status": status,
            "job": job,
            "code_commit": commit,
            "scientific_fits": 0,
            "validation_scored": False,
            **fields,
        }
        payload = json.dumps(row, sort_keys=True).encode()
        print(payload.decode(), flush=True)
        s3.put_object(
            Bucket=bucket,
            Key=status_prefix + "/status.json",
            Body=payload,
            ServerSideEncryption="AES256",
            ExpectedBucketOwner="560403859723",
        )

    def command(args: list[str], label: str, timeout: int) -> None:
        event("running", stage=label)
        log = root.parent / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("wb") as stream:
            process = subprocess.Popen(
                args,
                cwd=root,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(label + " exceeded its stage budget.") from error
        s3.upload_file(
            str(log),
            bucket,
            status_prefix + "/logs/" + log.name,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
        event("stage_completed", stage=label, log_bytes=log.stat().st_size)

    def pointer(prefix: str, arm: str) -> dict[str, Any]:
        response = s3.get_object(
            Bucket=bucket,
            Key=prefix.rstrip("/") + "/" + arm + "/checkpoints/checkpoint.json",
            ExpectedBucketOwner="560403859723",
        )
        return json.loads(body_bytes(response))

    try:
        event("starting")
        source_archive = root.parent / "source.tar.gz"
        repository_archive(commit, source_archive)
        unpack_repository(source_archive, root)
        private_archive = root.parent / "inputs.tar.gz"
        s3.download_file(bucket, ARCHIVE_KEY, str(private_archive))
        extract_verified_sample(private_archive, root)
        event("inputs_verified", sample_sha256=SAMPLE_SHA256)

        uv_target = root.parent / "uv-tools"
        command(
            [sys.executable, "-m", "pip", "install", "--target", str(uv_target), "uv==0.11.33"],
            "install-uv",
            90,
        )
        uv = str(uv_target / "bin/uv")
        command(
            [uv, "run", "--locked", "scripts/motion_supervision.py", "--prepare-training-plan"],
            "prepare-plan",
            180,
        )

        receipts = {}
        for arm in ("coordinate", "velocity"):
            resume_prefix = experiment_prefix + "/resume"
            clean_prefix = experiment_prefix + "/clean"
            command(
                [
                    uv,
                    "run",
                    "--script",
                    "scripts/motion_supervision_experiment.py",
                    "--arm",
                    arm,
                    "--engineering-steps",
                    "3",
                    "--bucket",
                    bucket,
                    "--prefix",
                    resume_prefix,
                ],
                arm + "-interrupted-3",
                120,
            )
            shutil.rmtree(root / "artifacts/motion_supervision/inner_1" / arm)
            command(
                [
                    uv,
                    "run",
                    "--script",
                    "scripts/motion_supervision_experiment.py",
                    "--arm",
                    arm,
                    "--engineering-steps",
                    "6",
                    "--bucket",
                    bucket,
                    "--prefix",
                    resume_prefix,
                ],
                arm + "-resume-to-6",
                120,
            )
            resumed = pointer(resume_prefix, arm)
            shutil.rmtree(root / "artifacts/motion_supervision/inner_1" / arm)
            command(
                [
                    uv,
                    "run",
                    "--script",
                    "scripts/motion_supervision_experiment.py",
                    "--arm",
                    arm,
                    "--engineering-steps",
                    "6",
                    "--bucket",
                    bucket,
                    "--prefix",
                    clean_prefix,
                ],
                arm + "-clean-6",
                120,
            )
            clean = pointer(clean_prefix, arm)
            if (
                resumed["step"] != clean["step"]
                or resumed["sha256"] != clean["sha256"]
            ):
                raise ValueError(
                    "Fresh-process resumed checkpoint differs from clean execution: " + arm
                )
            receipts[arm] = {
                "step": resumed["step"],
                "sha256": resumed["sha256"],
                "exact_remote_checkpoint_match": True,
            }
            event("arm_recovery_verified", arm=arm, **receipts[arm])
        result = {
            "status": "private_fresh_process_recovery_verified",
            "source_commit": commit,
            "sample_sha256": SAMPLE_SHA256,
            "arms": receipts,
            "scientific_fits": 0,
            "validation_scored": False,
            "new_rmse": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        key = status_prefix + "/engineering_recovery.json"
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ServerSideEncryption="AES256",
            ExpectedBucketOwner="560403859723",
        )
        response = s3.get_object(
            Bucket=bucket,
            Key=key,
            ExpectedBucketOwner="560403859723",
        )
        if body_bytes(response) != payload:
            raise ValueError("Engineering recovery receipt failed S3 read-back verification.")
        event("completed", result_key=key, arms=receipts)
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
