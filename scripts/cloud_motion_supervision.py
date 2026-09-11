"""Run the training-only motion-supervision profile in a bounded SageMaker job.

This runner restores only the verified inner_1 sample cache from the existing
minimal private archive, installs the script-locked Python 3.11 runtime, prepares
the training-only plan, and measures finite-gradient throughput. It never reads
validation labels for statistics, evaluates RMSE, or launches another AWS job.
"""

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
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config

REGION = "us-west-2"
ARCHIVE_SHA256 = "08b4a3fb6b2a85e4f253632e146ea52f67d55164ac823c720629361a386bd998"
ARCHIVE_KEY = "experiments/soft_coverage/checkpoints/" + ARCHIVE_SHA256 + ".tar.gz"
SAMPLE_PATH = "artifacts/temporal/research/inner_1/samples.pkl"
SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
MAX_RUNNER_SECONDS = 720


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
    if not found:
        raise ValueError("Verified minimal archive does not contain the sample cache.")
    if digest(destination / SAMPLE_PATH) != SAMPLE_SHA256:
        raise ValueError("Restored sample checksum mismatch.")


def main() -> None:
    bucket = os.environ["NFL_BUCKET"]
    job = os.environ["NFL_JOB_NAME"]
    commit = os.environ["NFL_REPO_REF"]
    mode = os.environ.get("NFL_MODE", "profile")
    if mode != "profile":
        raise ValueError("Only the bounded profile mode is implemented.")
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
    prefix = "cloud-runs/" + job

    def event(status: str, **fields: object) -> None:
        elapsed = time.monotonic() - started
        if elapsed > MAX_RUNNER_SECONDS:
            raise TimeoutError("Cloud motion profile exceeded its 12-minute runner budget.")
        row = {
            "utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - base image may be <3.11
            "elapsed_seconds": round(elapsed, 3),
            "status": status,
            "job": job,
            "code_commit": commit,
            **fields,
        }
        payload = json.dumps(row, sort_keys=True).encode()
        print(payload.decode(), flush=True)
        s3.put_object(
            Bucket=bucket,
            Key=prefix + "/status.json",
            Body=payload,
            ServerSideEncryption="AES256",
        )

    def upload_log(log: Path) -> None:
        s3.upload_file(
            str(log),
            bucket,
            prefix + "/logs/" + log.name,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )

    def command(args: list[str], label: str, timeout: int) -> None:
        event("running", stage=label, timeout_seconds=timeout)
        log = root.parent / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
        )
        with log.open("wb") as stream:
            process = subprocess.Popen(
                args,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + timeout
            heartbeat = time.monotonic() + 20
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError(label + " exceeded its stage budget.")
                if time.monotonic() >= heartbeat:
                    upload_log(log)
                    event("heartbeat", stage=label, log_bytes=log.stat().st_size)
                    heartbeat = time.monotonic() + 20
                time.sleep(0.25)
        upload_log(log)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
        event("stage_completed", stage=label, log_bytes=log.stat().st_size)

    try:
        event("starting", mode=mode, scientific_fits=0)
        source_archive = root.parent / "source.tar.gz"
        repository_archive(commit, source_archive)
        unpack_repository(source_archive, root)
        event("source_restored", source_bytes=source_archive.stat().st_size)

        private_archive = root.parent / "inputs.tar.gz"
        s3.download_file(bucket, ARCHIVE_KEY, str(private_archive))
        extract_verified_sample(private_archive, root)
        event(
            "input_verified",
            archive_sha256=ARCHIVE_SHA256,
            sample_sha256=SAMPLE_SHA256,
            sample_bytes=(root / SAMPLE_PATH).stat().st_size,
        )

        uv_target = root.parent / "uv-tools"
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(uv_target),
                "uv==0.12.5",
            ],
            "install-uv",
            90,
        )
        uv = str(uv_target / "bin/uv")
        command(
            [
                uv,
                "run",
                "--script",
                "scripts/motion_supervision.py",
                "--prepare-training-plan",
            ],
            "prepare-plan",
            180,
        )
        command(
            [uv, "run", "--script", "scripts/motion_supervision.py", "--profile-training"],
            "profile-training",
            120,
        )
        profile = root / "artifacts/motion_supervision/profile/result.json"
        plan = root / "artifacts/motion_supervision/preflight/training_plan.json"
        if not profile.is_file() or not plan.is_file():
            raise ValueError("Profile did not produce the required receipts.")
        result = json.loads(profile.read_text())
        if (
            result.get("status") != "training_only_throughput_measured"
            or result.get("scientific_fits") != 0
        ):
            raise ValueError("Unexpected profile result.")
        for path, key in ((profile, "profile.json"), (plan, "training_plan.json")):
            s3.upload_file(
                str(path),
                bucket,
                prefix + "/artifacts/" + key,
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
            remote = s3.get_object(Bucket=bucket, Key=prefix + "/artifacts/" + key)
            with remote["Body"] as body:
                payload = body.read()
            if payload != path.read_bytes():
                raise ValueError("S3 read-back differs for " + key)
        event(
            "completed",
            profile_signature=result["signature"],
            conservative_epoch_seconds=result["conservative_epoch_seconds"],
            worst_stress_batch_seconds=result["worst_stress_batch_seconds"],
            peak_rss_mib=result["peak_rss_mib"],
            scientific_fits=0,
            s3_readback_verified=True,
        )
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc)[:500], scientific_fits=0)
        raise


if __name__ == "__main__":
    main()
