"""Launch no jobs; execute one pinned matched motion study inside a SageMaker Processing job.

The external caller creates the bounded processing job. This script restores only
the verified inner_1 sample, builds the exact CPU runtime, reruns engineering
tests, and invokes the scientific runner. Holdout/Kaggle data are never read.
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
MAX_RUNNER_SECONDS = 1800


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
    if os.environ.get("NFL_MODE") != "scientific":
        raise ValueError("Only the bounded scientific mode is implemented.")
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

    def event(status: str, **fields: Any) -> None:
        elapsed = time.monotonic() - started
        if elapsed > MAX_RUNNER_SECONDS:
            raise TimeoutError("Cloud scientific runner exceeded its 30-minute budget.")
        row = {
            "utc": datetime.now(UTC).isoformat(),
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
            ExpectedBucketOwner="560403859723",
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
                    s3.upload_file(
                        str(log),
                        bucket,
                        prefix + "/logs/" + log.name,
                        ExtraArgs={"ServerSideEncryption": "AES256"},
                    )
                    event("heartbeat", stage=label, log_bytes=log.stat().st_size)
                    heartbeat = time.monotonic() + 20
                time.sleep(0.25)
        s3.upload_file(
            str(log),
            bucket,
            prefix + "/logs/" + log.name,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
        event("stage_completed", stage=label, log_bytes=log.stat().st_size)

    try:
        event("starting", mode="scientific", holdout_access=False)
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
            [sys.executable, "-m", "pip", "install", "--target", str(uv_target), "uv==0.11.33"],
            "install-uv",
            90,
        )
        uv = str(uv_target / "bin/uv")
        command([uv, "sync", "--frozen", "--group", "dev"], "locked-project-environment", 180)
        python = str(root / ".venv/bin/python")
        command(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.8.0",
            ],
            "exact-torch-overlay",
            180,
        )
        command(
            [
                python,
                "-c",
                "import sys,numpy,pandas,torch;"
                "assert sys.version_info[:2]==(3,11);"
                "assert numpy.__version__=='2.4.6';"
                "assert pandas.__version__=='3.0.5';"
                "assert torch.__version__=='2.8.0+cpu';"
                "print(sys.version.split()[0],numpy.__version__,pandas.__version__,torch.__version__)",
            ],
            "runtime-identity",
            30,
        )
        command(
            [python, "scripts/motion_supervision.py", "--prepare-training-plan"],
            "prepare-training-plan",
            120,
        )
        command(
            [python, "scripts/motion_supervision.py", "--self-test"],
            "supervision-engineering-tests",
            220,
        )
        command(
            [python, "scripts/run_motion_experiment.py"],
            "matched-scientific-experiment",
            1500,
        )
        event("completed", scientific_runner="finished", holdout_access=False)
    except Exception as exc:
        event("failed", error_type=type(exc).__name__, error=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
