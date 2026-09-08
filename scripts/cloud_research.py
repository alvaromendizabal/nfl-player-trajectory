"""Run bounded feature research on SageMaker with verified, durable checkpoints.

The caller supplies a pinned Git commit and private S3 snapshot. The runner never
downloads holdout tracking, scores the holdout, creates an owner submission, or
starts a persistent endpoint. SageMaker enforces the external runtime limit.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
from botocore.config import Config

REGION = "us-west-2"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def selected_input(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe source snapshot path.")
    if re.search(r"2023_w(?:16|17|18)(?:[./_]|$)", name):
        return False
    if name.startswith("data/raw/kaggle_evaluation/") or name in {
        "data/raw/test.csv", "data/raw/test_input.csv",
    }:
        return True
    if path.parts[0] in {"artifacts", ".state", "logs"}:
        return not name.startswith("artifacts/kaggle/")
    return bool(re.fullmatch(r"data/raw/train/input_2023_w(?:0[1-9]|1[0-5])\.csv", name))


def unpack(archive: Path, destination: Path, repository: bool = False) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = PurePosixPath(member.name).parts
            if repository:
                parts = parts[1:]
            if not parts:
                continue
            path = PurePosixPath(*parts)
            if path.is_absolute() or ".." in parts or not (member.isfile() or member.isdir()):
                raise ValueError("Archive paths must be ordinary contained files.")
            if not repository and path.parts[0] not in {"artifacts", ".state", "logs"}:
                continue
            output = destination.joinpath(*parts)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no content.")
            with source, output.open("wb") as target:
                shutil.copyfileobj(source, target)


def repository_archive(commit: str, output: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A pinned 40-character Git commit is required.")
    url = "https://codeload.github.com/alvaromendizabal/nfl-player-trajectory/tar.gz/" + commit
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, output.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main() -> None:
    bucket, job = os.environ["NFL_BUCKET"], os.environ["NFL_JOB_NAME"]
    commit, snapshot_key = os.environ["NFL_REPO_REF"], os.environ["NFL_SNAPSHOT"]
    root = Path("/opt/ml/processing/project")
    root.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        region_name=REGION,
        config=Config(retries={"mode": "standard", "max_attempts": 5}, max_pool_connections=16),
    )
    prefix = "cloud-runs/" + job
    started = time.monotonic()

    def event(status: str, **fields: Any) -> None:
        value = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - base container may be pre-3.11
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": status,
            "job": job,
            "code_commit": commit,
            **fields,
        }
        print(json.dumps(value), flush=True)
        s3.put_object(
            Bucket=bucket,
            Key=prefix + "/status.json",
            Body=json.dumps(value).encode(),
            ServerSideEncryption="AES256",
        )

    def command(args: list[str], label: str, threads: int = 2) -> None:
        event("running", stage=label)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"}
        output = root / "logs" / ("cloud-" + label + ".log")
        output.parent.mkdir(exist_ok=True)
        with output.open("w") as handle:
            process = subprocess.Popen(
                args,
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            while True:
                try:
                    return_code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    s3.upload_file(
                        str(output),
                        bucket,
                        prefix + "/logs/" + output.name,
                        ExtraArgs={"ServerSideEncryption": "AES256"},
                    )
            s3.upload_file(
                str(output),
                bucket,
                prefix + "/logs/" + output.name,
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
        if return_code:
            raise subprocess.CalledProcessError(return_code, args)
        event("stage_completed", stage=label)

    python = str(root / ".venv/bin/python")
    nfl = str(root / ".venv/bin/nfl")

    def backup(label: str) -> None:
        command([nfl, "backup", "--bucket", bucket], "backup-" + label)
        receipt = json.loads((root / "artifacts/last_backup.json").read_text())
        s3.put_object(
            Bucket=bucket,
            Key=prefix + "/checkpoint.json",
            Body=json.dumps({**receipt, "code_commit": commit, "stage": label}).encode(),
            ServerSideEncryption="AES256",
        )

    def batch(script: str, folds: list[str], workers: int, threads: int) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    command,
                    [uv, "run", "--locked", script, "--fold", fold],
                    Path(script).stem + "-" + fold,
                    threads,
                )
                for fold in folds
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    try:
        event("restoring")
        repository_tar = root.parent / "source.tar.gz"
        repository_archive(commit, repository_tar)
        unpack(repository_tar, root, repository=True)
        response = s3.get_object(Bucket=bucket, Key=snapshot_key)
        with response["Body"] as body:
            payload = body.read()
        if hashlib.sha256(payload).hexdigest() != Path(snapshot_key).stem:
            raise ValueError("Original snapshot manifest checksum failed.")
        manifest = json.loads(payload)
        selected = [item for item in manifest["files"] if selected_input(item["path"])]

        def restore(item: dict[str, Any]) -> None:
            path = root / item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            if (
                path.exists()
                and path.stat().st_size == item["size"]
                and digest(path) == item["sha256"]
            ):
                return
            s3.download_file(bucket, "objects/" + item["sha256"], str(path))
            if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
                raise ValueError("Restored input checksum or size failed.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(restore, selected))
        seed = root.parent / "seed.tar.gz"
        s3.download_file(bucket, os.environ["NFL_SEED_KEY"], str(seed))
        if digest(seed) != os.environ["NFL_SEED_SHA"]:
            raise ValueError("Research seed checkpoint checksum failed.")
        unpack(seed, root)
        event("restored", files=len(selected), holdout_tracking="excluded")
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(root.parent / "uv-tools"),
                "uv==0.11.33",
            ],
            "install-uv",
        )
        uv = str(root.parent / "uv-tools/bin/uv")
        command([uv, "sync", "--frozen", "--group", "dev"], "locked-environment")
        command([python, "-m", "pytest", "-q"], "tests", threads=2)
        folds = ["inner_1", "inner_2", "inner_3", "development"]
        for stage_name in (
            "feature-research",
            "context-research",
            "representation-research",
            "research-report",
        ):
            command([nfl, stage_name], stage_name, threads=4)
        backup("feature-banks")
        command([uv, "run", "--locked", "scripts/nonlinear_probe.py"], "nonlinear-probe", threads=6)
        backup("fixed-nonlinear")
        batch("scripts/ablate_features.py", folds, workers=4, threads=2)
        batch("scripts/joint_feature_fit.py", folds, workers=4, threads=1)
        backup("ablations-and-joint-fits")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            wide = pool.submit(batch, "scripts/feature_budget.py", folds, 2, 6)
            inference = pool.submit(
                command, [python, "scripts/validate_research.py"], "raw-inference", 1
            )
            wide.result()
            inference.result()
        backup("completed-feature-experiments")
        # Only presentation files may come from a later, explicitly pinned review commit.
        try:
            response = s3.get_object(Bucket=bucket, Key=prefix + "/report_ref.json")
        except s3.exceptions.NoSuchKey:
            report_commit = commit
        else:
            with response["Body"] as body:
                report_commit = json.loads(body.read())["commit"]
        if report_commit != commit:
            with tempfile.TemporaryDirectory() as temporary:
                stage = Path(temporary)
                repository_archive(report_commit, stage / "report.tar.gz")
                unpack(stage / "report.tar.gz", stage / "report", repository=True)
                candidates = [
                    *(stage / "report/notebooks").glob("*.ipynb"),
                    *(stage / "report/docs").glob("*.md"),
                    stage / "report/README.md",
                    stage / "report/START_HERE.md",
                    stage / "report/scripts/feature_attribution.py",
                    stage / "report/scripts/feature_attribution.py.lock",
                    stage / "report/scripts/validate_gateway.py",
                    stage / "report/scripts/validate_gateway.py.lock",
                ]
                for path in candidates:
                    if not path.is_file():
                        continue
                    target = root / path.relative_to(stage / "report")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        report_scripts = [
            str(p.relative_to(root)) for p in (
                root / "scripts/feature_attribution.py", root / "scripts/validate_gateway.py"
            ) if p.exists()
        ]
        if report_scripts:
            command([python, "-m", "ruff", "check", "--fix", *report_scripts], "report-script-lint")
            command([python, "-m", "ruff", "format", *report_scripts], "report-script-format")
        if (root / "scripts/feature_attribution.py").exists():
            command(
                [uv, "run", "--locked", "scripts/feature_attribution.py"],
                "wide-attribution",
                threads=6,
            )
            backup("wide-attribution")
        if (root / "scripts/validate_gateway.py").exists():
            if not (root / "scripts/validate_gateway.py.lock").exists():
                command([uv, "lock", "--script", "scripts/validate_gateway.py"], "gateway-lock")
            command([uv, "run", "--locked", "scripts/validate_gateway.py"], "organizer-gateway", threads=2)
        command([python, "scripts/notebooks.py", "--publish"], "publish-notebooks", threads=2)
        command([python, "scripts/quality.py"], "quality", threads=2)
        backup("published-and-tested")
        files = [
            *root.glob("notebooks/*.ipynb"),
            *root.glob("docs/results/*"),
            root / "artifacts/quality.json",
            root / "artifacts/notebooks/publication.json",
        ]
        files.extend(
            p for pattern in ("scripts/feature_attribution.py*", "scripts/validate_gateway.py*")
            for p in root.glob(pattern)
        )
        publication = []
        for path in files:
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                key = prefix + "/public/" + relative
                s3.upload_file(
                    str(path),
                    bucket,
                    key,
                    ExtraArgs={
                        "ServerSideEncryption": "AES256",
                        "Metadata": {"sha256": digest(path)},
                    },
                )
                entry = {"path": relative, "key": key, "sha256": digest(path)}
                if path.suffix == ".png":
                    entry["transfer_key"] = key + ".base64"
                    s3.put_object(
                        Bucket=bucket,
                        Key=entry["transfer_key"],
                        Body=base64.b64encode(path.read_bytes()),
                        ServerSideEncryption="AES256",
                    )
                publication.append(entry)
        s3.put_object(
            Bucket=bucket,
            Key=prefix + "/publication.json",
            Body=json.dumps(
                {"files": publication, "code_commit": commit, "report_commit": report_commit}
            ).encode(),
            ServerSideEncryption="AES256",
        )
        event("completed", report_commit=report_commit, holdout_evaluation="not_run")
    except BaseException as exc:
        event("failed", error_type=type(exc).__name__)
        # AWS owns job shutdown. Completed phase checkpoints remain in the private bucket.
        raise


if __name__ == "__main__":
    main()
