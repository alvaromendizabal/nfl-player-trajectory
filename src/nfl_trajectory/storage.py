"""Private, content-addressed S3 snapshots with checksum-verified restoration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock

from nfl_trajectory.data import safe_relative
from nfl_trajectory.runtime import Run, atomic_json, sha256

ALLOWED = {"data", "artifacts", "logs", ".state"}


def client(region: str) -> Any:
    import boto3
    from botocore.config import Config

    return boto3.Session(region_name=region).client(
        "s3",
        config=Config(
            retries={"mode": "standard", "total_max_attempts": 5},
            connect_timeout=10,
            read_timeout=120,
        ),
    )


def backup(root: Path, bucket: str, run: Run, s3: Any) -> str:
    """Upload completed immutable artifacts; publish a manifest only after all uploads succeed.

    Invoke after a command finishes. Active logs and partial downloads are excluded.
    The caller must not run another pipeline writer concurrently with the snapshot.
    """
    (root / ".state").mkdir(exist_ok=True)
    with FileLock(str(root / ".state" / "backup.lock"), timeout=1):
        known: set[str] = set()
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="objects/"):
            known.update(item["Key"] for item in page.get("Contents", []))
        candidates = sorted(
            p
            for prefix in ALLOWED
            for p in (root / prefix).rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and not p.name.endswith(".lock")
            and p != run.log_path
            and p.name != "last_backup.json"
        )
        entries = []
        for i, path in enumerate(candidates, 1):
            digest = sha256(path)
            key = f"objects/{digest}"
            if key not in known:
                s3.upload_file(
                    str(path),
                    bucket,
                    key,
                    ExtraArgs={
                        "ServerSideEncryption": "AES256",
                        "Metadata": {"sha256": digest},
                    },
                )
                # Detect a changing local source before any snapshot manifest can be published.
                if sha256(path) != digest:
                    raise ValueError(
                        "An artifact changed during backup; stop other writers and rerun."
                    )
                known.add(key)
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": digest,
                    "size": path.stat().st_size,
                }
            )
            run.event("backup_progress", completed_files=i, total_files=len(candidates))
        payload = json.dumps({"format": 1, "files": entries}, sort_keys=True).encode()
        manifest = f"snapshots/{hashlib.sha256(payload).hexdigest()}.json"
        s3.put_object(Bucket=bucket, Key=manifest, Body=payload, ServerSideEncryption="AES256")
        atomic_json(
            root / "artifacts" / "last_backup.json", {"bucket": bucket, "manifest": manifest}
        )
        run.event("backup_completed", bucket=bucket, manifest=manifest)
        return manifest


def restore(root: Path, bucket: str, manifest: str, run: Run, s3: Any) -> None:
    """Restore missing or checksum-matching files; never overwrite divergent local work."""
    name = safe_relative(manifest)
    if name.parts[0] != "snapshots" or name.suffix != ".json":
        raise ValueError("Choose a snapshots/<sha256>.json manifest returned by backup.")
    response = s3.get_object(Bucket=bucket, Key=manifest)
    with response["Body"] as body:
        payload = body.read()
    if hashlib.sha256(payload).hexdigest() != name.stem:
        raise ValueError("Snapshot manifest checksum failed.")
    contents = json.loads(payload)
    if contents.get("format") != 1:
        raise ValueError("Unsupported snapshot format.")
    entries = contents["files"]
    destinations = []
    for item in entries:
        relative = safe_relative(item["path"])
        if (
            relative.parts[0] not in ALLOWED
            or (root / relative).resolve().is_relative_to(root.resolve()) is False
        ):
            raise ValueError("Snapshot contains an unsupported destination.")
        if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("Snapshot contains an invalid checksum.")
        destination = root / relative
        if destination.exists() and sha256(destination) != item["sha256"]:
            raise ValueError(
                "Local file differs from snapshot; restore into a fresh project directory."
            )
        destinations.append(destination)
    if len(destinations) != len(set(destinations)):
        raise ValueError("Snapshot has duplicate destinations.")
    for i, (item, destination) in enumerate(zip(entries, destinations, strict=True), 1):
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, name_on_disk = tempfile.mkstemp(dir=destination.parent)
            os.close(fd)
            temporary = Path(name_on_disk)
            try:
                s3.download_file(bucket, f"objects/{item['sha256']}", str(temporary))
                if sha256(temporary) != item["sha256"] or temporary.stat().st_size != item["size"]:
                    raise ValueError("Restored object checksum or size failed.")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        run.event("restore_progress", completed_files=i, total_files=len(entries))
