"""Crash-safe local checkpoint generations for the new velocity study.

A verified immutable tensor blob is written first. The atomic checkpoint.json
pointer is advanced last. This tests local persistence, not S3 durability;
remote upload and independent download verification remain mandatory gates.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "nfl-velocity-checkpoint-v1"


def runtime_identity() -> dict[str, str | int | bool]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "machine": platform.machine(),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "device": "cpu",
    }


def _atomic(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError("Checkpoint destinations cannot be symlinks.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as f:
            temporary = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def save_generation(folder: Path, payload: dict[str, Any], signature: str) -> dict[str, Any]:
    """Publish a generation without destroying the previously committed generation."""
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise ValueError("A full source/data/config SHA256 signature is required.")
    if folder.is_symlink():
        raise ValueError("Checkpoint folder cannot be a symlink.")
    folder.mkdir(parents=True, exist_ok=True)
    if folder.is_symlink():
        raise ValueError("Checkpoint folder cannot be a symlink.")
    pointer = folder / "checkpoint.json"
    if pointer.is_symlink():
        raise ValueError("Checkpoint pointer cannot be a symlink.")
    if pointer.exists():
        previous = json.loads(pointer.read_text())
        if previous.get("signature") != signature:
            raise ValueError("Refusing to overwrite a different experiment.")
        if previous.get("step", -1) > int(payload["steps"]):
            raise ValueError("Refusing to replace a newer committed checkpoint with older state.")
    buffer = io.BytesIO()
    torch.save({"format": FORMAT, "signature": signature, "state": payload}, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    blob = folder / (digest + ".pt")
    if blob.is_symlink():
        raise ValueError("Checkpoint blob cannot be a symlink.")
    if blob.exists():
        if hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
            raise ValueError("Existing immutable checkpoint blob is corrupt.")
    else:
        _atomic(blob, data)
    # Independently read bytes from disk before advancing the pointer.
    if hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
        raise ValueError("Checkpoint read-back hash verification failed.")
    receipt = {
        "format": FORMAT,
        "signature": signature,
        "sha256": digest,
        "bytes": len(data),
        "step": int(payload["steps"]),
        "runtime": runtime_identity(),
        "remote_verified": False,
    }
    _atomic(pointer, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return receipt


def load_generation(folder: Path, signature: str) -> dict[str, Any]:
    """Reject changed bytes, experiment signatures or exact-replay runtimes."""
    if folder.is_symlink():
        raise ValueError("Checkpoint folder cannot be a symlink.")
    pointer = folder / "checkpoint.json"
    if pointer.is_symlink():
        raise ValueError("Checkpoint pointer cannot be a symlink.")
    receipt = json.loads(pointer.read_text())
    if receipt.get("format") != FORMAT or receipt.get("signature") != signature:
        raise ValueError("Checkpoint format/signature mismatch.")
    digest = receipt.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid checkpoint digest.")
    if receipt.get("runtime") != runtime_identity():
        raise ValueError("Exact-replay runtime changed; do not reuse this state silently.")
    blob = folder / (digest + ".pt")
    if blob.is_symlink():
        raise ValueError("Checkpoint blob cannot be a symlink.")
    data = blob.read_bytes()
    if len(data) != receipt["bytes"] or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Checkpoint content hash/size mismatch.")
    saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if saved["format"] != FORMAT or saved["signature"] != signature:
        raise ValueError("Tensor payload provenance mismatch.")
    if int(saved["state"]["steps"]) != receipt["step"]:
        raise ValueError("Tensor state and receipt cursor disagree.")
    return saved["state"]
