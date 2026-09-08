"""Structured progress, atomic checkpoints, and process-safe resumability."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


class Run:
    """One command's timestamped progress stream; heartbeats never imply task completion."""

    def __init__(self, root: Path, name: str, heartbeat_seconds: float = 15.0) -> None:
        self.root = root
        self.name = name
        self.run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.log_path = root / "logs" / f"{self.run_id}-{name}.jsonl"
        self.started = time.monotonic()
        self.stage_name = name
        self.stage_started = self.started
        self.interval = heartbeat_seconds
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def event(self, event: str, **fields: Any) -> None:
        now = time.monotonic()
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            "elapsed_seconds": round(now - self.started, 3),
            "total_elapsed_seconds": round(now - self.started, 3),
            "stage_elapsed_seconds": round(now - self.stage_started, 3),
            "stage": self.stage_name,
            **fields,
        }
        line = json.dumps(record, allow_nan=False)
        with self.lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
            print(line, flush=True)

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval):
            self.event("heartbeat", status="running")

    def __enter__(self) -> Run:
        self.event("started", command=self.name)
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop.set()
        self.thread.join()
        # Do not log exception strings: SDK errors can contain signed URLs or credentials.
        self.event(
            "completed" if exc_type is None else "failed",
            error_type=None if exc_type is None else exc_type.__name__,
        )


def stage(
    root: Path,
    name: str,
    signature: str,
    outputs: list[Path],
    action: Callable[[], None],
    run: Run,
) -> bool:
    """Reuse only verified outputs for the same input/code signature. Return True if executed."""
    state_dir = root / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state = state_dir / f"{name}.json"
    with FileLock(str(state) + ".lock", timeout=1):
        previous: dict[str, Any] = {}
        if state.exists():
            try:
                previous = json.loads(state.read_text())
            except (json.JSONDecodeError, OSError):
                previous = {}
        matches = previous.get("signature") == signature and previous.get("status") == "completed"
        if matches and all(
            p.is_file() and previous.get("outputs", {}).get(str(p.relative_to(root))) == sha256(p)
            for p in outputs
        ):
            run.event(
                "stage_reused", stage=name, original_elapsed_seconds=previous.get("elapsed_seconds")
            )
            return False
        stage_started = time.monotonic()
        previous_stage = (run.stage_name, run.stage_started)
        run.stage_name, run.stage_started = name, stage_started
        run.event("stage_started", stage=name)
        atomic_json(state, {"status": "running", "signature": signature})
        try:
            action()
            hashes = {str(p.relative_to(root)): sha256(p) for p in outputs}
        except BaseException:
            atomic_json(state, {"status": "failed", "signature": signature})
            run.stage_name, run.stage_started = previous_stage
            raise
        atomic_json(
            state,
            {
                "status": "completed",
                "signature": signature,
                "outputs": hashes,
                "elapsed_seconds": round(time.monotonic() - stage_started, 3),
            },
        )
        run.event(
            "stage_completed",
            stage=name,
            elapsed_stage_seconds=round(time.monotonic() - stage_started, 3),
        )
        run.stage_name, run.stage_started = previous_stage
        return True


def fingerprint(root: Path, inputs: list[Path], parameters: Any) -> str:
    """Include source and environment lock so changed code cannot reuse old computations."""
    paths = sorted(set(inputs + list((root / "src").rglob("*.py"))))
    if (root / "uv.lock").exists():
        paths.append(root / "uv.lock")
    value = {
        "files": {str(p.relative_to(root)): sha256(p) for p in paths},
        "parameters": parameters,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_command(command: list[str], root: Path, run: Run) -> None:
    run.event("check_started", program=command[0], arguments=command[1:])
    subprocess.run(command, cwd=root, check=True)
    run.event("check_completed", program=command[0])
