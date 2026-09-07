"""Interrupted, changed, and corrupted work must never be silently reused."""

import json
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

from nfl_trajectory.runtime import Run, atomic_bytes, fingerprint, stage


def test_completed_stage_is_reused_but_corruption_and_changed_signature_recompute(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    executions = []

    def action() -> None:
        executions.append(1)
        atomic_bytes(output, b"valid")

    with Run(tmp_path, "test") as run:
        assert stage(tmp_path, "one", "a", [output], action, run)
        assert not stage(tmp_path, "one", "a", [output], action, run)
        output.write_text("corrupt")
        assert stage(tmp_path, "one", "a", [output], action, run)
        assert stage(tmp_path, "one", "b", [output], action, run)
    assert len(executions) == 3


def test_interrupted_stage_does_not_claim_completion(tmp_path: Path) -> None:
    output = tmp_path / "result"

    def interrupted() -> None:
        output.write_text("partial")
        raise KeyboardInterrupt

    with Run(tmp_path, "test") as run:
        with pytest.raises(KeyboardInterrupt):
            stage(tmp_path, "one", "a", [output], interrupted, run)
        assert json.loads((tmp_path / ".state/one.json").read_text())["status"] == "failed"
        assert stage(tmp_path, "one", "a", [output], lambda: output.write_text("done"), run)


def test_stage_lock_prevents_concurrent_writer(tmp_path: Path) -> None:
    (tmp_path / ".state").mkdir()
    with FileLock(str(tmp_path / ".state/one.json.lock")), Run(tmp_path, "test") as run:
        with pytest.raises(Timeout):
            stage(tmp_path, "one", "a", [], lambda: None, run)


def test_heartbeat_and_total_time_logged(tmp_path: Path) -> None:
    import threading

    with Run(tmp_path, "test", heartbeat_seconds=0.005) as run:
        assert not threading.Event().wait(0.025)
    rows = [json.loads(line) for line in run.log_path.read_text().splitlines()]
    assert rows[0]["event"] == "started"
    assert rows[-1]["event"] == "completed"
    assert any(row["event"] == "heartbeat" for row in rows)
    assert all("timestamp" in row and "elapsed_seconds" in row for row in rows)


def test_source_change_invalidates_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/model.py"
    source.write_text("a = 1")
    before = fingerprint(tmp_path, [], {})
    source.write_text("a = 2")
    assert fingerprint(tmp_path, [], {}) != before
