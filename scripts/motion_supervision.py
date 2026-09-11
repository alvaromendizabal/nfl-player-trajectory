# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "boto3==1.43.89", "numpy==2.4.6", "pandas==3.0.5", "plotly==7.0.0",
#   "matplotlib==3.10.8", "filelock==3.32.5", "torch==2.8.0", "pytest==9.1.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Bounded motion-supervision engineering checks and training-only preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.runtime import atomic_json, sha256  # noqa: E402

SAMPLE_SHA256 = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
PLAN_SOURCES = [
    "src/nfl_trajectory/supervision_plan.py",
    "src/nfl_trajectory/motion_targets.py",
    "src/nfl_trajectory/temporal_data.py",
    "configs/motion_supervision.json",
]
TESTS = [
    "tests/test_motion_supervision.py",
    "tests/test_supervision_batches.py",
    "tests/test_supervision_execution.py",
    "tests/test_supervision_experiment.py",
    "tests/test_motion_experiment_runner.py",
    "tests/test_motion_targets.py",
    "tests/test_temporal_data.py",
    "tests/test_temporal_model.py",
]
SOURCES = [
    "src/nfl_trajectory/motion_supervision.py",
    "src/nfl_trajectory/supervision_batches.py",
    "src/nfl_trajectory/supervision_profile.py",
    "src/nfl_trajectory/supervision_experiment.py",
    "src/nfl_trajectory/supervision_evidence.py",
    "src/nfl_trajectory/motion_targets.py",
    "src/nfl_trajectory/temporal_data.py",
    "src/nfl_trajectory/temporal_model.py",
    "src/nfl_trajectory/runtime.py",
    "scripts/motion_supervision.py",
    "scripts/run_motion_experiment.py",
    "scripts/motion_supervision.py.lock",
    "configs/motion_supervision.json",
    *TESTS,
]


def prepare_training_plan() -> dict[str, Any]:
    """Create the private training-only plan from the already verified inner_1 cache."""
    from nfl_trajectory.supervision_plan import training_plan

    started = time.monotonic()
    cache = ROOT / "artifacts/temporal/research/inner_1/samples.pkl"
    if not cache.is_file() or cache.is_symlink() or sha256(cache) != SAMPLE_SHA256:
        raise ValueError("Existing private sample is missing or has the wrong checksum.")
    source_hashes = {name: sha256(ROOT / name) for name in PLAN_SOURCES}
    destination = ROOT / "artifacts/motion_supervision/preflight/training_plan.json"
    if destination.is_file():
        old = json.loads(destination.read_text())
        if old.get("sample_sha256") != SAMPLE_SHA256 or old.get("source_hashes") != source_hashes:
            raise ValueError(
                "Existing training plan uses different source/input; preserve and review."
            )
        return {**old, "reused": True}
    with cache.open("rb") as stream:
        samples = pickle.load(stream)
    training = [sample for sample in samples if str(sample["split"]) == "train"]
    plan = training_plan(training, 64, 2026, 0)
    expected = (4951, 94, 193452, 368)
    observed = (
        plan["training_plays"],
        plan["training_games"],
        plan["training_rows"],
        plan["training_rows_after_frame_48"],
    )
    if observed != expected:
        raise ValueError(f"Private training population differs: {observed!r} != {expected!r}")
    result = {
        "status": "private_training_plan_verified",
        "sample_sha256": SAMPLE_SHA256,
        "source_hashes": source_hashes,
        "plan": plan,
        "validation_labels_used": False,
        "scientific_fits": 0,
        "new_rmse": None,
        "feature_completion_gate": "open",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(destination, result)
    return result


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop the test process and its recovery-test children; no external jobs exist."""
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def run_tests() -> int:
    from nfl_trajectory.supervision_evidence import runtime_identity

    started = time.monotonic()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = ROOT / "artifacts/quality/motion_supervision" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    log_path = folder / "tests.log"
    xml_path = folder / "tests.xml"
    events_path = folder / "events.jsonl"
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    signature = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    result: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "scope": "synthetic engineering tests, not a private-data or scientific fit",
        "source_hashes": source_hashes,
        "source_signature": signature,
        "runtime": runtime_identity(),
        "budget_seconds": 180,
        "heartbeat_seconds": 15,
        "new_rmse": None,
        "scientific_fits": 0,
        "aws_mutations": 0,
        "git_updated": False,
        "remote_checkpoint_verified": False,
    }

    def event(name: str, **fields: Any) -> None:
        row = {
            "utc": datetime.now(UTC).isoformat(),
            "event": name,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            **fields,
        }
        text = json.dumps(row, sort_keys=True)
        print(text, flush=True)
        with events_path.open("a") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONPATH=str(ROOT / "src"),
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *TESTS,
        "-q",
        "--maxfail=1",
        "--junitxml=" + str(xml_path),
    ]
    event("synthetic_tests_started", tests=TESTS, log=str(log_path))
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            next_heartbeat = time.monotonic() + 15
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= 180:
                    raise TimeoutError("Synthetic test budget exhausted; no automatic retry.")
                if time.monotonic() >= next_heartbeat:
                    event("heartbeat", log_bytes=log_path.stat().st_size)
                    next_heartbeat = time.monotonic() + 15
                time.sleep(0.2)
        result["exit_code"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError("Targeted tests failed; inspect the saved test log.")
        suites = list(ET.parse(xml_path).getroot().iter("testsuite"))
        counts = {
            name: sum(int(s.attrib.get(name, 0)) for s in suites)
            for name in ("tests", "failures", "errors", "skipped")
        }
        result["test_counts"] = counts
        if counts["tests"] < 55 or any(counts[name] for name in ("failures", "errors", "skipped")):
            raise RuntimeError("Required tests did not all execute successfully.")
        result["status"] = "synthetic_implementation_verified_no_scientific_fit"
    except (Exception, KeyboardInterrupt) as exc:
        if process is not None:
            stop_process(process)
        result["status"] = (
            "stopped" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "blocked"
        )
        result["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["artifacts"] = {p.name: sha256(p) for p in (log_path, xml_path) if p.exists()}
        atomic_json(folder / "result.json", result)
        atomic_json(ROOT / "artifacts/quality/motion_supervision.json", result)
        event("finished", status=result["status"], report=str(folder / "result.json"))
        print(
            log_path.read_text()[-12000:] if log_path.exists() else "No test log created.",
            flush=True,
        )
    return 0 if result["status"] == "synthetic_implementation_verified_no_scientific_fit" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--prepare-training-plan", action="store_true")
    modes.add_argument("--profile-training", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return run_tests()
    if args.prepare_training_plan:
        print(json.dumps(prepare_training_plan(), sort_keys=True), flush=True)
        return 0
    from nfl_trajectory.supervision_profile import profile_training

    print(json.dumps(profile_training(ROOT), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
