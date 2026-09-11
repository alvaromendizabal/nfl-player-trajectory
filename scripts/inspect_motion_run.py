"""Read one existing NFL job and verify its saved bytes; never launch training.

Use inside AWS with the existing account identity. At most 64 MiB is downloaded.
Checkpoint SHA verification is not a claim of numerical model replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCOUNT = "560403859723"
BUCKET = "sagemaker-nfl-trajectory-560403859723-us-west-2"
JOB = "nfl-motion-scientific-20260911-055510-d265d9d"
SOURCE = "d265d9decb4ca5cd829781ceb78bd043480b5c35"
SAMPLE = "d84874f879e3d54d6f4ef66caefd965a9af9407c677c3c37ebc0245d3a9bb61d"
KEYS = ("game_id", "play_id", "nfl_id", "frame_id")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_key(key: str) -> str:
    if not isinstance(key, str) or key.startswith("/") or ".." in key.split("/"):
        raise ValueError("Unsafe S3 key")
    if not key.startswith((f"cloud-runs/{JOB}/", "experiments/velocity_isolation/inner_1/")):
        raise ValueError("S3 key is outside the named experiment")
    return key


class Reader:
    """Finite-size, finite-time reader retaining version and checksum receipts."""

    def __init__(self, client: Any, output: Path) -> None:
        self.client, self.output = client, output
        self.total = 0
        self.started = time.monotonic()
        self.receipts: list[dict[str, Any]] = []
        output.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, name: str, limit: int, expected: str | None = None) -> bytes:
        from botocore.exceptions import ClientError

        safe_key(key)
        if time.monotonic() - self.started > 180:
            raise TimeoutError("Read-only inspection exceeded its 180-second budget")
        if Path(name).name != name or self.total + limit > 64 * 1024**2:
            raise ValueError("Unsafe filename or download budget exhausted")
        try:
            response = self.client.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=ACCOUNT)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise FileNotFoundError(key) from exc
            raise
        body = response["Body"]
        try:
            length = int(response.get("ContentLength", limit + 1))
            if not 0 <= length <= limit:
                raise ValueError("Object exceeds the declared size limit")
            payload = body.read(length + 1)
        finally:
            body.close()
        if len(payload) != length:
            raise ValueError("Incomplete object read")
        actual = digest(payload)
        if expected is not None and actual != expected:
            raise ValueError("Object SHA256 differs from its immutable receipt")
        destination = self.output / name
        if destination.is_symlink():
            raise ValueError("Refusing a symlink destination")
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.replace(destination)
        self.total += len(payload)
        self.receipts.append({"key": key, "file": name, "sha256": actual,
                              "bytes": len(payload), "version": response.get("VersionId")})
        print(json.dumps({"utc": datetime.now(UTC).isoformat(), "event": "object_verified",
                          "file": name, "bytes": len(payload)}), flush=True)
        return payload


def error_rows(payload: bytes) -> dict[tuple[int, ...], tuple[float, float]]:
    rows: dict[tuple[int, ...], tuple[float, float]] = {}
    for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
        key = tuple(int(row[name]) for name in KEYS)
        delta = (float(row["dx"]), float(row["dy"]))
        if key in rows or not all(math.isfinite(x) for x in delta):
            raise ValueError("Duplicate forecast key or nonfinite saved error")
        rows[key] = delta
    if not rows:
        raise ValueError("No forecast errors")
    return rows


def pair_totals(control: bytes, treatment: bytes) -> dict[str, Any]:
    a, b = error_rows(control), error_rows(treatment)
    if a.keys() != b.keys():
        raise ValueError("Arms contain different forecast keys")
    groups: dict[int, dict[str, Any]] = {}
    for key in sorted(a):
        group = groups.setdefault(key[0], {"game_id": key[0], "rows": 0,
                                          "control_sse": 0.0, "velocity_sse": 0.0})
        group["rows"] += 1
        group["control_sse"] += sum(x * x for x in a[key])
        group["velocity_sse"] += sum(x * x for x in b[key])
    games = list(groups.values())
    ca = math.sqrt(math.fsum(g["control_sse"] for g in games) / (2 * len(a)))
    cb = math.sqrt(math.fsum(g["velocity_sse"] for g in games) / (2 * len(a)))
    return {"rows": len(a), "games": len(games), "control_rmse": ca,
            "velocity_rmse": cb, "relative_gain": 1 - cb / ca if ca else None,
            "per_game": games}


def inspect(output: Path, session: Any | None = None) -> dict[str, Any]:
    import boto3
    from botocore.config import Config

    output.mkdir(parents=True, exist_ok=True)
    config = Config(connect_timeout=5, read_timeout=15, retries={"total_max_attempts": 2})
    session = session or boto3.Session(region_name="us-west-2")
    reader = Reader(session.client("s3", config=config), output)
    result: dict[str, Any] = {"job": JOB, "observed_utc": datetime.now(UTC).isoformat(),
                              "new_training_jobs": 0, "numerical_model_replay": False,
                              "feature_research": "open"}
    try:
        identity = session.client("sts", config=config).get_caller_identity()
        if identity["Account"] != ACCOUNT:
            raise ValueError("Unexpected AWS account; no project artifacts were read")
        job = session.client("sagemaker", config=config).describe_processing_job(ProcessingJobName=JOB)
        result["job_status"] = job["ProcessingJobStatus"]
        result["failure_reason"] = job.get("FailureReason")
        result["job_source"] = job.get("Environment", {}).get("NFL_REPO_REF")
        if result["job_source"] != SOURCE:
            raise ValueError("Job source differs from the known pinned experiment")
        for field in ("CreationTime", "ProcessingStartTime", "ProcessingEndTime"):
            result[field] = str(job.get(field))
        statuses = {}
        for name in ("status.json", "scientific-status.json"):
            try:
                statuses[name] = json.loads(reader.get(f"cloud-runs/{JOB}/{name}", name, 1024**2))
            except FileNotFoundError:
                statuses[name] = None
        result["last_status"] = statuses
        status = statuses.get("scientific-status.json") or {}
        key = status.get("summary_key")
        if not key:
            result["status"] = "existing_run_requires_diagnosis_or_completion"
            try:
                reader.get(f"cloud-runs/{JOB}/logs/matched-scientific-experiment.log",
                           "scientific.log", 2 * 1024**2)
            except FileNotFoundError:
                pass
            return result
        match = re.fullmatch(
            r"experiments/velocity_isolation/inner_1/([0-9a-f]{64})/summaries/([0-9a-f]{64})\.json",
            key,
        )
        if match is None:
            raise ValueError("Summary key is not an immutable study result")
        study, summary_sha = match.groups()
        summary = json.loads(reader.get(key, "summary.json", 2 * 1024**2, summary_sha))
        if (summary.get("source_commit"), summary.get("sample_sha256")) != (SOURCE, SAMPLE):
            raise ValueError("Summary source or sample differs from the known experiment")
        if summary.get("study_signature") != study or summary.get("status") != "completed":
            raise ValueError("Summary signature or completion status is invalid")
        prefix = f"experiments/velocity_isolation/inner_1/{study}/"
        errors = {}
        for arm in ("coordinate", "velocity"):
            record = summary["arms"][arm]
            checkpoint = record["final_checkpoint"]
            if record["final_step"] != 1248 or checkpoint["step"] != 1248:
                raise ValueError("The frozen optimizer exposure did not complete")
            sha = checkpoint["sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("Invalid checkpoint digest")
            reader.get(prefix + arm + "/" + sha + ".pt", arm + ".pt", 12 * 1024**2, sha)
            receipt = record["private_errors"]
            if not receipt["key"].startswith(prefix + "errors/" + arm + "-"):
                raise ValueError("Error artifact belongs to another study")
            errors[arm] = reader.get(receipt["key"], arm + "_errors.csv", 12 * 1024**2,
                                     receipt["sha256"])
        totals = pair_totals(errors["coordinate"], errors["velocity"])
        if (totals["rows"], totals["games"]) != (83938, 41):
            raise ValueError("Evaluation row/game population differs from the frozen study")
        evaluation = summary["evaluation"]
        for field, arm in (("control_rmse", "control"), ("velocity_rmse", "velocity")):
            if abs(totals[field] - evaluation[arm]["coordinate_rmse_yards"]) > 1e-6:
                raise ValueError("Recomputed RMSE disagrees with the reported metric")
        result["recomputed"] = totals
        result["reported_evaluation"] = evaluation
        result["status"] = "saved_errors_and_checkpoint_bytes_verified"
        result["next_gate"] = "recompute paired bootstrap and numerically restore models"
        return result
    except Exception as exc:
        result["status"] = "stopped_review_required"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["downloads"] = reader.receipts
        result["elapsed_seconds"] = round(time.monotonic() - reader.started, 3)
        destination = output / "inspection.json"
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": result.get("status"), "output": str(destination)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.output)
    if result["status"] == "stopped_review_required":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
