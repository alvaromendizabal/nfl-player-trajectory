"""Command-line entry point; errors fail visibly without exposing credentials."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from nfl_trajectory.benchmark import benchmark
from nfl_trajectory.context_experiment import context_research
from nfl_trajectory.data import audit, download
from nfl_trajectory.feature_experiment import feature_experiment
from nfl_trajectory.feature_research import feature_research
from nfl_trajectory.report import demo
from nfl_trajectory.runtime import Run, atomic_json
from nfl_trajectory.storage import backup, client, restore


def find_root() -> Path:
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "pyproject.toml").exists() and (path / "src" / "nfl_trajectory").is_dir():
            return path
    raise ValueError("Run from the nfl-player-trajectory project directory.")


def preflight(root: Path, run: Run) -> None:
    free_gib = shutil.disk_usage(root).free / 1024**3
    if free_gib < 5:
        raise ValueError("At least 5 GiB free storage is required for Phase 0.")
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Use the locked Python 3.11 environment created by bootstrap.py.")
    versions = {
        package: importlib.metadata.version(package)
        for package in ["numpy", "pandas", "plotly", "kaggle", "boto3"]
    }
    payload = {
        "status": "passed",
        "python": sys.version.split()[0],
        "free_gib": round(free_gib, 2),
        "versions": versions,
        "gpu_required": False,
    }
    atomic_json(root / "artifacts" / "preflight.json", payload)
    run.event("preflight_passed", **payload)


def status_summary(path: Path) -> object:
    if not path.exists():
        return "not_run"
    value = json.loads(path.read_text())
    if path.name == "audit_summary.json":
        return {key: item for key, item in value.items() if key != "pairs"}
    if path.name == "summary.json":
        return {
            key: value[key]
            for key in [
                "status",
                "selected_model",
                "validation_games",
                "validation_rows_per_model",
                "improvement_vs_velocity_percent",
                "holdout_evaluation",
            ]
        }
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "download",
            "audit",
            "demo",
            "status",
            "backup",
            "restore",
            "benchmark",
            "features",
            "feature-research",
            "context-research",
            "research-report",
        ],
    )
    parser.add_argument("--bucket")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--checkpoint-s3",
        action="store_true",
        help="Back up after each prepared week, fitted model, and feature report.",
    )
    args = parser.parse_args()
    try:
        root = find_root()
        with Run(root, args.command) as run:
            if args.command == "preflight":
                preflight(root, run)
            elif args.command == "download":
                download(root, run)
            elif args.command == "audit":
                audit(root, run)
            elif args.command == "demo":
                demo(root, run)
            elif args.command == "benchmark":
                benchmark(root, run)
            elif args.command == "research-report":
                from nfl_trajectory.research import publish_research_report

                publish_research_report(root, run)
            elif args.command in ("features", "feature-research", "context-research"):
                checkpoint: Callable[[], object] | None = None
                if args.checkpoint_s3:
                    config = root / "aws.local.json"
                    feature_bucket = args.bucket or (
                        json.loads(config.read_text())["bucket"] if config.exists() else None
                    )
                    if not feature_bucket:
                        raise ValueError("S3 checkpointing requires --bucket or aws.local.json.")
                    feature_client = client(args.region)
                    checkpoint = partial(backup, root, feature_bucket, run, feature_client)
                experiment = {
                    "features": feature_experiment,
                    "feature-research": feature_research,
                    "context-research": context_research,
                }[args.command]
                experiment(root, run, checkpoint)
            elif args.command == "status":
                for name in [
                    "preflight.json",
                    "audit_summary.json",
                    "benchmark/summary.json",
                    "features/summary.json",
                    "last_backup.json",
                ]:
                    path = root / "artifacts" / name
                    run.event("artifact_status", file=name, result=status_summary(path))
            else:
                bucket = args.bucket
                local = root / "aws.local.json"
                if not bucket and local.exists():
                    bucket = json.loads(local.read_text())["bucket"]
                if not bucket:
                    raise ValueError("Provide --bucket or the supplied aws.local.json.")
                s3 = client(args.region)
                if args.command == "backup":
                    backup(root, bucket, run, s3)
                else:
                    if not args.manifest:
                        raise ValueError("Provide the --manifest printed by a successful backup.")
                    restore(root, bucket, args.manifest, run, s3)
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(
            f"STOPPED ({type(exc).__name__}). Completed checkpoints are retained.", file=sys.stderr
        )
        if isinstance(exc, (ValueError, FileNotFoundError)):
            print(str(exc), file=sys.stderr)
        else:
            print(
                "Check account access, network, and the last stage in logs/. "
                "No credentials or signed URLs are written to the log.",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
