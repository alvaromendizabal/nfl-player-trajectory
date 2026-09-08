"""Run the complete offline quality gate with timestamps and heartbeats."""

from __future__ import annotations

import sys
from pathlib import Path

from nfl_trajectory.runtime import Run, atomic_json, checked_command


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        with Run(root, "quality") as run:
            commands = [
                [
                    sys.executable,
                    "kaggle/export.py",
                    "--output",
                    "artifacts/quality/exports/submission.ipynb",
                ],
                [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
                [sys.executable, "-m", "ruff", "check", "."],
                [sys.executable, "-m", "ruff", "format", "--check", "."],
                [sys.executable, "-m", "mypy", "src", "scripts", "kaggle"],
                [sys.executable, "-m", "pytest", "-q"],
                [sys.executable, "-m", "nfl_trajectory.cli", "preflight"],
                [sys.executable, "-m", "nfl_trajectory.cli", "demo"],
                [sys.executable, "-m", "nfl_trajectory.cli", "demo"],
                [sys.executable, "scripts/notebooks.py"],
                [
                    sys.executable,
                    "kaggle/export.py",
                    "--model",
                    "role_ridge",
                    "--weights",
                    "docs/results/model.json",
                    "--output",
                    "artifacts/quality/exports/submission.ipynb",
                ],
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    "--no-respect-gitignore",
                    "artifacts/quality/exports/submission.ipynb",
                ],
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "format",
                    "--check",
                    "--no-respect-gitignore",
                    "artifacts/quality/exports/submission.ipynb",
                ],
            ]
            for command in commands:
                checked_command(command, root, run)
            atomic_json(
                root / "artifacts" / "quality.json",
                {
                    "status": "passed",
                    "run_id": run.run_id,
                    "scope": "offline tests, synthetic integration, notebook; no live Kaggle score",
                    "checks": len(commands),
                },
            )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Quality gate stopped: {type(exc).__name__}. See the failing check above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
