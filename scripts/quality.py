"""Run all maintained package tests and the public evaluation, without cloud writes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "-m", "mypy", "src", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-S", "scripts/run_portfolio_demo.py", "--output", "artifacts/portfolio"],
        [
            sys.executable,
            "-S",
            "scripts/validate_portfolio_release.py",
            "--demo-output",
            "artifacts/portfolio",
        ],
    ]
    for command in commands:
        print("CHECK", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True, timeout=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
