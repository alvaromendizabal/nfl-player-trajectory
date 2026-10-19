"""Public data audit and deterministic reference, never private training or submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from nfl_trajectory.data import audit, download
from nfl_trajectory.portfolio_demo import run_demo
from nfl_trajectory.runtime import Run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "audit", "download"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/portfolio"))
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "src/nfl_trajectory").is_dir():
        parser.error("Run from the repository root.")
    if args.command == "demo":
        run_demo(args.output)
    else:
        with Run(root, args.command) as run:
            {"audit": audit, "download": download}[args.command](root, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
