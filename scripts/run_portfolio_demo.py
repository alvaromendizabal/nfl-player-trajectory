"""Run the dependency-free public demonstration without the research CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo_output"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "src" / "nfl_trajectory" / "portfolio_demo.py"
    spec = importlib.util.spec_from_file_location("_nfl_portfolio_demo", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Public demo module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        receipt = module.run_demo(args.output, args.seed)
    except (OSError, ValueError) as exc:
        print(f"DEMO STOPPED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"Open {args.output / 'dashboard.html'} in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
