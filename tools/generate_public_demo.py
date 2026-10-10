"""Derive the browser fixture from the unchanged public, synthetic Python generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/nfl_trajectory/portfolio_demo.py"


def build() -> bytes:
    spec = importlib.util.spec_from_file_location("public_trajectory_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    data = module.synthetic_data(2026)
    fixture = {
        "schema_version": 1,
        "evidence_type": "SYNTHETIC_ONLY",
        "seed": 2026,
        "sampling_hz": 10,
        "observed_frames": 8,
        "future_frames": 12,
        "units": "yards",
        "splits": data.splits,
        "observations": data.observations,
        "requests": data.requests,
        "labels": data.labels,
        "provenance": {
            "author": "Alvaro Mendizabal",
            "source": "src/nfl_trajectory/portfolio_demo.py",
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "generator": "synthetic_data(seed=2026)",
            "scope": (
                "Authored simulated trajectories; no NFL tracking rows or private model assets."
            ),
            "generation": (
                "Six synthetic games, two plays per game, three players per play; curved motion."
            ),
            "future_frame_convention": "Relative frame 1 is 0.1 seconds after observed frame 8.",
            "training_fits": 0,
            "official_evaluation": False,
            "private_assets_used": False,
        },
    }
    encoded = json.dumps(fixture, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (
        "// Deterministically derived synthetic fixture. See tools/generate_public_demo.py.\n"
        "globalThis.NFL_TRAJECTORY_DATA = JSON.parse(\n  "
        + json.dumps(encoded, ensure_ascii=False)
        + "\n);\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify fixture contents without writing"
    )
    args = parser.parse_args()
    destination = ROOT / "public-demo/data.js"
    content = build()
    if args.check:
        # Formatters may change whitespace. Compare the actual JSON fixture.
        def payload(source: str) -> object:
            literal = (
                source.split("JSON.parse(", 1)[1]
                .rsplit(")", 1)[0]
                .strip()
                .removesuffix(",")
                .strip()
            )
            return json.loads(json.loads(literal))

        expected = payload(content.decode())
        observed = payload(destination.read_text())
        if observed != expected:
            raise SystemExit("Fixture differs from the unchanged public Python generator")
        print("PASS: browser fixture matches public Python source, seed and provenance")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        print(f"Wrote {destination.relative_to(ROOT)} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
