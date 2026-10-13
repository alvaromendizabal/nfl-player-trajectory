"""Independently audit exported synthetic forecasts by horizon and held-out game.

This reads the public CSV boundary. It does not call the forecasting engine, train
models, inspect private tracking data, or treat synthetic slices as NFL evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("game_id", "play_id", "nfl_id", "frame_id")
ROUNDING_TOLERANCE_YARDS = 1.5e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rows(path: Path, model: bool = False) -> dict[tuple[Any, ...], tuple[float, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = (["model"] if model else []) + list(KEYS) + ["x", "y"]
        require(reader.fieldnames == expected, "Unexpected CSV schema: " + path.name)
        for row in reader:
            require(
                None not in row and all(value is not None for value in row.values()), "Ragged CSV"
            )
            identifiers = tuple(int(row[name]) for name in KEYS)
            require(all(value > 0 for value in identifiers), "Nonpositive forecast identifier")
            require(identifiers[-1] <= 120, "Forecast frame outside public contract")
            key = ((row["model"],) if model else ()) + identifiers
            require(key not in result, "Duplicate forecast key")
            position = (float(row["x"]), float(row["y"]))
            require(
                all(math.isfinite(value) for value in position), "Nonfinite forecast coordinate"
            )
            result[key] = position
    require(bool(result), "Empty forecast CSV")
    return result


def pooled(errors: list[float]) -> dict[str, float | int]:
    require(bool(errors), "Empty metric slice")
    total = math.fsum(errors)
    return {
        "rows": len(errors),
        "sse_xy": total,
        "coordinate_rmse_yards": math.sqrt(total / (2 * len(errors))),
    }


def audit(directory: Path) -> dict[str, Any]:
    # Verify the generating release contract, then independently recalculate from CSV.
    spec = importlib.util.spec_from_file_location(
        "portfolio_release", ROOT / "scripts/validate_portfolio_release.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Public release validator is unavailable")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    validator.validate_demo(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    require(
        metrics["synthetic_only"] is True, "This audit accepts the public synthetic fixture only"
    )
    labels = rows(directory / "labels.csv")
    predictions = rows(directory / "predictions.csv", model=True)
    split: dict[int, str] = {}
    with (directory / "splits.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["game_id", "partition"], "Unexpected partition schema")
        for row in reader:
            game = int(row["game_id"])
            require(game > 0 and game not in split, "Duplicate or invalid game partition")
            require(row["partition"] in {"train", "validation", "holdout"}, "Unknown partition")
            split[game] = row["partition"]
    require(set(split) == {key[0] for key in labels}, "Game partition coverage mismatch")
    models = set(metrics["partitions"]["holdout"]["coordinate_rmse_yards"])
    require({key[0] for key in predictions} == models, "Model coverage mismatch")
    require(
        set(predictions) == {(model, *key) for model in models for key in labels},
        "Prediction/label key mismatch",
    )
    results = {}
    for model in sorted(models):
        errors = {
            key: math.fsum(
                (predictions[(model, *key)][axis] - position[axis]) ** 2 for axis in (0, 1)
            )
            for key, position in labels.items()
        }
        partitions = {}
        for partition in ("train", "validation", "holdout"):
            selected = {key: error for key, error in errors.items() if split[key[0]] == partition}
            summary = pooled(list(selected.values()))
            recorded = metrics["partitions"][partition]
            require(summary["rows"] == recorded["rows"], "Recorded row count mismatch")
            require(
                abs(summary["coordinate_rmse_yards"] - recorded["coordinate_rmse_yards"][model])
                <= ROUNDING_TOLERANCE_YARDS,
                "Exported CSV metric differs beyond rounding tolerance",
            )
            partitions[partition] = summary
        heldout = {key: error for key, error in errors.items() if split[key[0]] == "holdout"}
        horizons = {}
        for start, stop in ((1, 4), (5, 8), (9, 12)):
            selected_errors = [error for key, error in heldout.items() if start <= key[-1] <= stop]
            horizons[f"frames_{start}_{stop}"] = pooled(selected_errors)
        require(
            sum(record["rows"] for record in horizons.values()) == len(heldout),
            "Horizon slices do not cover the entire holdout",
        )
        games = {
            str(game): pooled([error for key, error in heldout.items() if key[0] == game])
            for game in sorted({key[0] for key in heldout})
        }
        results[model] = {
            "partitions": partitions,
            "holdout_horizon_slices": horizons,
            "holdout_game_slices": games,
        }
    sources = ("labels.csv", "predictions.csv", "splits.csv", "metrics.json", "manifest.json")
    return {
        "schema_version": 1,
        "status": "passed",
        "synthetic_only": True,
        "scope": (
            "Independent audit of rounded synthetic CSV exports; not historical ensemble evaluation"
        ),
        "metric": "sqrt(sum(dx^2 + dy^2) / (2 * rows))",
        "units": "yards",
        "rounding_tolerance_yards": ROUNDING_TOLERANCE_YARDS,
        "rounding_note": (
            "CSV coordinates have six decimal places; reference metrics use full precision."
        ),
        "uncertainty_note": (
            "Two synthetic holdout games do not support a generalization claim; "
            "no confidence interval is claimed."
        ),
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in sources
        },
        "models": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.demo_output)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    require(not args.output.is_symlink(), "Output must not be a symbolic link")
    if args.output.exists():
        require(args.output.read_text() == payload, "Refusing to overwrite a different audit")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(json.dumps({"status": "passed", "models": len(result["models"]), "synthetic_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
