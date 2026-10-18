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
from html import escape
from itertools import combinations
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
    require(all(math.isfinite(value) and value >= 0 for value in errors), "Invalid squared error")
    total = math.fsum(errors)
    return {
        "rows": len(errors),
        "sse_xy": total,
        "coordinate_rmse_yards": math.sqrt(total / (2 * len(errors))),
    }


def paired_comparison(
    reference: dict[tuple[Any, ...], float], candidate: dict[tuple[Any, ...], float]
) -> dict[str, Any]:
    """Compare the same rows, retaining game-level sensitivity and pooled weighting."""
    require(bool(reference) and set(reference) == set(candidate), "Paired row coverage mismatch")
    games = sorted({key[0] for key in reference})
    require(len(games) >= 2, "Game sensitivity requires at least two games")

    def compare(keys: list[tuple[Any, ...]]) -> dict[str, Any]:
        left = pooled([reference[key] for key in keys])
        right = pooled([candidate[key] for key in keys])
        return {
            "rows": len(keys),
            "reference_rmse_yards": left["coordinate_rmse_yards"],
            "candidate_rmse_yards": right["coordinate_rmse_yards"],
            "rmse_delta_yards": right["coordinate_rmse_yards"] - left["coordinate_rmse_yards"],
            "sse_delta_xy": right["sse_xy"] - left["sse_xy"],
        }

    by_game = {str(game): compare([key for key in reference if key[0] == game]) for game in games}
    return {
        "direction": "candidate minus reference; negative RMSE delta favors candidate",
        "pooled": compare(list(reference)),
        "by_game": by_game,
        "games_improved": sum(row["sse_delta_xy"] < 0 for row in by_game.values()),
        "games_worsened": sum(row["sse_delta_xy"] > 0 for row in by_game.values()),
        "games_tied": sum(row["sse_delta_xy"] == 0 for row in by_game.values()),
        "leave_one_game_out": {
            str(game): compare([key for key in reference if key[0] != game]) for game in games
        },
        "scope": "Descriptive sensitivity only; no model selection or significance claim",
    }


def trajectory_slices(errors: dict[tuple[Any, ...], float]) -> list[dict[str, Any]]:
    """Keep each player/play together and distinguish displacement from coordinate RMSE."""
    result = []
    for entity in sorted({key[:3] for key in errors}):
        selected = {key: value for key, value in errors.items() if key[:3] == entity}
        displacement = [math.sqrt(value) for value in selected.values()]
        result.append(
            {
                **dict(zip(KEYS[:3], entity, strict=True)),
                **pooled(list(selected.values())),
                "ade_yards": math.fsum(displacement) / len(displacement),
                "last_frame": max(key[-1] for key in selected),
                "fde_yards": math.sqrt(selected[max(selected, key=lambda key: key[-1])]),
                "maximum_displacement_yards": max(displacement),
            }
        )
    return sorted(result, key=lambda item: (-item["sse_xy"], *(item[key] for key in KEYS[:3])))


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
    heldout_errors = {}
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
        heldout_errors[model] = heldout
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
            "holdout_trajectories": trajectory_slices(heldout),
        }
    sources = ("labels.csv", "predictions.csv", "splits.csv", "metrics.json", "manifest.json")
    return {
        "schema_version": 2,
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
        "paired_comparisons": [
            {
                "reference": reference,
                "candidate": candidate,
                **paired_comparison(heldout_errors[reference], heldout_errors[candidate]),
            }
            for reference, candidate in combinations(sorted(models), 2)
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    """Render exact audited counts and errors without JavaScript or remote dependencies."""
    cards, horizons, tracks, paired = [], [], [], []
    for name, model in report["models"].items():
        label = escape(name.replace("_", " "))
        heldout = model["partitions"]["holdout"]
        cards.append(
            f'<article><p class="eyebrow">{label}</p>'
            f"<strong>{heldout['coordinate_rmse_yards']:.4f}<small> yd RMSE</small></strong>"
            f"<p>{heldout['rows']} forecast rows · "
            f"{len(model['holdout_trajectories'])} player/play trajectories</p></article>"
        )
        for horizon, values in model["holdout_horizon_slices"].items():
            horizons.append(
                f"<tr><th>{label}</th><td>{escape(horizon.replace('_', ' '))}</td>"
                f"<td>{values['rows']}</td><td>{values['coordinate_rmse_yards']:.6f}</td></tr>"
            )
        for row in model["holdout_trajectories"]:
            tracks.append(
                f"<tr><th>{label}</th><td>{row['game_id']} / {row['play_id']} / "
                f"{row['nfl_id']}</td><td>{row['rows']}</td>"
                f"<td>{row['coordinate_rmse_yards']:.6f}</td><td>{row['ade_yards']:.6f}</td>"
                f"<td>{row['fde_yards']:.6f}</td></tr>"
            )
    for comparison in report["paired_comparisons"]:
        for game, values in comparison["by_game"].items():
            sensitivity = comparison["leave_one_game_out"][game]
            paired.append(
                f"<tr><th>{escape(comparison['candidate'])}</th>"
                f"<td>{escape(comparison['reference'])}</td><td>{escape(game)}</td>"
                f"<td>{values['rmse_delta_yards']:+.6f}</td>"
                f"<td>{sensitivity['rmse_delta_yards']:+.6f}</td></tr>"
            )
    return (
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Route Lab · Forecast diagnostics</title><style>
:root{color-scheme:light;font-family:Inter,system-ui,sans-serif;color:#13283e;background:#f4f7fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:48px 24px}
header{border-top:5px solid #167461;padding:28px 0}h1{font-size:clamp(32px,5vw,52px);margin:12px 0}
h2{font-size:24px;margin:36px 0 12px}p{line-height:1.65;max-width:86ch}.eyebrow{font-size:12px;
letter-spacing:.08em;text-transform:uppercase;font-weight:750;color:#246957}.cards{display:grid;
grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px}article{background:white;
padding:24px;border:1px solid #dbe5eb;border-radius:12px}strong{font-size:34px}small{font-size:15px;
font-weight:500}.table{overflow:auto;background:white;border:1px solid #dbe5eb;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{text-align:left;
padding:13px 16px;border-bottom:1px solid #e4ebef}thead{background:#e7f1ed}td{font-variant-numeric:
tabular-nums}th{font-weight:650}footer{border-top:1px solid #cddce5;
margin-top:40px;padding-top:16px;
font-size:13px;color:#476174}code{font-size:13px}a{color:#146750}</style></head><body><main>
<header><p class="eyebrow">Route Lab / Independent evaluation</p>
<h1>Account for every forecast.</h1><p>These diagnostics recompute errors from the exported
CSV files. Every model is evaluated on the same held-out keys. All data are <b>synthetic</b>;
this report does not measure the private NFL ensemble.</p></header><section class="cards">"""
        + "".join(cards)
        + """</section>
<h2>Does the error grow with the horizon?</h2><p>Coordinate RMSE pools both coordinate errors
before taking the square root. Each horizon reports its own denominator.</p><div class="table">
<table><thead><tr><th>Forecast</th><th>Horizon</th><th>Rows</th><th>RMSE (yards)</th></tr></thead>
<tbody>"""
        + "".join(horizons)
        + """</tbody></table></div>
<h2>Does either game drive the comparison?</h2><p>Deltas are candidate minus reference;
negative values favor the candidate. Removing one game recomputes RMSE from the remaining
rows. Two invented games support a sensitivity check, not a confidence interval.</p>
<div class="table"><table><thead><tr><th>Candidate</th><th>Reference</th><th>Game</th>
<th>Game RMSE delta</th><th>RMSE delta without game</th></tr></thead><tbody>"""
        + "".join(paired)
        + """</tbody></table></div>
<h2>Which trajectories need inspection?</h2><p>All held-out trajectories, sorted within each
model by total squared error. ADE is mean Euclidean distance; FDE is Euclidean distance at
that trajectory's last requested frame. They are different from coordinate RMSE.</p>
<div class="table"><table><thead><tr><th>Forecast</th><th>Game / play / player</th><th>Rows</th>
<th>RMSE (yd)</th><th>ADE (yd)</th><th>FDE (yd)</th></tr></thead><tbody>"""
        + "".join(tracks)
        + """</tbody></table></div>
<footer>Machine-readable results include full-precision metrics, paired counts, and SHA-256
identities for all audited inputs. CSV coordinates are rounded to six decimals; the
independent audit enforces the documented rounding tolerance.</footer></main></body></html>\n"""
    )


def write_once(path: Path, payload: str) -> None:
    require(not path.is_symlink(), "Output must not be a symbolic link")
    if path.exists():
        require(path.read_text() == payload, "Refusing to overwrite a different audit")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path, help="Optional standalone diagnostic report")
    args = parser.parse_args()
    result = audit(args.demo_output)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_once(args.output, payload)
    if args.html is not None:
        write_once(args.html, render_report(result))
    print(json.dumps({"status": "passed", "models": len(result["models"]), "synthetic_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
