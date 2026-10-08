"""Dependency-free synthetic demonstration; not the private competition model."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

KEYS = ("game_id", "play_id", "nfl_id", "frame_id")
MODELS = ("constant_velocity", "hold_last_position")
Row = dict[str, int | float]
Key = tuple[int, int, int, int]
Entity = tuple[int, int, int]


@dataclass(frozen=True)
class SyntheticData:
    """Observations and labels are separate; requests contain identifiers only."""

    observations: list[Row]
    requests: list[Row]
    labels: list[Row]
    splits: dict[int, str]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Coordinates must be finite numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Coordinates must be finite numbers")
    return result


def _key(row: Mapping[str, object]) -> Key:
    try:
        values = tuple(_integer(row[name], name) for name in KEYS)
    except KeyError as exc:
        raise ValueError(f"Missing identifier: {exc.args[0]}") from None
    return values[0], values[1], values[2], values[3]


def predict_observed(
    observations: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    method: str = "constant_velocity",
) -> list[Row]:
    """Forecast relative frames using observed positions only, preserving request order.

    Observation frames start at one. Request frame one is the first future frame
    after the last observation, with the same 10 Hz sampling interval. Extra
    request columns, including any attached target x/y, are deliberately unread.
    """
    if method not in MODELS:
        raise ValueError("Unknown reference method")
    if not observations or not requests:
        raise ValueError("Nonempty observations and requests are required")
    history: dict[Entity, list[tuple[int, float, float]]] = defaultdict(list)
    seen: set[Key] = set()
    for row in observations:
        key = _key(row)
        if key in seen:
            raise ValueError("Duplicate observation key")
        seen.add(key)
        if "x" not in row or "y" not in row:
            raise ValueError("Observed x/y coordinates are required")
        history[key[:3]].append((key[3], _coordinate(row["x"]), _coordinate(row["y"])))
    for records in history.values():
        records.sort()
        if len(records) < 2 or [row[0] for row in records] != list(range(1, len(records) + 1)):
            raise ValueError("Each player needs at least two contiguous observation frames")
    result: list[Row] = []
    seen.clear()
    for row in requests:
        key = _key(row)
        if key in seen:
            raise ValueError("Duplicate request key")
        seen.add(key)
        if key[:3] not in history:
            raise ValueError("Requested player has no observed history")
        if key[3] > 120:
            raise ValueError("Demonstration horizon exceeds 120 frames")
        previous, last = history[key[:3]][-2:]
        dx = last[1] - previous[1] if method == "constant_velocity" else 0.0
        dy = last[2] - previous[2] if method == "constant_velocity" else 0.0
        prediction: Row = dict(zip(KEYS, key, strict=True))
        prediction.update(
            x=round(_coordinate(last[1] + key[3] * dx), 6),
            y=round(_coordinate(last[2] + key[3] * dy), 6),
        )
        result.append(prediction)
    return result


def coordinate_rmse(
    predictions: Sequence[Mapping[str, object]], labels: Sequence[Mapping[str, object]]
) -> float:
    """sqrt(sum((pred_x-x)^2 + (pred_y-y)^2) / (2 * number_of_rows))."""
    if not predictions or not labels:
        raise ValueError("The metric requires a nonempty prediction/label population")
    truth: dict[Key, tuple[float, float]] = {}
    for row in labels:
        key = _key(row)
        if key in truth:
            raise ValueError("Duplicate label key")
        truth[key] = _coordinate(row["x"]), _coordinate(row["y"])
    errors: list[float] = []
    seen: set[Key] = set()
    for row in predictions:
        key = _key(row)
        if key in seen or key not in truth:
            raise ValueError("Prediction keys must match labels exactly once")
        seen.add(key)
        x, y = truth[key]
        errors.extend([(_coordinate(row["x"]) - x) ** 2, (_coordinate(row["y"]) - y) ** 2])
    if seen != set(truth):
        raise ValueError("Prediction keys must cover every label")
    return math.sqrt(math.fsum(errors) / (2 * len(predictions)))


def synthetic_data(seed: int = 2026) -> SyntheticData:
    """Generate curved synthetic motion without consuming global random state."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer in 0..2**32-1")
    rng = random.Random(seed)
    observations: list[Row] = []
    requests: list[Row] = []
    labels: list[Row] = []
    splits = {
        101: "train",
        102: "train",
        103: "train",
        104: "validation",
        105: "holdout",
        106: "holdout",
    }
    for game in splits:
        for play in (1, 2):
            for player in (1, 2, 3):
                x0 = 39.0 + player * 3 + rng.uniform(-0.6, 0.6)
                y0 = 14.0 + player * 6 + rng.uniform(-0.6, 0.6)
                vx = rng.uniform(2.5, 5.0)
                vy = rng.uniform(-1.5, 1.5)
                ax = rng.uniform(-1.1, 1.1)
                ay = rng.uniform(-1.3, 1.3)
                for frame in range(1, 21):
                    time_s = (frame - 8) / 10
                    row: Row = {
                        "game_id": game,
                        "play_id": play,
                        "nfl_id": player,
                        "frame_id": frame if frame <= 8 else frame - 8,
                        "x": round(x0 + vx * time_s + 0.5 * ax * time_s**2, 6),
                        "y": round(y0 + vy * time_s + 0.5 * ay * time_s**2, 6),
                    }
                    if frame <= 8:
                        observations.append(row)
                    else:
                        labels.append(row)
                        requests.append({name: row[name] for name in KEYS})
    return SyntheticData(observations, requests, labels, splits)


def _csv(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row[name] for name in columns})
    return stream.getvalue().encode()


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _svg(data: SyntheticData, predictions: dict[str, list[Row]]) -> str:
    """An accessible inline SVG: three independent player paths from one heldout play."""
    colors = {"truth": "#117f73", "constant_velocity": "#d9861d", "hold_last_position": "#b45665"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 390" role="img" '
        'aria-labelledby="trajectory-title trajectory-description">',
        '<title id="trajectory-title">Synthetic heldout trajectories</title>',
        '<desc id="trajectory-description">Three players: observed motion in slate, '
        "future truth in teal, constant velocity in amber, and a stationary reference in rose. "
        "All paths are simulated, not NFL tracking data.</desc>",
        '<rect width="1080" height="390" rx="16" fill="#f6f9fa"/>',
    ]
    for player in (1, 2, 3):
        selected = lambda row, player=player: (  # noqa: E731
            row["game_id"] == 106 and row["play_id"] == 1 and row["nfl_id"] == player
        )
        observed = [row for row in data.observations if selected(row)]
        truth = [row for row in data.labels if selected(row)]
        velocity = [row for row in predictions["constant_velocity"] if selected(row)]
        hold = [row for row in predictions["hold_last_position"] if selected(row)]
        paths = observed + truth + velocity
        xmin = math.floor(min(row["x"] for row in paths)) - 1
        xmax = math.ceil(max(row["x"] for row in paths)) + 1
        ymin = math.floor(min(row["y"] for row in paths)) - 1
        ymax = math.ceil(max(row["y"] for row in paths)) + 1
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        scale = min(280 / (xmax - xmin), 235 / (ymax - ymin))
        left = (player - 1) * 360

        def point(
            row: Mapping[str, object],
            panel_left: int = left,
            midpoint_x: float = center_x,
            midpoint_y: float = center_y,
            pixels_per_yard: float = scale,
        ) -> tuple[float, float]:
            return (
                panel_left + 180 + (_coordinate(row["x"]) - midpoint_x) * pixels_per_yard,
                180 - (_coordinate(row["y"]) - midpoint_y) * pixels_per_yard,
            )

        parts.append(
            f'<text x="{left + 28}" y="32" font-family="sans-serif" '
            f'font-size="17" fill="#183a45">Player {player}</text>'
        )
        for x in range(xmin, xmax + 1, 2):
            px, _ = point({"x": x, "y": center_y})
            parts.append(f'<path d="M{px:.2f} 58 V310" stroke="#dce6e9" stroke-width="1"/>')
            parts.append(
                f'<text x="{px:.2f}" y="332" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11" fill="#536c76">{x}</text>'
            )
        for y in range(ymin, ymax + 1):
            _, py = point({"x": center_x, "y": y})
            if 58 <= py <= 310:
                parts.append(
                    f'<path d="M{left + 35} {py:.2f} H{left + 325}" '
                    'stroke="#e5edef" stroke-width="1"/>'
                )
                parts.append(
                    f'<text x="{left + 24}" y="{py + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="11" fill="#536c76">{y}</text>'
                )
        for rows, color, dashed in (
            (observed, "#526875", False),
            ([observed[-1], *truth], colors["truth"], False),
            ([observed[-1], *velocity], colors["constant_velocity"], True),
        ):
            coordinates = " ".join(f"{point(row)[0]:.2f},{point(row)[1]:.2f}" for row in rows)
            dash = ' stroke-dasharray="7 5"' if dashed else ""
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                f'stroke-width="3" stroke-linejoin="round"{dash}/>'
            )
        endpoint_x, endpoint_y = point(hold[-1])
        parts.append(
            f'<circle cx="{endpoint_x:.2f}" cy="{endpoint_y:.2f}" r="5" '
            f'fill="{colors["hold_last_position"]}" '
            'stroke="white" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{left + 180}" y="360" text-anchor="middle" font-family="sans-serif" '
            'font-size="12" fill="#536c76">x position (yards) · equal x/y scale</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _dashboard(svg: str, velocity_rmse: float, hold_rmse: float, rows: int) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trajectory forecast · synthetic public demo</title>
<style>
:root{{color-scheme:light;--ink:#173742;--muted:#546d77;--green:#117f73}}
*{{box-sizing:border-box}}
body{{margin:0;background:#eef4f5;color:var(--ink);font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:36px 28px 52px}}
header{{background:#173742;color:#fff;border-radius:20px;padding:36px}}
.eyebrow{{font-size:12px;font-weight:700;letter-spacing:.12em;
text-transform:uppercase;color:#91ddd0}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.2;margin:12px 0}}
h2{{font-size:21px;margin:0 0 8px}}p{{margin:8px 0}}
header p{{max-width:820px;color:#dce9ed}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:24px 0}}
.card,section{{background:#fff;border:1px solid #d9e5e8;border-radius:16px;padding:24px}}
.card strong{{display:block;font-size:29px;line-height:1.35}}
.card span{{font-size:13px;color:var(--muted)}}section{{margin-top:20px}}
.lead{{color:var(--muted);font-size:14px}}svg{{width:100%;height:auto;margin-top:16px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:10px 0}}
.legend b{{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}ul{{padding-left:20px}}
code{{font-family:ui-monospace,monospace;font-size:13px}}
.note{{border-left:4px solid var(--green);padding:12px 18px;background:#f1f9f7}}
footer{{font-size:13px;color:var(--muted);margin-top:20px}}
@media(max-width:760px){{main{{padding:18px}}header{{padding:24px}}
.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">NFL player trajectory · inspectable public demonstration</div>
<h1>From observed motion to a checked forecast</h1>
<p>Deterministic synthetic trajectories, two transparent motion references,
and game-heldout evaluation.
No NFL data, trained competition weights, GPU, account, network connection,
or external Python packages.</p></header>
<div class="cards"><div class="card"><span>Constant velocity · holdout RMSE</span>
<strong>{velocity_rmse:.4f} yd</strong></div>
<div class="card"><span>Hold last position · holdout RMSE</span>
<strong>{hold_rmse:.4f} yd</strong></div>
<div class="card"><span>Whole games in holdout</span><strong>2 / 6</strong></div>
<div class="card"><span>Aligned holdout forecast rows</span><strong>{rows}</strong></div></div>
<section><h2>One heldout play, three player paths</h2><p class="lead">Synthetic game 106 · play 1 ·
8 observed frames · 1.2 seconds forecast · 10 Hz sampling.
Each panel keeps equal x/y scale.</p>
<div class="legend"><span><b style="background:#526875"></b>Observed</span>
<span><b style="background:#117f73"></b>Future truth</span>
<span><b style="background:#d9861d"></b>Constant velocity</span>
<span><b style="background:#b45665"></b>Last position</span></div>{svg}</section>
<section class="grid"><div><h2>What is checked</h2><ul><li>Each game belongs to one partition.</li>
<li>Only observed coordinates enter the predictor.</li>
<li>Forecast identifiers retain request order.</li>
<li>Missing, duplicate, nonfinite, or misaligned rows fail visibly.</li>
<li>Every output has a reproducible SHA-256 receipt.</li></ul></div>
<div><h2>How to interpret the number</h2>
<p><code>RMSE = sqrt(sum(dx² + dy²) / (2 × forecast rows))</code></p>
<p>The denominator counts both coordinates. Rows are pooled before taking the square root.
The references have no learned parameters: partitioning demonstrates evaluation boundaries,
not a training claim.</p></div></section>
<section><h2>Evidence with a clear boundary</h2><p class="note">These are simulated-data metrics.
They are not the competition model, an NFL benchmark, or a reproduced leaderboard result.</p>
<p>Inspect <code>metrics.json</code>, <code>predictions.csv</code>, <code>observations.csv</code>,
<code>labels.csv</code>, <code>splits.csv</code>, and <code>manifest.json</code> beside this report.
The standalone <code>trajectories.svg</code> is available for inspection or reuse.</p></section>
<footer>Public engineering demonstration · standard library only ·
see docs/REPRODUCIBILITY.md for scope and checks.</footer>
</main></body></html>
"""


def run_demo(output: Path, seed: int = 2026) -> dict[str, object]:
    """Write deterministic artifacts; preserve existing files if content differs."""
    data = synthetic_data(seed)
    predictions = {
        method: predict_observed(data.observations, data.requests, method) for method in MODELS
    }
    request_keys = [_key(row) for row in data.requests]
    for rows in predictions.values():
        if [_key(row) for row in rows] != request_keys:
            raise ValueError("Prediction/request alignment check failed")
    poison = [{**row, "x": "labels must not be read", "y": float("nan")} for row in data.requests]
    if predict_observed(data.observations, poison) != predictions["constant_velocity"]:
        raise ValueError("Target-coordinate independence check failed")
    partitions: dict[str, object] = {}
    holdout_scores: dict[str, float] = {}
    holdout_rows = 0
    for partition in ("train", "validation", "holdout"):
        games = {game for game, split in data.splits.items() if split == partition}
        truth = [row for row in data.labels if row["game_id"] in games]
        scores = {}
        for method, rows in predictions.items():
            score = coordinate_rmse([row for row in rows if row["game_id"] in games], truth)
            scores[method] = round(score, 12)
        partitions[partition] = {
            "games": sorted(games),
            "rows": len(truth),
            "coordinate_rmse_yards": scores,
        }
        if partition == "holdout":
            holdout_scores, holdout_rows = scores, len(truth)
    metrics = {
        "schema_version": 1,
        "synthetic_only": True,
        "seed": seed,
        "training_fits": 0,
        "model_scope": "Observation-only references; no private competition model is loaded.",
        "metric": "sqrt(sum(dx^2 + dy^2) / (2 * rows))",
        "units": "yards",
        "partitions": partitions,
        "checks": {
            "game_disjoint": True,
            "request_order_preserved": True,
            "request_labels_not_read": True,
            "all_predictions_finite": True,
        },
    }
    svg = _svg(data, predictions)
    prediction_rows = [
        {"model": method, **row} for method, rows in predictions.items() for row in rows
    ]
    files = {
        "dashboard.html": _dashboard(
            svg, holdout_scores[MODELS[0]], holdout_scores[MODELS[1]], holdout_rows
        ).encode(),
        "trajectories.svg": (svg + "\n").encode(),
        "metrics.json": _json(metrics),
        "predictions.csv": _csv(prediction_rows, ("model", *KEYS, "x", "y")),
        "observations.csv": _csv(data.observations, (*KEYS, "x", "y")),
        "labels.csv": _csv(data.labels, (*KEYS, "x", "y")),
        "splits.csv": _csv(
            [{"game_id": game, "partition": split} for game, split in data.splits.items()],
            ("game_id", "partition"),
        ),
    }
    manifest = {
        "schema_version": 1,
        "synthetic_only": True,
        "seed": seed,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_scope": "Same source, seed, and Python runtime; no timestamps or absolute paths.",
        "manifest_self_hash_excluded": True,
        "files": [
            {"file": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(files.items())
        ],
    }
    files["manifest.json"] = _json(manifest)
    output = Path(output).absolute()
    for path in (output, *output.parents):
        if path.is_symlink():
            raise ValueError("Output directory must not be a symlink")
    for name, content in files.items():
        target = output / name
        if target.is_symlink() or (
            target.exists() and (not target.is_file() or target.read_bytes() != content)
        ):
            raise ValueError("Output contains different content; choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = output / name
        if not target.exists():
            target.write_bytes(content)
    return {
        "status": "PASS",
        "synthetic_only": True,
        "output": str(output),
        "holdout_rows": holdout_rows,
        "coordinate_rmse_yards": holdout_scores,
        "manifest_sha256": hashlib.sha256(files["manifest.json"]).hexdigest(),
    }
