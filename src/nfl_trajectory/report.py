"""A deterministic, explicitly synthetic demonstration of the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nfl_trajectory.motion import KEYS, constant_velocity, trajectory_metrics
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, fingerprint, stage


def synthetic_play() -> tuple[pd.DataFrame, pd.DataFrame]:
    past, future = [], []
    for player, lateral in [(1, 20), (2, 25)]:
        for frame in range(1, 11):
            past.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 30 + 0.5 * frame,
                    "y": lateral + 0.1 * frame,
                }
            )
        for frame in range(1, 21):
            t = frame / 10
            future.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 35 + 5 * t - 0.6 * t**2,
                    "y": lateral + 1 + t + (0.6 if player == 1 else -0.7) * t**2,
                }
            )
    return pd.DataFrame(past), pd.DataFrame(future)


def demo(root: Path, run: Run) -> None:
    destination = root / "artifacts" / "demo"
    html = destination / "trajectory.html"
    svg = destination / "trajectory.svg"
    metrics_path = destination / "metrics.json"

    def action() -> None:
        past, truth = synthetic_play()
        predictions = constant_velocity(past, truth[KEYS])
        metrics = trajectory_metrics(truth, predictions)
        atomic_json(metrics_path, {"data_kind": "synthetic_demonstration", "metrics": metrics})
        figure = go.Figure()
        for player, color in [(1, "#2bd9b1"), (2, "#ffb66b")]:
            for data, label, dash in [
                (past, "Observed", "dot"),
                (truth, "Actual", "solid"),
                (predictions, "Constant velocity", "dash"),
            ]:
                subset = data[data.nfl_id == player]
                figure.add_trace(
                    go.Scatter(
                        x=subset.x,
                        y=subset.y,
                        mode="lines+markers",
                        name=f"Player {player} · {label}",
                        line={"color": color, "dash": dash, "width": 3},
                        marker={"size": 4},
                        hovertemplate=(
                            "x=%{x:.2f} yd<br>y=%{y:.2f} yd<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
        for yard in range(0, 121, 10):
            figure.add_shape(
                type="line",
                x0=yard,
                x1=yard,
                y0=0,
                y1=53.3,
                line={"color": "rgba(255,255,255,0.12)", "width": 1},
            )
        figure.update_layout(
            title={
                "text": "NFL trajectory lab | Synthetic demonstration<br>"
                "<sup>Observed history → forecast → actual path. No NFL score claimed.</sup>"
            },
            template="plotly_dark",
            paper_bgcolor="#101927",
            plot_bgcolor="#163e38",
            xaxis={"title": "Field length (yards)", "range": [25, 50]},
            yaxis={"title": "Field width (yards)", "range": [15, 35], "scaleanchor": "x"},
            height=720,
            margin={"l": 70, "r": 40, "t": 100, "b": 70},
            legend={"orientation": "h", "y": -0.15},
        )
        # Inline JavaScript makes the report viewable without an internet connection.
        atomic_bytes(html, figure.to_html(include_plotlyjs=True, full_html=True).encode())
        # A compact SVG provides a static notebook preview alongside the interactive report.
        pieces = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="510" '
            'viewBox="0 0 960 510" role="img" aria-label="Synthetic player trajectories">',
            '<rect width="960" height="510" fill="#101927"/>',
            '<text x="35" y="40" fill="#ffffff" font-family="sans-serif" font-size="23">'
            "NFL trajectory lab · Synthetic demonstration</text>",
            '<text x="35" y="68" fill="#b5c8db" font-family="sans-serif" font-size="14">'
            "Dotted: observed history   Solid: actual future   Dashed: constant velocity</text>",
            '<rect x="60" y="90" width="840" height="340" fill="#163e38"/>',
        ]
        for yard in range(25, 51, 5):
            sx = 60 + (yard - 25) * 33.6
            pieces.append(f'<path d="M {sx} 90 V 430" stroke="#41615a"/>')
            pieces.append(
                f'<text x="{sx}" y="453" fill="#b5c8db" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13">{yard}</text>'
            )
        for player, color in [(1, "#2bd9b1"), (2, "#ffb66b")]:
            for data, dash in [(past, "2 5"), (truth, "none"), (predictions, "9 6")]:
                subset = data[data.nfl_id == player]
                points = " ".join(
                    f"{60 + (row.x - 25) * 33.6:.2f},{430 - (row.y - 15) * 17:.2f}"
                    for row in subset.itertuples()
                )
                pieces.append(
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    f'stroke-width="3" stroke-dasharray="{dash}"/>'
                )
        pieces += [
            '<text x="480" y="485" fill="#b5c8db" text-anchor="middle" '
            'font-family="sans-serif" font-size="14">Field length (yards) · '
            "Synthetic motion, no NFL result claimed</text>",
            "</svg>",
        ]
        atomic_bytes(svg, "".join(pieces).encode())
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError("Invalid demonstration metrics.")

    stage(root, "demo", fingerprint(root, [], {"demo": 1}), [html, svg, metrics_path], action, run)
    run.event(
        "demo_ready",
        report=str(html.relative_to(root)),
        metrics=json.loads(metrics_path.read_text())["metrics"],
    )
