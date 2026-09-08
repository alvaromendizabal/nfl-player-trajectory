"""Review figures tied to the same source-verified predictor as the reported score."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from nfl_trajectory.feature_experiment import load_week
from nfl_trajectory.feature_research import verify_inputs
from nfl_trajectory.models import predict
from nfl_trajectory.motion import KEYS, trajectory_metrics
from nfl_trajectory.research_inference import predict_research, research_bundle
from nfl_trajectory.runtime import sha256


@dataclass(frozen=True)
class TrajectoryExample:
    game_id: int
    play_id: int
    selected_model: str
    static_figure: Figure
    interactive_figure: go.Figure
    roles: pd.DataFrame
    metrics: pd.DataFrame


def trajectory_comparison(
    observed: pd.DataFrame,
    truth: pd.DataFrame,
    predicted: pd.DataFrame,
    physical: pd.DataFrame,
    selected_model: str,
) -> TrajectoryExample:
    """Compare up to three players chosen by identifier, independently of their error."""
    request = truth[KEYS]
    fig, ax = plt.subplots(figsize=(8.5, 6.5), layout="constrained")
    curve_rows = []
    example_metrics = []
    cloud = []
    colors = ["#247A89", "#D57838", "#8053A0"]
    for index, player in enumerate(sorted(request.nfl_id.unique())[:3]):
        history = observed.loc[observed.nfl_id.eq(player)].sort_values("frame_id").tail(15)
        actual = truth.loc[truth.nfl_id.eq(player)].sort_values("frame_id")
        estimate = predicted.loc[predicted.nfl_id.eq(player)].sort_values("frame_id")
        baseline_curve = physical.loc[physical.nfl_id.eq(player)].sort_values("frame_id")
        label = chr(ord("A") + index)
        example_metrics.append(
            {
                "Player": label,
                "Physical baseline RMSE": trajectory_metrics(actual, baseline_curve)[
                    "coordinate_rmse_yards"
                ],
                "Engineered RMSE": trajectory_metrics(actual, estimate)["coordinate_rmse_yards"],
            }
        )
        for frame, kind, style in [
            (history, "observed", ":"),
            (actual, "truth", "-"),
            (estimate, "predicted", "--"),
            (baseline_curve, "physical baseline", "-."),
        ]:
            ax.plot(
                frame.x,
                frame.y,
                style,
                color=colors[index],
                linewidth=2,
                label=f"Player {label}: {kind}",
            )
            cloud.append(frame[["x", "y"]].to_numpy())
            curve = frame[["x", "y", "frame_id"]].copy()
            curve["seconds_from_throw"] = (
                (curve.frame_id - history.frame_id.max()) / 10
                if kind == "observed"
                else curve.frame_id / 10
            )
            curve_rows.append(curve.assign(player="Player " + label, curve=kind))
        ax.scatter(history.x.iloc[-1], history.y.iloc[-1], color=colors[index], s=35)
    landing = observed[["ball_land_x", "ball_land_y"]].iloc[0].to_numpy(float)
    ax.scatter(
        landing[0], landing[1], marker="*", s=180, color="#C68A12", label="Supplied landing point"
    )
    points = np.vstack([*cloud, landing[None, :]])
    ax.set_xlim(points[:, 0].min() - 4, points[:, 0].max() + 4)
    ax.set_ylim(points[:, 1].min() - 4, points[:, 1].max() + 4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#F4F8F4")
    ax.grid(alpha=0.25)
    ax.set_xlabel("Field x (yards)")
    ax.set_ylabel("Field y (yards)")
    ax.set_title("Physical baseline and learned correction", fontsize=13)
    legend = [
        Line2D([0], [0], color=color, marker="o", linestyle="none", label=f"Player {chr(65 + i)}")
        for i, color in enumerate(colors[: min(3, request.nfl_id.nunique())])
    ]
    legend.extend(
        Line2D([0], [0], color="#53616B", linestyle=style, linewidth=2, label=label)
        for label, style in [
            ("Observed", ":"),
            ("Truth", "-"),
            ("Prediction", "--"),
            ("Physical baseline", "-."),
        ]
    )
    legend.append(
        Line2D(
            [0],
            [0],
            color="#C68A12",
            marker="*",
            markersize=12,
            linestyle="none",
            label="Supplied landing",
        )
    )
    ax.legend(handles=legend, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    interactive_play = px.line(
        pd.concat(curve_rows, ignore_index=True),
        x="x",
        y="y",
        color="player",
        line_dash="curve",
        hover_data=["seconds_from_throw"],
        title="Explore the same play: motion baseline and engineered correction",
        labels={"x": "Field x (yards)", "y": "Field y (yards)"},
        color_discrete_sequence=colors,
    )
    interactive_play.add_scatter(
        x=[landing[0]],
        y=[landing[1]],
        mode="markers",
        marker={"symbol": "star", "size": 13, "color": "#C68A12"},
        name="Supplied landing point",
    )
    interactive_play.update_yaxes(scaleanchor="x", scaleratio=1)
    roles = pd.DataFrame(
        [
            {
                "Player": chr(65 + i),
                "Role": observed.loc[observed.nfl_id.eq(player), "player_role"].iloc[0],
            }
            for i, player in enumerate(sorted(request.nfl_id.unique())[:3])
        ]
    )
    return TrajectoryExample(
        int(request.game_id.iloc[0]),
        int(request.play_id.iloc[0]),
        selected_model,
        fig,
        interactive_play,
        roles,
        pd.DataFrame(example_metrics),
    )


def load_development_example(root: Path) -> TrajectoryExample | None:
    """Require current raw replay before describing a private-data figure as validated."""
    report_path = root / "artifacts/research/inference/summary.json"
    if not report_path.is_file():
        return None
    fitted = research_bundle(root)
    report = json.loads(report_path.read_text())
    bundle_hash = hashlib.sha256(json.dumps(fitted, sort_keys=True).encode()).hexdigest()
    if (
        report.get("status") != "passed"
        or report.get("selected_model") != fitted["selected_model"]
        or report.get("bundle_sha256") != bundle_hash
        or report.get("validator_sha256") != sha256(root / "scripts/validate_research.py")
    ):
        raise ValueError("The example requires raw validation of the current predictor and source.")
    caches, _ = verify_inputs(root)
    for cache in caches:
        _, requests, arrays = load_week(cache)
        available = requests.loc[requests.game_id.isin(fitted["evaluation_games"])]
        if available.empty:
            continue
        game, play = available[["game_id", "play_id"]].sort_values(["game_id", "play_id"]).iloc[0]
        chosen = requests.game_id.eq(game) & requests.play_id.eq(play)
        request = requests.loc[chosen, KEYS].reset_index(drop=True)
        truth = request.assign(x=arrays["truth"][chosen, 0], y=arrays["truth"][chosen, 1])
        observed = pd.read_csv(root / "data/raw/train" / (cache.parent.name + ".csv"))
        observed = observed.loc[observed.game_id.eq(game) & observed.play_id.eq(play)]
        return trajectory_comparison(
            observed,
            truth,
            predict_research(observed, request, fitted),
            predict(observed, request, "role_ridge", fitted["baseline"]),
            fitted["selected_model"],
        )
    raise ValueError("No raw development example exists for the validated predictor.")
