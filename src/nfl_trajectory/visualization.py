"""Publication figures and an offline, animated field report from verified artifacts."""

from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path
from string import Template
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nfl_trajectory.models import BASIS, predict_from_design
from nfl_trajectory.motion import ENTITY, KEYS
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json

LABELS = {
    "last_position": "Last position",
    "constant_velocity": "Constant velocity",
    "smoothed_velocity": "Smoothed velocity",
    "constant_acceleration": "Constant acceleration",
    "ball_arrival": "Ball arrival",
    "role_ridge": "Role-conditioned ridge",
}
COLORS = {
    "last_position": "#8b9bb0",
    "constant_velocity": "#fbbf77",
    "smoothed_velocity": "#94b8ff",
    "constant_acceleration": "#ef8888",
    "ball_arrival": "#c5a3ff",
    "role_ridge": "#4ee0bd",
}
STYLE = {
    "figure.facecolor": "#101b2b",
    "axes.facecolor": "#101b2b",
    "axes.edgecolor": "#42536c",
    "axes.labelcolor": "#d4dfed",
    "text.color": "#f1f5fc",
    "xtick.color": "#b8c8dc",
    "ytick.color": "#b8c8dc",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "#101b2b",
}


def save_figure(figure: Any, destination: Path) -> None:
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    atomic_bytes(destination, stream.getvalue())
    plt.close(figure)


def aggregate_eda(root: Path) -> dict[str, Any]:
    weeks = [
        json.loads(path.read_text())
        for path in sorted((root / "artifacts/benchmark/weeks").glob("*/eda.json"))
    ]
    if not weeks:
        raise ValueError("No verified weekly EDA artifacts exist.")
    roles: dict[str, int] = {}
    for week in weeks:
        for name, count in week["roles"].items():
            roles[name] = roles.get(name, 0) + count
    return {
        "split": "train",
        "weeks": weeks,
        "roles": roles,
        "horizon_bins_seconds": weeks[0]["horizon_bins_seconds"],
        "horizon_counts": np.sum([x["horizon_counts"] for x in weeks], axis=0).tolist(),
        "speed_bins_yards_per_second": weeks[0]["speed_bins_yards_per_second"],
        "speed_counts": np.sum([x["speed_counts"] for x in weeks], axis=0).tolist(),
        **{
            name: sum(x[name] for x in weeks)
            for name in [
                "games",
                "plays",
                "trajectories",
                "input_rows",
                "target_rows",
                "missing_observed_cells",
            ]
        },
    }


def plot_eda(eda: dict[str, Any], destination: Path) -> None:
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")
        figure.suptitle("Before the forecast · Training data only", fontsize=21, fontweight="bold")
        roles = sorted(eda["roles"], key=eda["roles"].get)
        axes[0].barh(
            [name.replace(" ", "\n", 1) for name in roles],
            [eda["roles"][name] for name in roles],
            color="#4ee0bd",
            height=0.5,
        )
        axes[0].set_title("Who are we predicting?")
        axes[0].set_xlabel("Scored player trajectories")
        for axis, counts, edges, title, label in [
            (
                axes[1],
                eda["horizon_counts"],
                eda["horizon_bins_seconds"],
                "How far into the future?",
                "Forecast horizon (seconds)",
            ),
            (
                axes[2],
                eda["speed_counts"],
                eda["speed_bins_yards_per_second"],
                "How fast at release?",
                "Recent speed (yards/second)",
            ),
        ]:
            labels = [
                f"{left:g}–{right:g}" for left, right in zip(edges[:-1], edges[1:], strict=True)
            ]
            axis.bar(labels, counts, color="#94b8ff", width=0.65)
            axis.set_title(title)
            axis.set_xlabel(label)
            axis.tick_params(axis="x", labelrotation=35)
            axis.set_ylabel("Trajectories")
        save_figure(figure, destination)


def plot_benchmark(summary: dict[str, Any], destination: Path) -> None:
    with plt.rc_context(STYLE):
        figure = plt.figure(figsize=(15, 10), layout="constrained")
        grid = figure.add_gridspec(2, 2, height_ratios=[1.15, 1])
        ranking = figure.add_subplot(grid[0, :])
        horizon = figure.add_subplot(grid[1, 0])
        roles = figure.add_subplot(grid[1, 1])
        figure.suptitle(
            "NFL trajectory benchmark · Temporal validation", fontsize=23, fontweight="bold"
        )
        rows = list(reversed(summary["models"]))
        values = np.array([x["coordinate_rmse_yards"] for x in rows])
        y = np.arange(len(rows))
        ranking.barh(y, values, color=[COLORS[x["model"]] for x in rows], height=0.56, alpha=0.9)
        low = values - np.array([x["rmse_ci95_low"] for x in rows])
        high = np.array([x["rmse_ci95_high"] for x in rows]) - values
        ranking.errorbar(
            values,
            y,
            xerr=[np.maximum(low, 0), np.maximum(high, 0)],
            fmt="none",
            color="#f1f5fc",
            capsize=4,
        )
        ranking.set_yticks(y, [LABELS[x["model"]] for x in rows])
        ranking.set_xlim(0, max(max(values), 0.01) * 1.3)
        ranking.set_xlabel("Coordinate RMSE (yards) · lower is better · 95% game-cluster intervals")
        for row, value in enumerate(values):
            ranking.text(
                value * 1.13, row, f"{value:.4f}", va="center", fontsize=12, fontweight="bold"
            )
        slices = pd.DataFrame(summary["slices"])
        for model in ["constant_velocity", "smoothed_velocity", "role_ridge"]:
            data = slices[(slices.model == model) & (slices.dimension == "forecast_second")].copy()
            data["second"] = data.value.astype(int)
            data = data.sort_values("second")
            horizon.plot(
                data.second,
                data.coordinate_rmse_yards,
                "o-",
                color=COLORS[model],
                label=LABELS[model],
                linewidth=2,
                markersize=5,
            )
        horizon.set_title("Error grows with forecast time")
        horizon.set_xlabel("Forecast time bin ending at (seconds)")
        horizon.set_ylabel("Coordinate RMSE (yards)")
        horizon.legend(fontsize=9, frameon=False)
        categories = sorted(slices.loc[slices.dimension == "role", "value"].unique())
        for offset, model in enumerate(["constant_velocity", "smoothed_velocity", "role_ridge"]):
            data = slices[(slices.model == model) & (slices.dimension == "role")].set_index("value")
            roles.bar(
                np.arange(len(categories)) + (offset - 1) * 0.24,
                [data.loc[name, "coordinate_rmse_yards"] for name in categories],
                width=0.22,
                color=COLORS[model],
                label=LABELS[model],
            )
        roles.set_xticks(
            np.arange(len(categories)), [name.replace(" ", "\n", 1) for name in categories]
        )
        roles.set_ylabel("Coordinate RMSE (yards)")
        roles.set_title("Which roles remain difficult?")
        for axis in [ranking, horizon, roles]:
            axis.grid(axis="x" if axis is ranking else "y", alpha=0.13)
            axis.set_axisbelow(True)
        save_figure(figure, destination)


def plot_coefficients(fitted: dict[str, Any], destination: Path) -> None:
    with plt.rc_context(STYLE):
        names = sorted(fitted["roles"])
        values = np.array([fitted["roles"][name]["coefficients"] for name in names])
        figure, axis = plt.subplots(figsize=(13, 3.6), layout="constrained")
        limit = max(float(np.abs(values).max()), 0.01)
        axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(BASIS)), [name.replace("_", "\n") for name in BASIS])
        axis.set_yticks(range(len(names)), names)
        axis.set_title("Learned vector weights · Training-only fit", fontsize=19, pad=18)
        for row in range(len(names)):
            for column in range(len(BASIS)):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    color="#08111e",
                    fontweight="bold",
                )
        figure.supxlabel(
            "Weights act jointly; correlated basis terms are not causal feature importance.",
            fontsize=10,
        )
        save_figure(figure, destination)


def field_figure(
    observed: pd.DataFrame,
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    roles: pd.DataFrame,
    model: str,
) -> go.Figure:
    figure = go.Figure()
    colors = ["#4ee0bd", "#fbbf77", "#94b8ff", "#e8a0d9", "#fff1a6", "#acdd7c"]
    trajectories = []
    for index, player in enumerate(sorted(truth.nfl_id.unique())):
        color = colors[index % len(colors)]
        role = roles.loc[roles.nfl_id == player, "player_role"].iloc[0]
        history = observed[observed.nfl_id == player].sort_values("frame_id").tail(20)
        actual = truth[truth.nfl_id == player].sort_values("frame_id")
        forecast = prediction[prediction.nfl_id == player].sort_values("frame_id")
        name = f"{role} · {player}"
        figure.add_trace(
            go.Scatter(
                x=history.x,
                y=history.y,
                mode="lines",
                line={"color": color, "width": 2, "dash": "dot"},
                name=name + " / observed",
                legendgroup=str(player),
                opacity=0.5,
            )
        )
        for data, label, dash in [(actual, "actual", "solid"), (forecast, "forecast", "dash")]:
            trace_index = len(figure.data)
            x = [float(history.x.iloc[-1]), *data.x.tolist()]
            y = [float(history.y.iloc[-1]), *data.y.tolist()]
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line={"color": color, "width": 3, "dash": dash},
                    marker={"size": 4},
                    name=name + " / " + label,
                    legendgroup=str(player),
                )
            )
            trajectories.append((trace_index, x, y))
    ball = roles.iloc[0]
    figure.add_trace(
        go.Scatter(
            x=[ball.ball_land_x],
            y=[ball.ball_land_y],
            mode="markers",
            marker={"symbol": "x", "size": 14, "color": "white"},
            name="Supplied ball landing point",
        )
    )
    frames = []
    for frame in range(int(truth.frame_id.max()) + 1):
        frames.append(
            go.Frame(
                name=str(frame),
                data=[go.Scatter(x=x[: frame + 1], y=y[: frame + 1]) for _, x, y in trajectories],
                traces=[trace for trace, _, _ in trajectories],
            )
        )
    figure.frames = frames
    all_x = pd.concat([observed.x, truth.x, prediction.x])
    all_y = pd.concat([observed.y, truth.y, prediction.y])
    for yard in range(0, 121, 5):
        figure.add_shape(
            type="line",
            x0=yard,
            x1=yard,
            y0=0,
            y1=53.3,
            line={"color": "rgba(255,255,255,.15)", "width": 1},
        )
    game, play = int(truth.game_id.iloc[0]), int(truth.play_id.iloc[0])
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101b2b",
        plot_bgcolor="#163e38",
        height=720,
        title={
            "text": f"At the throw · Game {game}, play {play}<br><sup>{LABELS[model]} · "
            "First validation play by ID, selected before examining errors</sup>"
        },
        xaxis={
            "title": "Field length (yards)",
            "range": [float(all_x.min()) - 3, float(all_x.max()) + 3],
        },
        yaxis={
            "title": "Field width (yards)",
            "range": [float(all_y.min()) - 3, float(all_y.max()) + 3],
            "scaleanchor": "x",
        },
        legend={"orientation": "h", "y": -0.24, "font": {"size": 10}},
        margin={"t": 100, "b": 160},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.09,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": False}},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "y": -0.13,
                "currentvalue": {"prefix": "Seconds after throw: "},
                "steps": [
                    {
                        "label": f"{i / 10:.1f}",
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for i in range(len(frames))
                ],
            }
        ],
    )
    return figure


def example_figure(root: Path, summary: dict[str, Any], fitted: dict[str, Any]) -> go.Figure:
    from nfl_trajectory.benchmark import load_design

    for path in sorted((root / "artifacts/benchmark/weeks").glob("*/design.npz")):
        state, features, truth, labels = load_design(path)
        validation = state.loc[labels == "validation"]
        if validation.empty:
            continue
        first = validation.sort_values(KEYS).iloc[0]
        mask = (state.game_id == first.game_id) & (state.play_id == first.play_id)
        target = truth.loc[mask].reset_index(drop=True)
        context = state.loc[mask].reset_index(drop=True)
        predicted = predict_from_design(context, features[mask], summary["selected_model"], fitted)
        observed = pd.read_csv(
            root / "data/raw/train" / f"{path.parent.name}.csv", usecols=KEYS + ["x", "y"]
        )
        observed = observed[
            (observed.game_id == first.game_id)
            & (observed.play_id == first.play_id)
            & observed.nfl_id.isin(target.nfl_id)
        ]
        return field_figure(
            observed, target, predicted, context.drop_duplicates(ENTITY), summary["selected_model"]
        )
    raise ValueError("A validation example is required for the field report.")


def build_report(root: Path, run: Run) -> None:
    destination = root / "artifacts/benchmark"
    summary = json.loads((destination / "summary.json").read_text())
    fitted = json.loads((destination / "model.json").read_text())
    eda = aggregate_eda(root)
    atomic_json(destination / "eda.json", eda)
    plot_eda(eda, destination / "eda.png")
    plot_benchmark(summary, destination / "benchmark.png")
    plot_coefficients(fitted, destination / "coefficients.png")
    animation = example_figure(root, summary, fitted)
    atomic_bytes(
        destination / "trajectory.html",
        animation.to_html(include_plotlyjs=True, full_html=True).encode(),
    )
    rows = "".join(
        "<tr><td>"
        + escape(LABELS[item["model"]])
        + "</td>"
        + "".join(
            f"<td>{item[key]:.4f}</td>"
            for key in [
                "coordinate_rmse_yards",
                "ade_frame_weighted_yards",
                "fde_trajectory_weighted_yards",
                "p95_displacement_yards",
            ]
        )
        + "</tr>"
        for item in summary["models"]
    )
    images = {
        name: base64.b64encode((destination / f"{name}.png").read_bytes()).decode()
        for name in ["eda", "benchmark", "coefficients"]
    }
    best = summary["models"][0]
    content = Template((Path(__file__).parent / "templates/report.html").read_text()).substitute(
        {
            "value0": f"{best['coordinate_rmse_yards']:.4f}",
            "value1": f"{summary['improvement_vs_velocity_percent']:.1f}",
            "value2": f"{summary['validation_games']}",
            "value3": f"{summary['validation_rows_per_model']:,}",
            "value4": f"{escape(LABELS[best['model']])}",
            "value5": f"{images['benchmark']}",
            "value6": f"{rows}",
            "value7": f"{eda['games']}",
            "value8": f"{eda['plays']:,}",
            "value9": f"{eda['trajectories']:,}",
            "value10": f"{images['eda']}",
            "value11": f"{animation.to_html(include_plotlyjs=True, full_html=False)}",
            "value12": f"{images['coefficients']}",
        }
    )
    atomic_bytes(destination / "report.html", content.encode())
    run.event(
        "report_ready",
        path="artifacts/benchmark/report.html",
        field_animation="artifacts/benchmark/trajectory.html",
    )
