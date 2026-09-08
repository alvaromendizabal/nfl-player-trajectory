"""Explain completed experiments without refitting or changing numerical checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_experiment import numerical_sources
from nfl_trajectory.runtime import sha256

METRIC = "coordinate_rmse_yards"
LABELS = {
    "constant_velocity": "Constant velocity",
    "role_ridge": "Role-conditioned baseline",
    "motion_ridge": "Motion residual",
    "landing_ridge": "Landing-aware residual",
    "interaction_ridge": "Interaction residual",
}


def validate_summary(summary: dict[str, Any]) -> None:
    """Reject ambiguous, incomplete, or internally inconsistent development evidence."""
    if (summary.get("status"), summary.get("split"), summary.get("screening_split")) != (
        "passed",
        "validation",
        "train",
    ) or summary.get("holdout_evaluation") != "not_run":
        raise ValueError("A completed training-screened validation experiment is required.")
    models = summary["models"]
    names = [m["model"] for m in models]
    scores = np.asarray([m[METRIC] for m in models], dtype=float)
    if len(names) != len(set(names)) or "role_ridge" not in names:
        raise ValueError("Model names must be unique and include the preserved baseline.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("All model scores must be finite and nonnegative.")
    if summary.get("selected_model") != names[int(np.argmin(scores))]:
        raise ValueError("Selected model disagrees with the recorded selection metric.")
    for name in names:
        for dimension in ("role", "forecast_second"):
            rows = [
                s for s in summary["slices"] if s["model"] == name and s["dimension"] == dimension
            ]
            if not rows or len({s["value"] for s in rows}) != len(rows):
                raise ValueError("Evaluation slices must be nonempty and uniquely named.")
            counts = np.asarray([s["rows"] for s in rows], dtype=float)
            values = np.asarray([s[METRIC] for s in rows], dtype=float)
            if (
                not np.isfinite(counts).all()
                or (counts <= 0).any()
                or not np.equal(counts, np.floor(counts)).all()
                or not np.isfinite(values).all()
                or (values < 0).any()
            ):
                raise ValueError("Slice counts and scores must be valid measurements.")
            if int(counts.sum()) != summary["validation_rows_per_model"]:
                raise ValueError("Slices must account for the full validation population.")
            pooled = float(np.sqrt(np.sum(counts * values**2) / counts.sum()))
            recorded = models[names.index(name)][METRIC]
            if not np.isclose(pooled, recorded, atol=1e-10, rtol=1e-10):
                raise ValueError("Slice squared errors do not reconcile with the pooled RMSE.")


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use local completed data when present; label the checksum-verified public snapshot."""
    local = root / "artifacts/features/summary.json"
    path = local if local.is_file() else root / "docs/results/feature_summary.json"
    if (root / "artifacts/features").exists() and not local.is_file():
        raise ValueError("Local feature experiment is incomplete; finish it before publication.")
    summary = json.loads(path.read_text())
    validate_summary(summary)
    if summary.get("source_sha256") != numerical_sources():
        raise ValueError("Feature evidence was produced by different numerical source code.")
    baseline = root / (
        "artifacts/benchmark/model.json" if path == local else "docs/results/model.json"
    )
    if sha256(baseline) != summary.get("baseline_sha256"):
        raise ValueError("Feature evidence refers to a different baseline model.")
    audit_path = root / "docs/results/feature_analysis.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    if audit.get("summary_sha256") != sha256(path):
        if path != local:
            raise ValueError("Published analysis and experiment summary checksums disagree.")
        audit = {}
    label = "Completed local experiment" if path == local else "Verified published experiment"
    return summary, audit, label


def comparison(summary: dict[str, Any]) -> pd.DataFrame:
    """Coordinate RMSE is authoritative; auxiliary metrics retain their weighting labels."""
    validate_summary(summary)
    result = pd.DataFrame(summary["models"]).sort_values(METRIC).reset_index(drop=True)
    baseline = float(result.loc[result.model.eq("role_ridge"), METRIC].iloc[0])
    result["rmse_reduction_vs_baseline_percent"] = (
        100 * (1 - result[METRIC] / baseline) if baseline > 0 else np.nan
    )
    return result


def error_budget(summary: dict[str, Any], dimension: str, model: str | None = None) -> pd.DataFrame:
    """Decompose squared error, not an unweighted average of slice RMSEs."""
    validate_summary(summary)
    if dimension not in ("role", "forecast_second"):
        raise ValueError("Choose role or forecast_second.")
    model = summary["selected_model"] if model is None else model
    rows = [s for s in summary["slices"] if s["model"] == model and s["dimension"] == dimension]
    if not rows:
        raise ValueError("Model is not represented in the experiment.")
    frame = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)
    frame["squared_error_yards2"] = 2 * frame.rows * frame[METRIC] ** 2
    frame["frame_share_percent"] = 100 * frame.rows / frame.rows.sum()
    total = float(frame.squared_error_yards2.sum())
    frame["squared_error_share_percent"] = (
        100 * frame.squared_error_yards2 / total if total > 0 else 0.0
    )
    return frame


def experiment_figure(summary: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    rows = comparison(summary)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    y = np.arange(len(rows))
    ax.errorbar(
        rows[METRIC],
        y,
        xerr=np.maximum(
            np.stack((rows[METRIC] - rows.rmse_ci95_low, rows.rmse_ci95_high - rows[METRIC])), 0
        ),
        fmt="o",
        capsize=5,
        markersize=7,
    )
    ax.set_yticks(y, [LABELS.get(n, n) for n in rows.model])
    ax.invert_yaxis()
    ax.set_xlabel("Coordinate RMSE (yards) · lower is better")
    ax.set_title("Landing-aware features improve the measured baseline", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    for position, row in rows.iterrows():
        ax.annotate(
            f"{row[METRIC]:.4f}",
            (row.rmse_ci95_high, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_xlim(0.75 * rows.rmse_ci95_low.min(), 1.12 * rows.rmse_ci95_high.max())
    fig.supxlabel(
        "95% game-cluster intervals. Development validation only; holdout unscored.", fontsize=9
    )
    return fig


def budget_figure(summary: dict[str, Any], dimension: str) -> Any:
    import matplotlib.pyplot as plt

    frame = error_budget(summary, dimension)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    x = np.arange(len(frame))
    ax.bar(x - 0.18, frame.frame_share_percent, 0.36, label="Share of frames")
    ax.bar(x + 0.18, frame.squared_error_share_percent, 0.36, label="Share of squared error")
    labels = frame.value.tolist()
    if dimension == "forecast_second":
        labels = [
            f"({int(v) - 1}, {v}] seconds\n{n:,} frames"
            for v, n in zip(frame.value, frame.rows, strict=True)
        ]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent of validation total")
    ax.set_ylim(0, 100)
    ax.set_title("Where the remaining prediction error is concentrated", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def association_figure(summary: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(summary["top_training_associations"]).head(10)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.barh(np.arange(len(rows)), rows.training_residual_correlation)
    ax.set_yticks(np.arange(len(rows)), rows.feature)
    ax.invert_yaxis()
    ax.set_xlabel("Maximum absolute training correlation with x/y baseline residual")
    ax.set_title("Training-only screening: hypotheses, not causal importance", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def figure_png(figure: Any) -> bytes:
    """Embed a portable, fully rendered figure in GitHub-visible notebook output."""
    import io

    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(figure)
