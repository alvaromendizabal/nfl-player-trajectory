"""Auditable notebook analysis of completed experiments; never fit or submit silently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_experiment import numerical_sources
from nfl_trajectory.runtime import sha256


def validate_summary(summary: dict[str, Any]) -> None:
    """Reject incomplete evidence and arithmetic-inconsistent validation slices."""
    if (summary.get("status") != "passed" or summary.get("split") != "validation"
            or summary.get("screening_split") != "train"
            or summary.get("holdout_evaluation") != "not_run"):
        raise ValueError("A completed, training-screened validation experiment is required.")
    scores = summary.get("models", [])
    names = [r["model"] for r in scores]
    if not scores or len(names) != len(set(names)):
        raise ValueError("Model names must be nonempty and unique.")
    for row in scores:
        for key in ("coordinate_rmse_yards", "ade_frame_weighted_yards",
                    "fde_trajectory_weighted_yards", "p95_displacement_yards"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Metrics must be finite and nonnegative.")
        for dimension in ("role", "forecast_second"):
            records = [r for r in summary["slices"]
                       if r["model"] == row["model"] and r["dimension"] == dimension]
            rows = sum(r["rows"] for r in records)
            if rows != summary["validation_rows_per_model"] or rows <= 0:
                raise ValueError("Diagnostic slices must cover every scored row.")
            rmse = math.sqrt(sum(r["rows"] * r["coordinate_rmse_yards"] ** 2
                                 for r in records) / rows)
            if not math.isclose(rmse, row["coordinate_rmse_yards"], rel_tol=1e-10):
                raise ValueError("Diagnostic slices disagree with the pooled RMSE.")
    best = min(scores, key=lambda r: r["coordinate_rmse_yards"])["model"]
    if summary.get("selected_model") != best:
        raise ValueError("Selected model is not the measured RMSE winner.")


def selection_study(bundle: dict[str, Any], summary_hash: str, model_hash: str) -> dict[str, Any]:
    """Inspect selected feature membership and standardized coefficients, not causal importance."""
    landing = bundle["models"]["landing_ridge"]
    interaction = bundle["models"]["interaction_ridge"]
    names = landing["features"]
    weights = np.asarray(landing["coefficients"], dtype=float)
    if weights.shape != (len(names), 2) or not np.isfinite(weights).all():
        raise ValueError("Invalid coefficients in completed model.")
    ranked = sorted(range(len(names)), key=lambda i: -float(np.linalg.norm(weights[i])))[:12]
    return {
        "format": 1, "summary_sha256": summary_hash, "feature_model_sha256": model_hash,
        "training_rows": landing["training_rows"], "landing_selected_count": len(names),
        "landing_ball_ux_count": sum("ball_ux" in name for name in names),
        "overlap_count": len(set(names) & set(interaction["features"])),
        "removed_from_landing": sorted(set(names) - set(interaction["features"])),
        "added_by_interaction": sorted(set(interaction["features"]) - set(names)),
        "coefficient_rows": [
            {"feature": names[i], "x_coefficient": float(weights[i, 0]),
             "y_coefficient": float(weights[i, 1])} for i in ranked
        ],
    }


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Prefer verified local evidence; never silently hide a partial local run."""
    local = root / "artifacts/features"
    published = root / "docs/results"
    path = local / "summary.json" if local.exists() else published / "feature_summary.json"
    summary = json.loads(path.read_text())
    validate_summary(summary)
    if summary.get("source_sha256") != numerical_sources():
        raise ValueError("Feature evidence was generated with different numerical source.")
    if local.exists():
        model_path = local / "model.json"
        checkpoint = json.loads((root / ".state/features-report.json").read_text())
        if (checkpoint.get("status") != "completed"
                or checkpoint.get("signature") != summary.get("numerical_signature")):
            raise ValueError("Feature completion receipt is stale or incomplete.")
        for item in (path, model_path, local / "benchmark.png"):
            if checkpoint.get("outputs", {}).get(item.relative_to(root).as_posix()) != sha256(item):
                raise ValueError("Feature output checksum failed.")
        model = json.loads(model_path.read_text())
        baseline = root / "artifacts/benchmark/model.json"
        if (summary.get("baseline_sha256") != sha256(baseline)
                or model.get("baseline_sha256") != sha256(baseline)
                or summary.get("split_sha256") != sha256(root / "artifacts/game_splits.csv")):
            raise ValueError("Feature evidence no longer matches the baseline or frozen split.")
        study = selection_study(model, sha256(path), sha256(model_path))
        return summary, study, "Verified local experiment"
    study = json.loads((published / "feature_selection.json").read_text())
    if study["summary_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Published selection study does not match the published summary.")
    if summary.get("baseline_sha256") != sha256(published / "model.json"):
        raise ValueError("Published evidence does not match its baseline.")
    return summary, study, "Published real-data experiment snapshot"


def error_budget(summary: dict[str, Any], dimension: str) -> pd.DataFrame:
    """Decompose pooled squared coordinate error; never average slice RMSEs."""
    validate_summary(summary)
    if dimension not in ("role", "forecast_second"):
        raise ValueError("Choose role or forecast_second.")
    selected = summary["selected_model"]
    frame = pd.DataFrame([r for r in summary["slices"]
                          if r["model"] == selected and r["dimension"] == dimension])
    squared = frame.rows * frame.coordinate_rmse_yards ** 2
    frame["row_share_percent"] = 100 * frame.rows / frame.rows.sum()
    frame["squared_error_share_percent"] = 100 * squared / squared.sum()
    return frame[["value", "rows", "coordinate_rmse_yards", "row_share_percent",
                  "squared_error_share_percent"]]
