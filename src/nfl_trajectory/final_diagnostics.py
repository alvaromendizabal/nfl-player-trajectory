"""Descriptive error concentration after the immutable final evaluation; never selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.final_protocol import digest
from nfl_trajectory.final_results import load_final_results
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.runtime import Run, atomic_json, sha256, stage


def summarize_errors(errors: pd.DataFrame) -> dict[str, Any]:
    """Decompose the unchanged pooled error; retain all rare and difficult observations."""
    primary = errors.loc[errors.scenario.eq("complete")].copy()
    require_keys(primary)
    if primary.empty or not np.isfinite(primary[["dx", "dy"]].to_numpy()).all():
        raise ValueError("Diagnostics require nonempty, finite, unique primary errors.")
    primary["sse"] = primary.dx**2 + primary.dy**2
    total = float(primary.sse.sum())
    if total <= 0:
        raise ValueError("Error shares require a positive total squared error.")
    games = primary.groupby("game_id").sse.sum().sort_values(ascending=False)
    plays = primary.groupby(["game_id", "play_id"]).sse.sum()
    trajectories = primary.groupby(ENTITY).sse.sum()
    bins = []
    for second, part in primary.groupby(np.ceil(primary.frame_id / 10).astype(int)):
        bins.append(
            {
                "forecast_second": int(second),
                "rows": len(part),
                "games": int(part.game_id.nunique()),
                "trajectories": len(part[ENTITY].drop_duplicates()),
                "row_share": len(part) / len(primary),
                "squared_error_share": float(part.sse.sum() / total),
                "coordinate_rmse_yards": float(np.sqrt(part.sse.sum() / (2 * len(part)))),
            }
        )
    comparisons = []
    for name in ("constant_velocity", "role_ridge", "without_telemetry"):
        reference = errors.loc[errors.scenario.eq(name)]
        require_keys(reference)
        alignment = primary[KEYS].merge(reference[KEYS], how="outer", indicator=True)
        if not alignment._merge.eq("both").all():
            raise ValueError("Reference diagnostics must cover exactly the primary forecast keys.")
        reference_sse = reference.assign(sse=reference.dx**2 + reference.dy**2)
        per_game = reference_sse.groupby("game_id").sse.sum().reindex(games.index)
        comparisons.append(
            {
                "reference": name,
                "games_improved": int((games < per_game).sum()),
                "games_tied": int((games == per_game).sum()),
                "games": len(games),
                "rmse_reduction_percent": float(
                    100 * (1 - np.sqrt(total / reference_sse.sse.sum()))
                ),
            }
        )
    return {
        "rows": len(primary),
        "games": len(games),
        "plays": len(plays),
        "trajectories": len(trajectories),
        "coordinate_rmse_yards": float(np.sqrt(total / (2 * len(primary)))),
        "forecast_bins": bins,
        "top_game_squared_error_share": float(games.iloc[0] / total),
        "top_three_games_squared_error_share": float(games.iloc[:3].sum() / total),
        "top_play_squared_error_share": float(plays.max() / total),
        "top_trajectory_squared_error_share": float(trajectories.max() / total),
        "reference_comparisons": comparisons,
        "scope": (
            "Post-evaluation descriptive analysis, not a model-selection experiment. "
            "Every row remains in the official score. Horizon bins contain different "
            "trajectory cohorts; high-horizon support is sparse. Baseline improvements "
            "combine representation and estimator effects, not isolated feature attribution."
        ),
    }


def publish_diagnostics(root: Path, run: Run) -> dict[str, Any]:
    reports = load_final_results(root)
    if reports is None:
        raise ValueError("Publish the verified final evaluation before its diagnostics.")
    errors = root / "artifacts/final/evaluation/errors.csv"
    if sha256(errors) != reports["evaluation"]["errors_sha256"]:
        raise ValueError("Diagnostic errors differ from the scored, sealed evaluation.")
    provenance = {
        "seal": reports["evaluation"]["seal"],
        "errors_sha256": sha256(errors),
        "evaluation_sha256": sha256(root / "docs/results/final_evaluation.json"),
        "implementation_sha256": sha256(Path(__file__)),
    }
    output = root / "artifacts/final/evaluation/diagnostics.json"

    def action() -> None:
        values = summarize_errors(pd.read_csv(errors))
        primary = next(m for m in reports["evaluation"]["metrics"] if m["scenario"] == "complete")
        if not np.isclose(
            values["coordinate_rmse_yards"], primary["coordinate_rmse_yards"], rtol=0, atol=1e-12
        ):
            raise ValueError("Diagnostics disagree with the unchanged official metric.")
        report = {"status": "passed", "provenance": provenance, **values}
        atomic_json(output, {**report, "source_signature": digest(report)})

    stage(root, "final-diagnostics", digest(provenance), [output], action, run)
    report = json.loads(output.read_text())
    atomic_json(root / "docs/results/final_diagnostics.json", report)
    return dict(report)


def load_diagnostics(root: Path) -> dict[str, Any] | None:
    path = root / "docs/results/final_diagnostics.json"
    if not path.is_file():
        return None
    reports = load_final_results(root)
    report = json.loads(path.read_text())
    expected = report["provenance"]
    if (
        reports is None
        or report["source_signature"]
        != digest({k: v for k, v in report.items() if k != "source_signature"})
        or expected["seal"] != reports["evaluation"]["seal"]
        or expected["errors_sha256"] != reports["evaluation"]["errors_sha256"]
        or expected["evaluation_sha256"] != sha256(root / "docs/results/final_evaluation.json")
        or expected["implementation_sha256"] != sha256(Path(__file__))
    ):
        raise ValueError("Final diagnostic lineage or report content is stale.")
    return dict(report)


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    with Run(project, "final-diagnostics") as current:
        publish_diagnostics(project, current)
