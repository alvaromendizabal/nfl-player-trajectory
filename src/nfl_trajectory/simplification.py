"""A bounded combined-omission decision using chronological inner evidence only."""

from __future__ import annotations

import math
from typing import Any

REMOVED_FAMILIES = frozenset(
    {"history_lags", "history_summaries", "robust_history", "forecast_interactions"}
)
FOLDS = ("inner_1", "inner_2", "inner_3", "development")
MODEL = "without_direct_histories_and_forecast_crosses"


def decision(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep development out of the stopping calculation; reject incomplete evidence."""
    if [fold["fold"] for fold in folds] != list(FOLDS):
        raise ValueError("Combined omission requires all four authorized folds in order.")
    reference, candidate, weights, costs = [], [], [], []
    for fold in folds:
        before = fold["reference"]["coordinate_rmse_yards"]
        after = fold["metrics"]["coordinate_rmse_yards"]
        rows = fold["rows"]
        if (
            not all(math.isfinite(value) for value in (before, after, rows))
            or before <= 0
            or after < 0
            or rows <= 0
            or fold["feature_count"] >= fold["parent_feature_count"]
            or fold["feature_count"] < 1
            or fold["holdout_evaluation"] != "not_run"
        ):
            raise ValueError("Combined omission has invalid metrics, widths, or scope.")
        if fold["fold"] == "development":
            continue
        reference.append(before)
        candidate.append(after)
        weights.append(rows)
        costs.append(after / before - 1)

    def pooled(values: list[float]) -> float:
        return math.sqrt(
            math.fsum(value * value * weight for value, weight in zip(values, weights, strict=True))
            / sum(weights)
        )

    before, after = pooled(reference), pooled(candidate)
    gain = 1 - after / before
    return {
        "pooled_reference_rmse_yards": before,
        "pooled_candidate_rmse_yards": after,
        "pooled_relative_gain": gain,
        "inner_relative_costs": costs,
        "requires_smaller_representation_followup": gain >= 0.005 and max(costs) <= 0.01,
        "minimum_gain_to_reopen": 0.005,
        "maximum_fold_cost": 0.01,
        "selection_scope": "Three chronological inner folds; development excluded.",
    }
