"""Controlled estimator-capacity diagnostic on the preserved training-only core screen."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.features import feature_catalog
from nfl_trajectory.motion import KEYS, require_keys


def selected_features(fitted: dict[str, Any], training_games: list[int]) -> list[str]:
    """Reuse only a screen fitted on exactly this experiment's training partition."""
    if sorted(fitted["training_games"]) != sorted(training_games):
        raise ValueError("Feature screen must use exactly the training games.")
    names = list(
        dict.fromkeys(name for model in fitted["models"].values() for name in model["features"])
    )
    if not names or not set(names).issubset(set(feature_catalog().feature)):
        raise ValueError("Capacity experiment requires known, nonempty core features.")
    return names


def validate_partitions(targets: pd.DataFrame, labels: np.ndarray, splits: pd.DataFrame) -> None:
    """Reject game overlap, holdout rows, unknown games, and mismatched cache labels."""
    require_keys(targets)
    if splits.game_id.duplicated().any() or len(labels) != len(targets):
        raise ValueError("Split assignments and cache labels must be unambiguous.")
    expected = targets.game_id.map(splits.set_index("game_id").split)
    if not expected.isin(["train", "validation"]).all() or not np.array_equal(expected, labels):
        raise ValueError("Only matching training/development rows may enter this experiment.")
    train = splits[splits.split.eq("train")].game_id
    development = splits[splits.split.eq("validation")].game_id
    if train.empty or development.empty or train.max() // 100 >= development.min() // 100:
        raise ValueError("Development must follow training on disjoint dates.")


def error_rows(
    targets: pd.DataFrame,
    baseline: np.ndarray,
    response: np.ndarray,
    sign: np.ndarray,
    truth: np.ndarray,
) -> pd.DataFrame:
    """Restore field coordinates before computing the official coordinate errors."""
    require_keys(targets)
    shape = (len(targets), 2)
    if any(a.shape != shape for a in (baseline, response, truth)) or sign.shape != (
        len(targets),
        1,
    ):
        raise ValueError("Predictions must preserve every requested coordinate row.")
    if not np.isin(sign, [-1, 1]).all():
        raise ValueError("Field direction must be an invertible sign.")
    delta = baseline + response * sign - truth
    if not np.isfinite(delta).all():
        raise ValueError("Prediction errors must be finite.")
    result = targets[KEYS].copy()
    result[["dx", "dy"]] = delta
    return result
