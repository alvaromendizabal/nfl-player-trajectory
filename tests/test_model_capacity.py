"""Independent controls for the bounded capacity comparison."""

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.model_capacity import error_rows, selected_features, validate_partitions
from nfl_trajectory.motion import KEYS


def test_screen_rejects_development_trained_features() -> None:
    fitted = {
        "training_games": [2023090700, 2023091400],
        "models": {"a": {"features": ["time_seconds"]}},
    }
    with pytest.raises(ValueError, match="training games"):
        selected_features(fitted, [2023090700])


@pytest.mark.parametrize("corruption", ["holdout", "wrong_label", "duplicate_split"])
def test_capacity_rejects_invalid_partitions(corruption: str) -> None:
    targets = pd.DataFrame([[2023090700, 1, 1, 1], [2023091400, 1, 1, 1]], columns=KEYS)
    splits = pd.DataFrame({"game_id": [2023090700, 2023091400], "split": ["train", "validation"]})
    labels = np.array(["train", "validation"])
    if corruption == "holdout":
        splits.loc[1, "split"] = "holdout"
    elif corruption == "wrong_label":
        labels[1] = "train"
    else:
        splits = pd.concat([splits, splits.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_partitions(targets, labels, splits)


def test_inverse_rotation_preserves_coordinate_error() -> None:
    targets = pd.DataFrame([[2023090700, 1, 1, 1], [2023091400, 1, 1, 1]], columns=KEYS)
    baseline = np.array([[30.0, 20.0], [90.0, 33.0]])
    response = np.array([[3.0, 4.0], [3.0, 4.0]])
    errors = error_rows(targets, baseline, response, np.array([[1], [-1]]), baseline)
    np.testing.assert_array_equal(errors[["dx", "dy"]], [[3.0, 4.0], [-3.0, -4.0]])
    assert np.sqrt(np.mean(errors[["dx", "dy"]].to_numpy() ** 2)) == pytest.approx(np.sqrt(12.5))
