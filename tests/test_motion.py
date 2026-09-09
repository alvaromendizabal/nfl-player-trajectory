"""Independent numerical and leakage-contract tests."""

import math

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.motion import KEYS, constant_velocity, split_games, trajectory_metrics
from nfl_trajectory.report import synthetic_play


def test_official_rmse_has_two_coordinate_denominator() -> None:
    truth = pd.DataFrame([[2023090700, 1, 1, 1, 0.0, 0.0]], columns=KEYS + ["x", "y"])
    prediction = truth.copy()
    prediction[["x", "y"]] = [3.0, 4.0]
    metrics = trajectory_metrics(truth, prediction)
    assert metrics["coordinate_rmse_yards"] == pytest.approx(math.sqrt(12.5))
    assert metrics["ade_frame_weighted_yards"] == 5
    assert metrics["fde_trajectory_weighted_yards"] == 5


@pytest.mark.parametrize(
    ("predicted_x", "predicted_y", "expected_rmse"),
    [
        ([1.1, 2.0, 3.0], [4.0, 2.2, 3.0], 0.0913),
        ([0.0, 2.0, 3.0], [4.0, 2.2, 3.0], 0.4163),
        ([1.0, 2.0, 1.0], [4.0, 0.0, 3.0], 1.1547),
    ],
)
def test_rmse_matches_published_kaggle_examples(
    predicted_x: list[float], predicted_y: list[float], expected_rmse: float
) -> None:
    # Independent examples from https://www.kaggle.com/code/metric/nfl-2025 (v4).
    truth = pd.DataFrame(
        [[2023090700, 12, 2, frame, x, y] for frame, x, y in [(1, 1, 4), (2, 2, 2), (3, 3, 3)]],
        columns=KEYS + ["x", "y"],
    )
    prediction = truth[KEYS].assign(x=predicted_x, y=predicted_y)
    score = trajectory_metrics(truth, prediction)["coordinate_rmse_yards"]
    assert round(score, 4) == expected_rmse


def test_row_order_does_not_change_score() -> None:
    _, truth = synthetic_play()
    assert (
        trajectory_metrics(truth, truth.sample(frac=1, random_state=12))["coordinate_rmse_yards"]
        == 0
    )


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate", "nan", "infinity", "key"])
def test_bad_predictions_are_rejected(corruption: str) -> None:
    _, truth = synthetic_play()
    prediction = truth.copy()
    if corruption == "missing":
        prediction = prediction.iloc[:-1]
    elif corruption == "extra":
        prediction = pd.concat([prediction, prediction.iloc[:1].assign(frame_id=100)])
    elif corruption == "duplicate":
        prediction = pd.concat([prediction.iloc[:-1], prediction.iloc[:1]])
    elif corruption == "key":
        prediction.loc[0, "nfl_id"] = 999
    else:
        prediction.loc[0, "x"] = np.nan if corruption == "nan" else np.inf
    with pytest.raises(ValueError):
        trajectory_metrics(truth, prediction)


def test_output_frame_clock_restarts_at_one_and_ignores_target_coordinates() -> None:
    past, future = synthetic_play()
    prediction = constant_velocity(past, future)
    first = prediction.query("nfl_id == 1 and frame_id == 1").iloc[0]
    assert first.x == pytest.approx(35.5)
    assert first.y == pytest.approx(21.1)
    future[["x", "y"]] = 1e6
    pd.testing.assert_frame_equal(constant_velocity(past, future), prediction)


def test_irregular_observed_frames_use_elapsed_time() -> None:
    past = pd.DataFrame(
        [[2023090700, 1, 1, 2, 0.0, 0.0], [2023090700, 1, 1, 4, 2.0, 0.0]],
        columns=KEYS + ["x", "y"],
    )
    target = past.iloc[:1][KEYS].assign(frame_id=1)
    assert constant_velocity(past, target).x.iloc[0] == 3


def test_single_observation_is_stationary_and_missing_entity_fails() -> None:
    past, future = synthetic_play()
    one = past.groupby(KEYS[:3]).tail(1)
    prediction = constant_velocity(one, future[KEYS])
    assert (prediction.x == 35).all()
    with pytest.raises(ValueError):
        constant_velocity(one, future[KEYS].assign(nfl_id=999))


def test_fde_uses_last_frame_per_trajectory() -> None:
    _, truth = synthetic_play()
    prediction = truth.copy()
    prediction.loc[prediction.frame_id == 20, "x"] += 4
    score = trajectory_metrics(truth, prediction)
    assert score["fde_trajectory_weighted_yards"] == 4
    assert score["ade_frame_weighted_yards"] == pytest.approx(0.2)


def test_temporal_split_keeps_dates_and_games_together() -> None:
    ids = [
        int(day.strftime("%Y%m%d") + suffix)
        for day in pd.date_range("2023-09-01", periods=20)
        for suffix in ["00", "01"]
    ]
    result = split_games(pd.DataFrame({"game_id": ids + ids[:2]}))
    assert len(result) == 40
    assert result.groupby("game_date").split.nunique().max() == 1
    ranges = result.groupby("split").game_date.agg(["min", "max"])
    assert ranges.loc["train", "max"] < ranges.loc["validation", "min"]
    assert ranges.loc["validation", "max"] < ranges.loc["holdout", "min"]


def test_tiny_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_games(pd.DataFrame({"game_id": [2023090700]}))
