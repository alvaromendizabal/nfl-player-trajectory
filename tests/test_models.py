"""Physical limits, strict alignment, leakage boundaries, and equivariant inference."""

import copy

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.models import (
    BASELINES,
    design,
    fit_statistics,
    predict,
    sufficient_statistics,
)
from nfl_trajectory.motion import KEYS, constant_velocity


@pytest.fixture
def tracking():
    observed, truth = [], []
    for player, role in [(1, "Targeted Receiver"), (2, "Defensive Coverage")]:
        for frame in [1, 2, 4, 7, 8]:
            observed.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 30 + 0.5 * frame,
                    "y": 20 + player + 0.2 * frame,
                    "ball_land_x": 40,
                    "ball_land_y": 28,
                    "num_frames_output": 12,
                    "player_role": role,
                }
            )
        for frame in range(1, 13):
            truth.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 34 + frame * 0.5 - frame**2 * 0.01,
                    "y": 21.6 + player + frame * 0.2 + frame**2 * 0.01,
                }
            )
    return pd.DataFrame(observed), pd.DataFrame(truth)


def fitted_model(x, y):
    state, features = design(x, y[KEYS])
    return fit_statistics([sufficient_statistics(state, features, y)])


def test_recent_velocity_is_exact_for_irregular_constant_speed(tracking):
    x, y = tracking
    expected = constant_velocity(x, y[KEYS])
    for model in ["constant_velocity", "smoothed_velocity", "constant_acceleration"]:
        actual = predict(x.sample(frac=1, random_state=4), y[KEYS], model)
        np.testing.assert_allclose(actual[["x", "y"]], expected[["x", "y"]], atol=1e-11)


def test_single_frame_has_zero_velocity_and_acceleration(tracking):
    x, y = tracking
    x = x[x.frame_id == 8]
    for model in ["constant_velocity", "smoothed_velocity", "constant_acceleration"]:
        actual = predict(x, y[KEYS], model)
        reference = predict(x, y[KEYS], "last_position")
        pd.testing.assert_frame_equal(actual, reference)


def test_ball_arrival_reaches_supplied_endpoint(tracking):
    x, y = tracking
    actual = predict(x, y[y.frame_id == 12][KEYS], "ball_arrival")
    np.testing.assert_allclose(actual[["x", "y"]], [[40, 28], [40, 28]])


@pytest.mark.parametrize("model", [*BASELINES, "role_ridge"])
def test_targets_never_supply_coordinates_and_order_is_preserved(tracking, model):
    x, y = tracking
    fitted = fitted_model(x, y)
    target = y.sample(frac=1, random_state=5).copy()
    target[["x", "y"]] = np.nan
    actual = predict(x, target, model, fitted)
    expected = predict(x, target[KEYS], model, fitted)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual[KEYS], target[KEYS].reset_index(drop=True))


@pytest.mark.parametrize("model", [*BASELINES, "role_ridge"])
def test_translation_and_rotation_equivariance(tracking, model):
    x, y = tracking
    fitted = fitted_model(x, y)
    reference = predict(x, y[KEYS], model, fitted)
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([51.0, -13.0])
    transformed = x.copy()
    for columns in [["x", "y"], ["ball_land_x", "ball_land_y"]]:
        transformed[columns] = x[columns].to_numpy() @ rotation.T + offset
    actual = predict(transformed, y[KEYS], model, fitted)
    np.testing.assert_allclose(
        actual[["x", "y"]], reference[["x", "y"]].to_numpy() @ rotation.T + offset, atol=1e-10
    )


def test_additive_training_statistics_match_single_batch(tracking):
    x, y = tracking
    state, features = design(x, y[KEYS])
    whole = fit_statistics([sufficient_statistics(state, features, y)])
    parts = []
    for ids in [np.arange(0, len(y), 2), np.arange(1, len(y), 2)]:
        parts.append(sufficient_statistics(state.iloc[ids], features[ids], y.iloc[ids]))
    sharded = fit_statistics(parts)
    np.testing.assert_allclose(
        whole["global"]["coefficients"], sharded["global"]["coefficients"], atol=1e-11
    )
    for role in whole["roles"]:
        np.testing.assert_allclose(
            whole["roles"][role]["coefficients"], sharded["roles"][role]["coefficients"], atol=1e-11
        )


def test_unseen_role_uses_trained_global_fallback(tracking):
    x, y = tracking
    fitted = fitted_model(x, y)
    x["player_role"] = "Unseen role"
    actual = predict(x, y[KEYS], "role_ridge", fitted)
    reference = copy.deepcopy(fitted)
    reference["roles"] = {"Unseen role": reference["global"]}
    pd.testing.assert_frame_equal(actual, predict(x, y[KEYS], "role_ridge", reference))


@pytest.mark.parametrize(
    "issue", ["missing_player", "horizon", "duplicate", "nan_input", "zero_horizon"]
)
def test_invalid_tracking_fails_before_predictions(tracking, issue):
    x, y = tracking
    if issue == "missing_player":
        y.loc[0, "nfl_id"] = 99
    elif issue == "horizon":
        y.loc[0, "frame_id"] = 13
    elif issue == "duplicate":
        y = pd.concat([y, y.iloc[:1]])
    elif issue == "nan_input":
        x.loc[0, "ball_land_x"] = np.nan
    else:
        x["num_frames_output"] = 0
    with pytest.raises(ValueError):
        predict(x, y[KEYS], "constant_velocity")
