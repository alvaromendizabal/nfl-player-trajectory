"""Contracts for causal, aligned, bounded football feature generation."""

import numpy as np
import pandas as pd
import pytest

import nfl_trajectory.features as module
from nfl_trajectory.features import WIDTH, build_player_features, feature_catalog
from nfl_trajectory.motion import ENTITY, KEYS


@pytest.fixture
def tracking():
    rows, targets = [], []
    for player in range(1, 5):
        for frame in (1, 2, 4, 6, 7):
            rows.append({"game_id": 2023090700, "play_id": 1, "nfl_id": player,
                         "frame_id": frame, "x": 30 + player + frame * 0.3,
                         "y": 20 + player + frame * 0.1, "play_direction": "right",
                         "player_side": "Offense" if player < 3 else "Defense",
                         "player_role": "Targeted Receiver" if player == 1 else "Defensive Coverage",
                         "ball_land_x": 44.0, "ball_land_y": -0.2, "num_frames_output": 10,
                         "s": 3.2, "a": 0.0, "dir": 359.0, "o": 1.0,
                         "absolute_yardline_number": 30.0})
        for frame in range(1, 11):
            targets.append({"game_id": 2023090700, "play_id": 1, "nfl_id": player,
                            "frame_id": frame, "x": 50.0, "y": 10.0})
    return pd.DataFrame(rows), pd.DataFrame(targets)


def test_catalog_is_unique_and_contains_thousands_of_candidates(tracking):
    x, targets = tracking
    catalog = feature_catalog()
    assert len(catalog) == 2843
    assert catalog.feature.is_unique
    features = build_player_features(x).matrix(targets)
    assert features.shape == (len(targets), len(catalog))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_targets_cannot_supply_future_coordinates_or_metadata(tracking):
    x, targets = tracking
    bank = build_player_features(x)
    reference = bank.matrix(targets[KEYS])
    targets["x"], targets["y"], targets["player_role"] = np.inf, np.nan, "secret"
    np.testing.assert_array_equal(bank.matrix(targets), reference)


def test_shuffle_and_batch_parity(tracking):
    x, targets = tracking
    bank = build_player_features(x.sample(frac=1, random_state=3))
    shuffled = targets.sample(frac=1, random_state=7)
    expected = build_player_features(x).matrix(shuffled)
    actual = np.concatenate([bank.matrix(shuffled.iloc[:13]), bank.matrix(shuffled.iloc[13:])])
    np.testing.assert_array_equal(actual, expected)


def test_selected_columns_equal_full_matrix(tracking):
    x, targets = tracking
    bank = build_player_features(x)
    names = feature_catalog().feature.tolist()
    indices = [1, 98, 2020, len(names) - 1, len(names) - 90]
    selected = [names[i] for i in indices]
    np.testing.assert_array_equal(bank.matrix(targets, selected), bank.matrix(targets)[:, indices])


def test_actual_frame_lags_do_not_treat_gaps_as_contiguous(tracking):
    x, targets = tracking
    bank = build_player_features(x)
    values = bank.matrix(targets.iloc[:1], ["lag02__present", "lag03__present", "lag03__vx"])
    np.testing.assert_allclose(values, [[0.0, 1.0, 3.0]], atol=1e-5)


def test_missing_neighbors_and_telemetry_are_masked(tracking):
    x, targets = tracking
    x = x[x.nfl_id == 1].drop(columns=["s", "a", "dir", "o"])
    targets = targets[targets.nfl_id == 1]
    values = build_player_features(x).matrix(targets, ["opponent1__present", "opponent1__distance",
                                                     "telemetry__s__present", "lag00__s"])
    np.testing.assert_array_equal(values, 0.0)


def test_single_observation_is_finite(tracking):
    x, targets = tracking
    bank = build_player_features(x[x.frame_id == 7])
    assert np.isfinite(bank.matrix(targets)).all()
    np.testing.assert_array_equal(bank.matrix(targets, ["lag00__vx", "lag00__ax", "lag00__jerk"]), 0.0)


def test_prethrow_nonpredicted_players_still_supply_context(tracking):
    x, targets = tracking
    targets = targets[targets.nfl_id == 1]
    bank = build_player_features(x, targets[ENTITY])
    assert len(bank.state) == 1
    assert (bank.matrix(targets, ["opponent1__present"]) == 1).all()


def test_left_right_canonicalization_and_angle_wrapping(tracking):
    x, targets = tracking
    reference = build_player_features(x).matrix(targets)
    rotated = x.copy()
    rotated["play_direction"] = "left"
    for column in ("x", "ball_land_x", "absolute_yardline_number"):
        rotated[column] = 120 - rotated[column]
    for column in ("y", "ball_land_y"):
        rotated[column] = WIDTH - rotated[column]
    for column in ("dir", "o"):
        rotated[column] = (rotated[column] + 180) % 360
    np.testing.assert_allclose(build_player_features(rotated).matrix(targets), reference,
                               atol=2e-5, rtol=2e-5)
    angles = build_player_features(x).matrix(targets.iloc[:1], ["lag00__orientation_alignment"])
    assert angles[0, 0] > 0.99


def test_coordinates_outside_field_are_not_clipped(tracking):
    x, targets = tracking
    values = build_player_features(x).matrix(targets.iloc[:1], ["lag00__ball_dy"])
    assert values[0, 0] == pytest.approx(-0.2 - 21.7)


@pytest.mark.parametrize("column,value", [("x", np.nan), ("x", np.inf),
                                        ("num_frames_output", 0), ("num_frames_output", 1.5),
                                        ("play_direction", "north"), ("player_side", "unknown"),
                                        ("dir", np.inf)])
def test_invalid_inputs_fail_explicitly(tracking, column, value):
    x, _ = tracking
    x[column] = value
    with pytest.raises(ValueError):
        build_player_features(x)


def test_duplicate_keys_and_inconsistent_metadata_fail(tracking):
    x, _ = tracking
    with pytest.raises(ValueError):
        build_player_features(pd.concat([x, x.iloc[:1]]))
    x.loc[0, "ball_land_x"] += 1
    with pytest.raises(ValueError):
        build_player_features(x)


def test_unknown_target_and_forecast_overrun_fail(tracking):
    x, targets = tracking
    bank = build_player_features(x)
    targets.loc[0, "nfl_id"] = 99
    with pytest.raises(ValueError):
        bank.matrix(targets)
    targets = targets.iloc[:1].copy()
    targets["nfl_id"], targets["frame_id"] = 1, 11
    with pytest.raises(ValueError):
        bank.matrix(targets)


def test_memory_guard_is_applied_before_expansion(tracking, monkeypatch):
    x, targets = tracking
    bank = build_player_features(x)
    monkeypatch.setattr(module, "MAX_MATRIX_BYTES", 1024)
    with pytest.raises(MemoryError):
        bank.matrix(targets)
    assert bank.matrix(targets, ["time_seconds"]).shape == (len(targets), 1)


def test_other_plays_and_games_cannot_contaminate_neighbors(tracking):
    x, targets = tracking
    reference = build_player_features(x).matrix(targets)
    other = x.copy()
    other["game_id"] += 100
    other["x"] += 1000
    np.testing.assert_array_equal(build_player_features(pd.concat([x, other])).matrix(targets), reference)


def test_no_identity_or_outcome_columns_enter_schema(tracking):
    x, targets = tracking
    reference = build_player_features(x).matrix(targets)
    x["player_name"] = "secret"
    x["target_x"] = np.inf
    x["pass_result"] = "leak"
    np.testing.assert_array_equal(build_player_features(x).matrix(targets), reference)


def test_target_receiver_anchor_is_not_limited_to_nearest_neighbors(tracking):
    x, targets = tracking
    x.loc[x.nfl_id == 1, "x"] += 60
    selected = targets[targets.nfl_id == 3].iloc[:1]
    bank = build_player_features(x, selected[ENTITY])
    value = bank.matrix(selected, ["receiver1__present", "receiver1__dx", "passer1__present"])
    np.testing.assert_allclose(value, [[1.0, 58.0, 0.0]], atol=1e-5)
