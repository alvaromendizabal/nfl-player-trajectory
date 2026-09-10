"""Independent temporal input, coordinate-system, and output coverage contracts."""

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.feature_candidates import historical_encodings
from nfl_trajectory.features import WIDTH
from nfl_trajectory.motion import KEYS
from nfl_trajectory.temporal_data import (
    attach_targets,
    edge_features,
    history_lookup,
    play_features,
    reflect,
)


def example(players=3):
    rows = []
    for player in range(players):
        for frame in range(1, 25):
            rows.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player + 1,
                    "frame_id": frame,
                    "x": 40 + player * 2 + 0.3 * frame,
                    "y": 20 + player + 0.15 * frame,
                    "play_direction": "right",
                    "player_side": "Offense" if player < 2 else "Defense",
                    "player_role": ["Targeted Receiver", "Passer", "Defensive Coverage"][
                        min(player, 2)
                    ],
                    "ball_land_x": 55.0,
                    "ball_land_y": -0.2,
                    "num_frames_output": 60,
                    "s": 3.3,
                    "a": 0.2,
                    "dir": 60.0,
                    "o": 80.0,
                    "player_to_predict": player == 0,
                }
            )
    raw = pd.DataFrame(rows)
    targets = pd.DataFrame([[2023090700, 1, 1, f] for f in (1, 5, 51, 60)], columns=KEYS)
    baseline = np.array([[48.0, 23.0]] * len(targets))
    return raw, targets, baseline


def sample(players=3):
    raw, targets, baseline = example(players)
    truth = baseline + targets.frame_id.to_numpy()[:, None] / 10 * np.array([0.2, -0.3])
    return attach_targets(play_features(raw), targets, baseline, truth)


def test_future_coordinates_do_not_enter_temporal_inputs():
    raw, targets, baseline = example()
    observed = play_features(raw)
    first = attach_targets(observed, targets, baseline, baseline)
    targets["x"], targets["y"], targets["player_role"] = np.inf, np.nan, "secret"
    second = attach_targets(observed, targets, baseline, baseline + 999)
    for name in set(first) - {"truth"}:
        np.testing.assert_array_equal(first[name], second[name])
    assert second["time"].max() == 6
    assert len(second["keys"]) == 4  # Includes frames 51 and 60; no 48-frame truncation.


def test_reflection_matches_independently_transformed_tracking():
    raw, _, _ = example()
    expected = reflect(play_features(raw))
    raw["y"] = WIDTH - raw.y
    raw["ball_land_y"] = WIDTH - raw.ball_land_y
    raw["dir"] = (180 - raw.dir) % 360
    raw["o"] = (180 - raw.o) % 360
    actual = play_features(raw)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-6)
    np.testing.assert_allclose(edge_features(actual), edge_features(expected), atol=2e-6)
    original = sample()
    recovered = reflect(reflect(original))
    for name in original:
        np.testing.assert_allclose(original[name], recovered[name], atol=2e-6)


def test_history_padding_and_player_slots_survive_shuffle():
    raw, _, _ = example()
    raw = raw[~((raw.nfl_id == 2) & (raw.frame_id < 21))]
    a, b = play_features(raw), play_features(raw.sample(frac=1, random_state=7))
    assert a["observed"][1].sum() == 4
    assert np.count_nonzero(a["history"][1, :16]) == 0
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_reject_unknown_requested_player_and_outside_horizon():
    raw, targets, baseline = example()
    features = play_features(raw)
    targets.loc[0, "nfl_id"] = 999
    with pytest.raises(ValueError, match="requested row"):
        attach_targets(features, targets, baseline)
    targets.loc[0, "nfl_id"], targets.loc[0, "frame_id"] = 1, 61
    with pytest.raises(ValueError, match="requested row"):
        attach_targets(features, targets, baseline)


def test_frozen_neural_history_matches_chronological_encoder():
    observations = pd.DataFrame(
        {
            "game_id": [2023090700, 2023091400, 2023092100, 2023092100],
            "play_id": [1, 1, 1, 1],
            "nfl_id": [1, 1, 1, 999],
            "player_role": ["Targeted Receiver"] * 4,
            "error_x": [2.0, 4.0, np.nan, np.nan],
            "error_y": [-3.0, -6.0, np.nan, np.nan],
        }
    )
    encoded, fitted = historical_encodings(observations, {2023090700, 2023091400})
    actual = history_lookup(fitted, observations.iloc[2:])
    np.testing.assert_allclose(actual, encoded[2:])
    assert actual[1, 1] == 1  # Unknown player's cold prior is explicit.
    assert actual[1, 2] == actual[1, 3] == 0
    transformed = sample()
    transformed["static"][:, 13] = -2.0
    transformed["static"][:, 18] = -4.0
    reflected = reflect(transformed)
    np.testing.assert_array_equal(reflected["static"][:, 13], 2)
    np.testing.assert_array_equal(reflected["static"][:, 18], 4)
