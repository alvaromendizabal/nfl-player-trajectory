"""Input timing and numerical contracts for the raw-context feature study."""

import numpy as np
import pytest
from test_feature_research import example_tracking

from nfl_trajectory.context_features import (
    build_context_features,
    context_catalog,
    load_context,
    save_context,
)
from nfl_trajectory.features import WIDTH
from nfl_trajectory.motion import ENTITY


def metadata_tracking():
    inputs, targets = example_tracking()
    inputs["player_height"] = "6-1"
    inputs["player_weight"] = 210
    inputs["player_birth_date"] = "1999-09-07"
    inputs["player_position"] = "CB"
    return inputs, targets


def test_metadata_age_and_target_poisoning():
    inputs, targets = metadata_tracking()
    bank = build_context_features(inputs, targets[ENTITY])
    names = context_catalog().feature.tolist()
    assert len(names) == 1468
    assert len(names) == len(set(names))
    values = bank.matrix(targets)
    assert values[0, names.index("metadata__height_inches")] == 73
    assert values[0, names.index("metadata__weight_pounds")] == 210
    assert abs(values[0, names.index("metadata__age_years")] - 24) < 0.01
    inputs["future_x"] = np.inf
    inputs["pass_result"] = "forbidden"
    targets[["x", "y"]] = np.nan
    changed = build_context_features(inputs, targets[ENTITY]).matrix(targets)
    np.testing.assert_array_equal(changed, values)


def test_context_batch_subset_and_rotation_parity(tmp_path):
    inputs, targets = metadata_tracking()
    bank = build_context_features(inputs, targets[ENTITY])
    names = context_catalog().feature.tolist()
    columns = names[::57]
    expected = bank.matrix(targets, columns)
    np.testing.assert_array_equal(
        expected,
        np.concatenate(
            [bank.matrix(targets.iloc[:12], columns), bank.matrix(targets.iloc[12:], columns)]
        ),
    )
    inputs.x = 120 - inputs.x
    inputs.y = WIDTH - inputs.y
    inputs.ball_land_x = 120 - inputs.ball_land_x
    inputs.ball_land_y = WIDTH - inputs.ball_land_y
    inputs.play_direction = "left"
    rotated = build_context_features(inputs, targets[ENTITY])
    np.testing.assert_allclose(rotated.matrix(targets, columns), expected, atol=1e-5)
    path = tmp_path / "context.npz"
    save_context(path, bank)
    np.testing.assert_array_equal(load_context(path).matrix(targets), bank.matrix(targets))


def test_missing_metadata_and_neighbours_have_zero_masks():
    inputs, targets = example_tracking()
    inputs, targets = inputs[inputs.nfl_id.eq(1)], targets[targets.nfl_id.eq(1)]
    bank = build_context_features(inputs, targets[ENTITY])
    columns = [
        "metadata__height_present",
        "metadata__weight_present",
        "metadata__age_present",
        "matched__opponent1__w03__coverage",
    ]
    np.testing.assert_array_equal(bank.matrix(targets, columns), 0)


def test_context_metadata_consistency_and_height_format():
    inputs, targets = metadata_tracking()
    inputs.loc[0, "player_weight"] = 999
    with pytest.raises(ValueError, match="changes"):
        build_context_features(inputs, targets[ENTITY])
    inputs["player_weight"] = 210
    inputs["player_height"] = "unknown-format"
    with pytest.raises(ValueError, match="feet-inches"):
        build_context_features(inputs, targets[ENTITY])


def test_matched_history_uses_exact_frame_ids():
    inputs, targets = metadata_tracking()
    inputs = inputs[~(inputs.nfl_id.eq(2) & inputs.frame_id.eq(7))]
    # Nearest opponent for player 1 is player 2; missing frame 7 is not forward filled.
    bank = build_context_features(inputs, targets[ENTITY])
    subset = targets[targets.nfl_id.eq(1)]
    observed = bank.matrix(subset, ["matched__opponent1__w03__coverage"])
    np.testing.assert_allclose(observed, 2 / 3)
