"""Independent finite-difference, information-time and scale contracts."""

import copy

import numpy as np
import pytest

from nfl_trajectory.motion_targets import fit_motion_scales, motion_targets
from nfl_trajectory.temporal_data import STATIC_NAMES


def trajectory(frames=(1, 2, 3, 4)):
    frames = np.asarray(frames)
    time = frames / 10
    displacement = time[:, None] * np.array([3.0, -2.0])
    displacement += 0.5 * time[:, None] ** 2 * np.array([4.0, 6.0])
    return {
        "keys": np.column_stack([np.ones((len(frames), 3), int), frames]),
        "player": np.zeros(len(frames), int),
        "ids": np.array([1]),
        "baseline": displacement * 0.3,
        "truth": displacement * 0.7,
        "static": np.zeros((1, len(STATIC_NAMES))),
        "split": "train",
    }


def test_quadratic_path_has_correct_midpoint_velocity_and_acceleration():
    sample = trajectory()
    labels = motion_targets(sample)
    midpoint = (sample["keys"][:, 3] - 0.5) / 10
    expected = np.array([3, -2]) + midpoint[:, None] * np.array([4, 6])
    np.testing.assert_allclose(labels["velocity"], expected)
    np.testing.assert_array_equal(labels["velocity_mask"], True)
    np.testing.assert_array_equal(labels["acceleration_mask"], [False, True, True, True])
    np.testing.assert_allclose(labels["acceleration"][1:], [[4, 6]] * 3, atol=1e-12)


def test_gaps_and_stale_endpoint_do_not_create_targets():
    sample = trajectory((1, 3, 4, 5))
    sample["static"][0, STATIC_NAMES.index("last_observation_age")] = 0.1
    labels = motion_targets(sample)
    np.testing.assert_array_equal(labels["velocity_mask"], [False, False, True, True])
    np.testing.assert_array_equal(labels["acceleration_mask"], [False, False, False, True])
    np.testing.assert_array_equal(labels["velocity"][:2], 0)


def test_row_permutation_and_other_player_cannot_supply_predecessor():
    sample = trajectory((1, 3, 4))
    sample["ids"] = np.array([1, 2])
    sample["static"] = np.zeros((2, len(STATIC_NAMES)))
    sample["keys"][1, 2], sample["player"][1] = 2, 1
    before = motion_targets(sample)
    assert not before["velocity_mask"][2]
    order = [2, 0, 1]
    for name in ("keys", "player", "baseline", "truth"):
        sample[name] = sample[name][order]
    after = motion_targets(sample)
    for name in before:
        np.testing.assert_array_equal(after[name], before[name][order])


def test_translation_baseline_decomposition_and_reflection():
    sample = trajectory()
    before = copy.deepcopy(sample)
    expected = motion_targets(sample)
    for name in sample:
        np.testing.assert_array_equal(sample[name], before[name])
    sample["baseline"] += np.array([500, -200])
    sample["truth"] -= np.array([500, -200])
    for name in expected:
        np.testing.assert_allclose(motion_targets(sample)[name], expected[name], atol=1e-10)
    sample["baseline"][:, 1] *= -1
    sample["truth"][:, 1] *= -1
    actual = motion_targets(sample)
    for name in ("velocity", "acceleration"):
        np.testing.assert_allclose(actual[name], expected[name] * [1, -1], atol=1e-10)


def test_scale_is_training_only_and_shared_across_axes():
    sample = trajectory()
    labels = motion_targets(sample)
    scales = fit_motion_scales([sample])
    for name, value in scales.items():
        assert value == pytest.approx(np.sqrt(np.mean(labels[name][labels[name + "_mask"]] ** 2)))
    sample["split"] = "validation"
    with pytest.raises(ValueError, match="training"):
        fit_motion_scales([sample])


@pytest.mark.parametrize("problem", ["duplicate", "identity", "nonfinite", "play"])
def test_invalid_label_contract_is_rejected(problem):
    sample = trajectory()
    if problem == "duplicate":
        sample["keys"][1] = sample["keys"][0]
    elif problem == "identity":
        sample["ids"][0] = 2
    elif problem == "play":
        sample["keys"][0, 1] = 99
    else:
        sample["truth"][0, 0] = np.nan
    with pytest.raises(ValueError):
        motion_targets(sample)
