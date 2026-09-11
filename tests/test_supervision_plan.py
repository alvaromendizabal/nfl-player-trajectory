"""Matched batching is deterministic, training-only and uniformly coordinate weighted."""

import copy

import numpy as np
import pytest
from test_motion_targets import trajectory

from nfl_trajectory.supervision_plan import training_plan


def samples():
    result = [trajectory((1,)), trajectory((1, 2, 3)), trajectory((1, 2, 49, 50))]
    for i, sample in enumerate(result):
        sample["keys"][:, 1] = i + 1
    return result


def test_permutation_is_stable_independent_of_input_order():
    data = samples()
    before = copy.deepcopy(data)
    first = training_plan(data, 2)
    assert first == training_plan(list(reversed(data)), 2)
    assert sorted(first["permutation"]) == [0, 1, 2]
    for original, preserved in zip(data, before, strict=True):
        for name in original:
            np.testing.assert_array_equal(original[name], preserved[name])


def test_fixed_denominators_count_all_long_and_variable_rows():
    result = training_plan(samples(), 2)
    assert result["training_rows"] == 8
    assert result["training_rows_after_frame_48"] == 2
    assert result["coordinate_loss_denominator"] == 8
    assert sum(result["batch_coordinate_counts"]) == 16
    assert result["velocity_coordinates"] == 14
    assert result["velocity_loss_denominator"] == 7
    assert result["training_ready"] is False


def test_validation_samples_are_rejected_before_label_access():
    bad = {"split": "validation"}
    with pytest.raises(ValueError, match="Only training"):
        training_plan([bad])


def test_duplicate_play_rejected():
    sample = trajectory()
    with pytest.raises(ValueError, match="Duplicate"):
        training_plan([sample, sample])


@pytest.mark.parametrize("kwargs", [{"batch_plays": 0}, {"seed": -1}, {"epoch": -1}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        training_plan(samples(), **kwargs)


def test_scale_matches_direct_training_coordinate_calculation():
    from nfl_trajectory.motion_targets import motion_targets

    data = samples()
    values = []
    for sample in data:
        target = motion_targets(sample)
        values.extend(target["velocity"][target["velocity_mask"]])
    expected = float(np.sqrt(np.mean(np.asarray(values) ** 2)))
    assert training_plan(data)["training_only_velocity_rms"] == pytest.approx(expected)
