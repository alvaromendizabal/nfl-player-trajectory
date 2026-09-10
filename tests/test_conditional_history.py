"""Chronological, geometric and support contracts for conditional motion history."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.conditional_history import (
    GROUPS,
    NAMES,
    fit_transform,
    observed_rows,
    training_targets,
    transform,
)


def fixture():
    rows = pd.DataFrame(
        {
            "game_id": [2023090100, 2023090101, 2023090200, 2023090300],
            "play_id": [1] * 4,
            "nfl_id": [1, 2, 1, 3],
            "frame_id": [1] * 4,
            "date": [20230901, 20230901, 20230902, 20230903],
            "role": [1] * 4,
            "time_bin": [0] * 4,
            "approach_bin": [0] * 4,
            "speed_bin": [0] * 4,
            "phase_bin": [0] * 4,
            "ux": [0.6] * 4,
            "uy": [0.8] * 4,
        }
    )
    return (
        rows,
        np.array([[2.0, 1.0], [4.0, -1.0], [3.0, 2.0], [np.nan, np.nan]]),
        np.array([True, True, True, False]),
    )


def test_entire_date_and_future_outcomes_are_excluded():
    rows, labels, train = fixture()
    values, state = fit_transform(rows, labels, train)
    np.testing.assert_array_equal(values[:2], 0)
    changed = labels.copy()
    changed[2] = 1e9
    actual, _ = fit_transform(rows, changed, train)
    np.testing.assert_array_equal(actual[:3], values[:3])
    changed = labels.copy()
    changed[~train] = 1e9
    actual, recovered = fit_transform(rows, changed, train)
    np.testing.assert_array_equal(actual, values)
    assert recovered == state


def test_known_shrinkage_and_unseen_context_fallback():
    rows, labels, train = fixture()
    _, state = fit_transform(rows, labels, train)
    query = rows.iloc[[-1]].copy()
    query["approach_bin"] = 99
    values = transform(query, state)[0]
    mean = labels[train].sum(0) / 23
    expected = [mean[0] * 0.6 - mean[1] * 0.8, mean[0] * 0.8 + mean[1] * 0.6]
    np.testing.assert_allclose(values[:2], expected)
    np.testing.assert_allclose(values[5:7], expected)
    assert values[7] == 0 and values[8] == 0


def test_multiple_frames_are_one_player_play_context_observation():
    rows, labels, train = fixture()
    expected, state = fit_transform(rows, labels, train)
    duplicate = rows.iloc[[0]].copy()
    duplicate["frame_id"] = 2
    actual, repeated = fit_transform(
        pd.concat([rows, duplicate], ignore_index=True),
        np.vstack([labels, labels[0]]),
        np.append(train, True),
    )
    assert state == repeated
    np.testing.assert_array_equal(actual[:4], expected)


def test_row_order_and_json_recovery_are_invariant():
    rows, labels, train = fixture()
    expected, state = fit_transform(rows, labels, train)
    order = [3, 1, 2, 0]
    actual, permuted = fit_transform(
        rows.iloc[order].reset_index(drop=True), labels[order], train[order]
    )
    np.testing.assert_array_equal(actual, expected[order])
    assert permuted == state
    before = copy.deepcopy(state)
    np.testing.assert_array_equal(
        transform(rows.iloc[[3]], json.loads(json.dumps(state))), expected[[3]]
    )
    assert before == state


def test_reflecting_history_and_query_preserves_scalar_features():
    rows, labels, train = fixture()
    expected, _ = fit_transform(rows, labels, train)
    rows["uy"] *= -1
    labels[:, 1] *= -1
    actual, _ = fit_transform(rows, labels, train)
    parity = np.tile([1, -1, 1, 1, 1], len(GROUPS))
    np.testing.assert_array_equal(actual, expected * parity)


def sample():
    return {
        "player": np.zeros(2, dtype=int),
        "time": np.array([0.1, 0.2]),
        "horizon": np.array([2]),
        "static": np.array([[0.0, 0.0, 0.3, 0.4]]),
        "velocity": np.array([[3.0, 4.0]]),
        "role": np.array([1]),
        "keys": np.array([[2023090100, 1, 9, 1], [2023090100, 1, 9, 2]]),
        "baseline": np.zeros((2, 2)),
        "truth": np.array([[0.6, 0.8], [1.2, 1.6]]),
        "split": "train",
    }


def test_covariates_ignore_labels_and_baseline_and_targets_have_known_units():
    source = sample()
    rows = observed_rows(source)
    np.testing.assert_allclose(training_targets(source, rows), [[5.0, 0.0], [5.0, 0.0]], atol=1e-12)
    source["baseline"] += 100
    source["truth"] -= 100
    np.testing.assert_allclose(training_targets(source, rows), [[5.0, 0.0], [5.0, 0.0]], atol=1e-10)
    del source["truth"], source["baseline"]
    pd.testing.assert_frame_equal(observed_rows(source), rows)


def test_validation_outcomes_cannot_be_extracted_for_history():
    source = sample()
    source["split"] = "validation"
    with pytest.raises(ValueError, match="training"):
        training_targets(source, observed_rows(source))


def test_bad_time_partition_and_nonfinite_training_labels_fail():
    rows, labels, train = fixture()
    rows.loc[3, "date"] = 20230901
    with pytest.raises(ValueError, match="follow"):
        fit_transform(rows, labels, train)
    rows.loc[3, "date"] = 20230903
    labels[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_transform(rows, labels, train)
    assert len(NAMES) == len(set(NAMES)) == 20
