"""Counterfactual leakage checks for matched temporal feature research."""

import numpy as np
import pandas as pd
import pytest
from test_temporal_data import sample

from nfl_trajectory.feature_candidates import HISTORY_NAMES
from nfl_trajectory.motion import ENTITY, KEYS
from nfl_trajectory.temporal_research import (
    TARGET_STATISTICS,
    ablate,
    rebase_sample,
    validate_fold,
)


def test_ablation_keeps_counts_geometry_targets_and_source_immutable():
    source = sample()
    source["static"][:, 10:] = np.arange(10) + 1
    before = {k: v.copy() for k, v in source.items()}
    removed = ablate(source, "without_target_statistics")
    for key in source:
        np.testing.assert_array_equal(source[key], before[key])
        if key != "static":
            np.testing.assert_array_equal(removed[key], source[key])
    retained = [i for i in range(20) if i not in TARGET_STATISTICS]
    np.testing.assert_array_equal(removed["static"][:, retained], source["static"][:, retained])
    assert np.count_nonzero(removed["static"][:, TARGET_STATISTICS]) == 0


@pytest.mark.parametrize(
    "training,evaluation",
    [
        ([2023090700], [2023090700]),
        ([2023090700], [2023090701]),
        ([2023091400], [2023090700]),
        ([2023090700], [2023092100]),
        ([], [2023091400]),
    ],
)
def test_reject_overlap_same_date_reverse_time_and_outer_partition(training, evaluation):
    splits = pd.DataFrame(
        {
            "game_id": [2023090700, 2023090701, 2023091400, 2023092100],
            "split": ["train", "train", "train", "validation"],
        }
    )
    with pytest.raises(ValueError):
        validate_fold({"training_games": training, "evaluation_games": evaluation}, splits)


def test_rebase_discards_full_training_priors_and_targets():
    source = sample()
    targets = pd.DataFrame(source["keys"], columns=KEYS)
    priors = targets[ENTITY].drop_duplicates()
    priors[HISTORY_NAMES] = np.array([[2, 0, 0.1, 0.2, 0.3, 3, 0, 0.4, 0.5, 0.6]])
    baseline = np.tile([48.0, 24.0], (len(targets), 1))
    truth = baseline + 2
    first = rebase_sample(source, targets, baseline, truth, priors, "train")
    source["baseline"][:] = 1e6
    source["truth"][:] = -1e6
    source["static"][:, 10:] = 1e6
    second = rebase_sample(source, targets, baseline, truth, priors, "train")
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first["truth"], 2)
    # Unscored context players have explicit cold priors, never stale label statistics.
    np.testing.assert_array_equal(first["static"][1:, [11, 16]], 1)
    assert np.count_nonzero(first["static"][1:, TARGET_STATISTICS]) == 0


def test_rebase_rejects_reordered_requests():
    source = sample()
    targets = pd.DataFrame(source["keys"], columns=KEYS).iloc[::-1]
    with pytest.raises(ValueError, match="ordered forecast keys"):
        rebase_sample(source, targets, source["baseline"], source["truth"], pd.DataFrame(), "train")
