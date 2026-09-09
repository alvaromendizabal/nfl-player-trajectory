"""The extra stopping check cannot use development success or conceal a weak fold."""

from copy import deepcopy

import pytest

from nfl_trajectory.simplification import FOLDS, decision


def examples():
    return [
        {
            "fold": fold,
            "reference": {"coordinate_rmse_yards": 1.0},
            "metrics": {"coordinate_rmse_yards": 0.99},
            "rows": rows,
            "parent_feature_count": 100,
            "feature_count": 55,
            "holdout_evaluation": "not_run",
        }
        for fold, rows in zip(FOLDS, [10, 20, 30, 1000], strict=True)
    ]


def test_development_cannot_change_the_combined_omission_decision():
    original = examples()
    changed = deepcopy(original)
    changed[-1]["metrics"]["coordinate_rmse_yards"] = 10.0
    assert decision(original) == decision(changed)
    assert decision(original)["requires_smaller_representation_followup"]


def test_one_harmed_fold_blocks_an_apparently_good_pooled_combination():
    folds = examples()
    folds[0]["metrics"]["coordinate_rmse_yards"] = 1.02
    for fold in folds[1:3]:
        fold["metrics"]["coordinate_rmse_yards"] = 0.98
    assert decision(folds)["pooled_relative_gain"] > 0.005
    assert not decision(folds)["requires_smaller_representation_followup"]


def test_small_consistent_gains_remain_below_the_recorded_stopping_tolerance():
    folds = examples()
    for fold in folds:
        fold["metrics"]["coordinate_rmse_yards"] = 0.996
    assert not decision(folds)["requires_smaller_representation_followup"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
def test_invalid_metrics_cannot_supply_a_completion_decision(bad):
    folds = examples()
    folds[1]["metrics"]["coordinate_rmse_yards"] = bad
    with pytest.raises(ValueError, match="invalid metrics"):
        decision(folds)


def test_missing_fold_or_contaminated_scope_is_rejected():
    with pytest.raises(ValueError, match="all four"):
        decision(examples()[:-1])
    folds = examples()
    folds[-1]["holdout_evaluation"] = "run"
    with pytest.raises(ValueError, match="scope"):
        decision(folds)
