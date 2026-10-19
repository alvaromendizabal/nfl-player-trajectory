"""Retained final scoring contracts, independent of retired fitting campaigns."""

import math

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.final_evaluation import bootstrap_scores, error_metrics, score_predictions
from nfl_trajectory.motion import KEYS


def fixture():
    return pd.DataFrame(
        [[1, 1, 1, 1, 0.0, 0.0], [2, 1, 1, 1, 2.0, 3.0], [2, 1, 1, 2, 3.0, 4.0]],
        columns=KEYS + ["x", "y"],
    )


def test_outer_join_and_coordinate_denominator():
    truth = fixture()
    keys = truth[KEYS].iloc[::-1].reset_index(drop=True)
    values = keys.merge(truth, on=KEYS)[["x", "y"]].to_numpy() + [3, 4]
    errors = score_predictions(truth, keys, values)
    pd.testing.assert_frame_equal(errors[KEYS], keys)
    assert error_metrics(errors)["coordinate_rmse_yards"] == pytest.approx(math.sqrt(12.5))
    for bad in (truth.iloc[1:], pd.concat([truth, truth.iloc[:1]])):
        with pytest.raises(ValueError):
            score_predictions(bad, keys, values)
    with pytest.raises(ValueError):
        score_predictions(truth, keys, np.full((3, 2), np.nan))


def test_whole_game_bootstrap_pools_sse_and_preserves_replication():
    errors = fixture()[KEYS].assign(dx=[10.0, 0.0, 0.0], dy=0.0)
    scores = bootstrap_scores(errors, repeats=200, seed=7)
    np.testing.assert_array_equal(scores, bootstrap_scores(errors, repeats=200, seed=7))
    allowed = (0.0, math.sqrt(100 / 6), math.sqrt(100 / 2))
    assert all(any(math.isclose(x, y, abs_tol=1e-12) for y in allowed) for x in scores)
    with pytest.raises(ValueError):
        bootstrap_scores(errors.iloc[:0])
