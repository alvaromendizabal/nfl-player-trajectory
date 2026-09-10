"""The comparison must retain the complete official forecast row set."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.motion import KEYS

SPEC = importlib.util.spec_from_file_location(
    "temporal_evaluator", Path(__file__).resolve().parents[1] / "scripts/evaluate_temporal.py"
)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonfinite"])
def test_comparison_rejects_invalid_forecast_rows(corruption):
    original = pd.DataFrame([[2023090700, 1, 1, f] for f in (1, 51, 60)], columns=KEYS)
    original["dx"], original["dy"] = 0.3, -0.4
    other = original.copy()
    if corruption == "missing":
        other = other.iloc[:2]
    elif corruption == "duplicate":
        other = pd.concat([other, other.iloc[:1]])
    else:
        other.loc[0, "dx"] = np.inf
    with pytest.raises(ValueError):
        evaluator.aligned_errors(original, other)


def test_comparison_aligns_shuffled_rows_without_changing_errors():
    original = pd.DataFrame([[2023090700, 1, 1, f] for f in (1, 51, 60)], columns=KEYS)
    original["dx"], original["dy"] = [1.0, 2.0, 3.0], -0.4
    first, second = evaluator.aligned_errors(original, original.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(first, second)
