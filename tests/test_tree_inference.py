"""Independent numerical and integrity contracts for portable tree inference."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from nfl_trajectory.tree_inference import tree_correction


def model() -> dict:
    tree = {
        "value": [0.0, -2.0, 3.0], "feature": [0, 0, 0],
        "threshold": [0.5, 0.0, 0.0], "left": [1, 0, 0],
        "right": [2, 0, 0], "leaf": [False, True, True],
    }
    return {"features": ["x"], "initial": [1.0, -1.0], "axes": [[tree], [tree, tree]]}


def test_tree_threshold_equality_and_accumulation() -> None:
    x = np.array([[-1.0], [0.5], [np.nextafter(0.5, np.inf)], [3.0]])
    expected = np.array([[-1.0, -5.0], [-1.0, -5.0], [4.0, 5.0], [4.0, 5.0]])
    np.testing.assert_array_equal(tree_correction(x, model()), expected)


def test_float32_inputs_use_raw_double_threshold_comparisons() -> None:
    fitted = model()
    fitted["axes"][0][0]["threshold"][0] = 0.499999999999
    np.testing.assert_array_equal(tree_correction(np.array([[0.5]], dtype=np.float32), fitted),
                                  np.array([[4.0, 5.0]]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        tree_correction(np.array([[value]]), model())


@pytest.mark.parametrize("fault", ["cycle", "child", "feature", "shape", "parameter"])
def test_corrupt_trees_are_rejected(fault: str) -> None:
    fitted = copy.deepcopy(model())
    tree = fitted["axes"][0][0]
    if fault == "cycle":
        tree["left"][0] = 0
    elif fault == "child":
        tree["right"][0] = 9
    elif fault == "feature":
        tree["feature"][0] = 1
    elif fault == "shape":
        tree["value"].pop()
    else:
        tree["value"][1] = float("nan")
    with pytest.raises(ValueError):
        tree_correction(np.array([[1.0]]), fitted)
