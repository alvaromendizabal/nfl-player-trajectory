"""Evidence rules: exact forecast coverage, pooled error and consistent fold gains."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.motion import KEYS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "temporal_research_evaluation", SCRIPTS / "evaluate_temporal_research.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_pooled_comparison_weights_coordinates_and_preserves_pairing():
    # A large difficult game must not have the same influence as one easy frame.
    keys = [[2023090700, 1, 1, 1]] + [[2023091400, 1, 1, i] for i in range(1, 10)]
    reference = pd.DataFrame(keys, columns=KEYS)
    reference[["dx", "dy"]] = np.array([[1, 1]] + [[3, 3]] * 9)
    challenger = reference.copy()
    challenger[["dx", "dy"]] *= 0.5
    result = evaluation.compare(challenger.sample(frac=1, random_state=2), reference)
    assert result["coordinate_rmse_yards"] == pytest.approx(np.sqrt(8.2) / 2)
    assert result["relative_rmse_reduction"] == pytest.approx(0.5)
    assert result["paired_game_bootstrap_delta_ci95"][1] < 0
    with pytest.raises(ValueError, match="identical forecast keys"):
        evaluation.compare(challenger.iloc[:-1], reference)


def test_positive_pooled_result_cannot_hide_a_failed_fold_or_incomplete_study():
    comparison = {"rmse_difference": -0.02}
    rows = [
        {
            "full_vs_without_target_statistics": dict(comparison),
            "equal_blend_vs_tree": dict(comparison),
        }
        for _ in range(3)
    ]
    pooled = {
        "full_vs_without_target_statistics": {"paired_game_bootstrap_delta_ci95": [-0.03, -0.01]},
        "equal_blend_vs_tree": {
            "paired_game_bootstrap_delta_ci95": [-0.03, -0.01],
            "relative_rmse_reduction": 0.03,
        },
    }
    assert all(evaluation.gates(rows, pooled).values())
    rows[1]["full_vs_without_target_statistics"]["rmse_difference"] = 0.001
    assert not evaluation.gates(rows, pooled)["consistent_target_statistics_benefit"]
    assert not any(evaluation.gates(rows[:2], pooled).values())
    pooled["equal_blend_vs_tree"]["relative_rmse_reduction"] = 0.009
    assert not evaluation.gates(rows, pooled)["equal_blend_ready_for_inference_research"]
