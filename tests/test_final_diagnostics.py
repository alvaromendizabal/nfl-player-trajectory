"""Small exact examples verify error concentration and reference coverage."""

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.final_diagnostics import summarize_errors


def errors_fixture():
    primary = pd.DataFrame(
        {
            "game_id": [1, 1, 2],
            "play_id": [1, 1, 1],
            "nfl_id": [1, 1, 2],
            "frame_id": [1, 31, 1],
            "dx": [1.0, 3.0, 2.0],
            "dy": [0.0, 0.0, 0.0],
        }
    )
    return pd.concat(
        [
            primary.assign(scenario=name, dx=primary.dx * scale)
            for name, scale in (
                ("complete", 1),
                ("constant_velocity", 2),
                ("role_ridge", 2),
                ("without_telemetry", 1),
            )
        ],
        ignore_index=True,
    )


def test_error_mass_and_baseline_comparisons_are_exact():
    report = summarize_errors(errors_fixture().sample(frac=1, random_state=7))
    assert report["coordinate_rmse_yards"] == pytest.approx(np.sqrt(14 / 6))
    assert report["top_game_squared_error_share"] == pytest.approx(10 / 14)
    assert sum(row["squared_error_share"] for row in report["forecast_bins"]) == pytest.approx(1)
    assert sum(row["row_share"] for row in report["forecast_bins"]) == pytest.approx(1)
    assert report["forecast_bins"][1]["rows"] == 1
    assert report["reference_comparisons"][0]["rmse_reduction_percent"] == 50
    assert report["reference_comparisons"][0]["games_improved"] == 2
    assert report["reference_comparisons"][2]["games_tied"] == 2


def test_incomplete_reference_cannot_improve_a_diagnostic():
    errors = errors_fixture().drop(index=3)
    with pytest.raises(ValueError, match="exactly the primary forecast keys"):
        summarize_errors(errors)
