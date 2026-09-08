"""Research reporting contracts use recorded evidence, not invented performance."""

import ast
import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from nfl_trajectory.research import error_budget, load_evidence, validate_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return json.loads((ROOT / "docs/results/feature_summary.json").read_text())


def test_recorded_champion_and_metric_reconciliation(evidence):
    validate_summary(evidence)
    assert evidence["selected_model"] == "landing_ridge"
    assert evidence["models"][0]["coordinate_rmse_yards"] == pytest.approx(0.9268683891794343)
    for dimension in ("role", "forecast_second"):
        budget = error_budget(evidence, dimension)
        assert budget.rows.sum() == 67857
        assert budget.squared_error_share_percent.sum() == pytest.approx(100)
    defense = error_budget(evidence, "role").set_index("value").loc["Defensive Coverage"]
    assert defense.squared_error_share_percent == pytest.approx(89.04839371915061)


@pytest.mark.parametrize(
    "field,value",
    [("status", "running"), ("screening_split", "validation"),
     ("holdout_evaluation", "scored"), ("split", "holdout"),
     ("selected_model", "constant_velocity")],
)
def test_invalid_completion_evidence_is_rejected(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_summary(evidence)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_nonfinite_and_negative_scores_are_rejected(evidence, value):
    evidence["models"][0]["coordinate_rmse_yards"] = value
    with pytest.raises(ValueError):
        validate_summary(evidence)


def test_missing_rows_or_inconsistent_slice_scores_are_rejected(evidence):
    altered = copy.deepcopy(evidence)
    altered["slices"][0]["rows"] += 1
    with pytest.raises(ValueError):
        validate_summary(altered)
    altered = copy.deepcopy(evidence)
    altered["slices"][0]["coordinate_rmse_yards"] += 0.1
    with pytest.raises(ValueError):
        validate_summary(altered)


def test_published_selection_study_is_bound_to_actual_summary(tmp_path):
    import shutil

    shutil.copytree(ROOT / "docs/results", tmp_path / "docs/results")
    summary, study, label = load_evidence(tmp_path)
    assert label == "Published real-data experiment snapshot"
    assert study["training_rows"] == 395813
    assert study["overlap_count"] == 41
    assert study["landing_ball_ux_count"] == 31
    assert "fraction__lateral_speed" in study["removed_from_landing"]
    assert summary["candidate_features"] == 2843


def test_default_notebooks_neither_train_nor_export_implicitly():
    expected = {
        "01_data_analysis.ipynb": "RUN_FEATURE_EXPERIMENT",
        "02_motion_benchmarks.ipynb": "GENERATE_EXPORT",
    }
    for filename, switch in expected.items():
        notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        assignments = [
            node for cell in notebook.cells if cell.cell_type == "code"
            for node in ast.walk(ast.parse(cell.source))
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == switch for t in node.targets)
        ]
        assert len(assignments) == 1
        assert isinstance(assignments[0].value, ast.Constant)
        assert assignments[0].value.value is False


def test_quality_export_does_not_overwrite_owner_artifact():
    source = (ROOT / "scripts/quality.py").read_text()
    assert "artifacts/quality/exports/submission.ipynb" in source
    assert "artifacts/kaggle/submission.ipynb" not in source


def test_unknown_error_budget_dimension_fails(evidence):
    with pytest.raises(ValueError):
        error_budget(evidence, "player_identity")


def notebook_runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("review_controls_test", ROOT / "scripts/notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("switch", ["RUN_FEATURE_EXPERIMENT", "GENERATE_EXPORT"])
def test_automatic_render_refuses_enabled_owner_switch(switch):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f"{switch} = True")])
    with pytest.raises(ValueError, match="Turn off manual"):
        notebook_runner().validate_review_controls(notebook)


@pytest.mark.parametrize("switch", ["RUN_FEATURE_EXPERIMENT", "GENERATE_EXPORT"])
def test_automatic_render_accepts_default_owner_switch(switch):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(f"{switch} = False")])
    notebook_runner().validate_review_controls(notebook)
