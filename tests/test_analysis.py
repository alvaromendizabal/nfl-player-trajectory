"""Contract tests for the actual published experiment and its derived diagnostics."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nfl_trajectory.analysis import (
    association_figure,
    budget_figure,
    comparison,
    error_budget,
    experiment_figure,
    figure_png,
    load_evidence,
    validate_summary,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    return json.loads((ROOT / "docs/results/feature_summary.json").read_text())


def test_measured_evidence_has_matching_source_and_baseline(tmp_path):
    # Do not let a user's valid local experiment change this published-snapshot test.
    shutil.copytree(ROOT / "docs/results", tmp_path / "docs/results")
    summary, audit, label = load_evidence(tmp_path)
    assert label == "Verified published experiment"
    assert summary["selected_model"] == "landing_ridge"
    assert audit["shared_landing_interaction"] == 41
    assert 41 + len(audit["landing_not_in_interaction"]) == 64
    assert 41 + len(audit["interaction_not_in_landing"]) == 64


def test_actual_improvement_and_error_budgets_reconcile(evidence):
    rows = comparison(evidence)
    assert rows.iloc[0].rmse_reduction_vs_baseline_percent == pytest.approx(6.336132923852366)
    for dimension in ("role", "forecast_second"):
        result = error_budget(evidence, dimension)
        assert result.rows.sum() == 67857
        assert result.squared_error_share_percent.sum() == pytest.approx(100)
        pooled = np.sqrt(result.squared_error_yards2.sum() / (2 * result.rows.sum()))
        assert pooled == pytest.approx(0.9268683891794343)
    time = error_budget(evidence, "forecast_second")
    assert time.loc[time.value.astype(int) > 1, "squared_error_share_percent"].sum() == pytest.approx(83.5739574691)


@pytest.mark.parametrize("field,value", [("status", "running"), ("split", "holdout"),
                                         ("screening_split", "validation"),
                                         ("holdout_evaluation", "completed"),
                                         ("selected_model", "interaction_ridge")])
def test_incomplete_or_misrepresented_experiments_fail(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError):
        validate_summary(evidence)


def test_mismatched_slice_population_fails(evidence):
    evidence["slices"][0]["rows"] += 1
    with pytest.raises(ValueError, match="population"):
        validate_summary(evidence)


def test_slice_score_cannot_drift_from_pooled_score(evidence):
    evidence["slices"][0]["coordinate_rmse_yards"] *= 2
    with pytest.raises(ValueError, match="reconcile"):
        validate_summary(evidence)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_scores_fail(evidence, value):
    evidence["models"][0]["coordinate_rmse_yards"] = value
    with pytest.raises(ValueError):
        validate_summary(evidence)


def test_analysis_does_not_modify_measurements(evidence):
    before = copy.deepcopy(evidence)
    comparison(evidence)
    error_budget(evidence, "role")
    assert evidence == before


@pytest.mark.parametrize("builder", [experiment_figure, association_figure,
                                       lambda x: budget_figure(x, "role"),
                                       lambda x: budget_figure(x, "forecast_second")])
def test_figures_are_rendered_portable_pngs(evidence, builder):
    png = figure_png(builder(evidence))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 10000


def test_publication_never_opts_into_training_or_submission_creation():
    import nbformat

    for name, flag in [("00_project_readiness.ipynb", "RUN_FEATURE_EXPERIMENT"),
                       ("02_motion_benchmarks.ipynb", "CREATE_SUBMISSION")]:
        notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
        source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        assert f"{flag} = False" in source
        assert 'os.getenv("NFL_NOTEBOOK_AUTORUN") != "1"' in source
    assert '"NFL_NOTEBOOK_AUTORUN": "1"' in (ROOT / "scripts/notebooks.py").read_text()


def test_quality_exports_do_not_target_the_user_artifact():
    source = (ROOT / "scripts/quality.py").read_text()
    assert '"artifacts/kaggle/submission.ipynb"' not in source
    assert '"artifacts/quality/submission.ipynb"' in source
