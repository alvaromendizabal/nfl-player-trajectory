"""Verify final model lineage, chronological inference, and standalone availability behavior."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_feature_research import research_project as research_project
from test_final_fit import (
    ENVIRONMENT,
    constant_convert,
    constant_fit,
)
from test_final_fit import (
    fitting_project as fitting_project,
)

from nfl_trajectory.final_fit import fit_models
from nfl_trajectory.final_inference import (
    final_bundle,
    final_variant,
    predict_final,
    standalone_source,
)
from nfl_trajectory.models import predict
from nfl_trajectory.motion import KEYS
from nfl_trajectory.runtime import Run


@pytest.fixture
def inference_project(fitting_project):
    root, _, _ = fitting_project
    with Run(root, "final-inference-fixture") as run:
        fit_models(root, run, constant_fit, constant_convert, ENVIRONMENT)
    return root


def test_final_bundle_requires_complete_current_fit_evidence(inference_project):
    root = inference_project
    bundle = final_bundle(root)
    assert bundle["kind"] == "final_trajectory"
    assert bundle["fitted_features"] == {"without_metadata": 13, "without_optional_inputs": 12}
    assert set(bundle["trees"]) == set(bundle["fitted_features"])
    for relative in (
        "artifacts/final/models/without_metadata/x.pkl",
        "artifacts/final/models/without_optional_inputs/tree.json",
        "artifacts/final/models/without_metadata/training_predictions.npz",
        "artifacts/final/preprocessing/history_model.json",
        "scripts/fit_final.py",
    ):
        path = root / relative
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with pytest.raises(ValueError):
                final_bundle(root)
        finally:
            path.write_bytes(original)
    path = root / "artifacts/final/models/summary.json"
    report = json.loads(path.read_text())
    report["profiles"][0]["training_rows"] -= 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="report does not match"):
        final_bundle(root)


def test_final_predictor_uses_frozen_history_and_future_dates(inference_project):
    root = inference_project
    bundle = final_bundle(root)
    path = sorted((root / "data/raw/train").glob("input_*.csv"))[-1]
    observed = pd.read_csv(path)
    targets = observed.groupby(KEYS[:3], sort=False).tail(1)[KEYS].copy()
    targets["frame_id"] = 1
    with pytest.raises(ValueError, match="strictly follow"):
        predict_final(observed, targets, bundle)
    observed["game_id"] = targets["game_id"] = 2025010100
    expected = predict_final(observed, targets, bundle)
    poisoned = targets.assign(x=np.inf, y=np.nan).sample(frac=1, random_state=2026)
    actual = predict_final(observed.sample(frac=1, random_state=2026), poisoned, bundle)
    pd.testing.assert_frame_equal(actual[KEYS], poisoned[KEYS].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        actual.sort_values(KEYS).reset_index(drop=True),
        expected.sort_values(KEYS).reset_index(drop=True),
    )
    before = copy.deepcopy(bundle["history"])
    predict_final(observed, targets, bundle, cold_history=True)
    assert bundle["history"] == before
    assert np.isfinite(expected[["x", "y"]]).all().all()
    source = Path(__file__).resolve().parents[1]
    namespace = {}
    exec(compile(standalone_source(source, bundle), "final-standalone.py", "exec"), namespace)
    pd.testing.assert_frame_equal(
        namespace["final_predict"](observed, targets, namespace["FINAL_MODEL"]), expected
    )


def test_final_availability_routes_partial_and_nonfinite_telemetry(inference_project):
    root = inference_project
    bundle = final_bundle(root)
    observed = pd.read_csv(sorted((root / "data/raw/train").glob("input_*.csv"))[-1])
    targets = observed.groupby(KEYS[:3], sort=False).tail(1)[KEYS].copy()
    observed["game_id"] = targets["game_id"] = 2025010100
    targets["frame_id"] = 1
    for column in ("s", "a", "o", "dir"):
        observed[column] = 0.0
    bundle["trees"]["without_metadata"]["initial"] = [1.0, 2.0]
    bundle["trees"]["without_optional_inputs"]["initial"] = [7.0, 8.0]
    assert final_variant(observed) == "without_metadata"
    reference = predict_final(observed.drop(columns=["s", "a", "o", "dir"]), targets, bundle)
    for column in ("s", "a", "o", "dir"):
        missing = observed.drop(columns=column)
        assert final_variant(missing) == "without_optional_inputs"
        pd.testing.assert_frame_equal(predict_final(missing, targets, bundle), reference)
        for value in (np.nan, np.inf, -np.inf):
            partial = observed.copy()
            partial.loc[partial.index[0], column] = value
            assert final_variant(partial) == "without_optional_inputs"
            pd.testing.assert_frame_equal(predict_final(partial, targets, bundle), reference)
    complete = predict_final(observed, targets, bundle)
    assert not np.array_equal(complete[["x", "y"]], reference[["x", "y"]])
    baseline = predict(observed, targets, "role_ridge", bundle["baseline"])
    assert np.isfinite(baseline[["x", "y"]]).all().all()
