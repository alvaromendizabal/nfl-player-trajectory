"""Test final data boundaries, frozen schemas, provenance, and independent fit recovery."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_feature_research import research_project as research_project

from nfl_trajectory.context_experiment import prepare_context
from nfl_trajectory.context_features import build_context_features, context_catalog
from nfl_trajectory.feature_candidates import research_catalog
from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
from nfl_trajectory.feature_experiment import load_week
from nfl_trajectory.feature_research import history_rows, verify_inputs
from nfl_trajectory.final_features import (
    feature_groups,
    feature_matrix,
    load_preprocessing,
    materialize,
    validate_observed_features,
)
from nfl_trajectory.final_fit import fit_models, validate_profiles
from nfl_trajectory.final_protocol import digest, freeze_protocol, load_protocol
from nfl_trajectory.final_training import preprocess
from nfl_trajectory.representation_features import build_representation, representation_catalog
from nfl_trajectory.runtime import Run, sha256

ENVIRONMENT = {"scikit-learn": "1.8.0", "scope": "injected constant estimator fixture"}


class ConstantModel:
    def __init__(self, matrix, target):
        self.n_features_in_ = matrix.shape[1]
        self.mean = float(np.mean(target))

    def predict(self, matrix):
        return np.full(len(matrix), self.mean)


def constant_fit(matrix, target, settings):
    return ConstantModel(matrix, target)


def constant_convert(models, names):
    return {
        "features": [],
        "initial": [model.mean for model in models],
        "axes": [[], []],
        "screened_feature_count": len(names),
    }


@pytest.fixture
def fitting_project(research_project, monkeypatch):
    root = research_project
    source = Path(__file__).resolve().parents[1]
    for name in (
        "scripts/refit_final.py",
        "scripts/fit_final.py",
        "scripts/fit_final.py.lock",
        "scripts/prepare_tree.py",
        "src/nfl_trajectory/final_features.py",
        "src/nfl_trajectory/final_fit.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    with Run(root, "context-fixture") as run:
        caches = prepare_context(root, run)
    _, inputs = verify_inputs(root)
    inputs.update(
        {
            str(p.relative_to(root)): sha256(p)
            for p in (root / "artifacts/context/weeks").rglob("*.npz")
        }
    )
    splits = pd.read_csv(root / "artifacts/game_splits.csv")
    names = []
    for catalog in (research_catalog(), context_catalog(), representation_catalog()):
        names.extend(
            [
                name
                for name in catalog.feature
                if not metadata_dependent(name) and not telemetry_dependent(name)
            ][:4]
        )
    names.append(
        next(
            name
            for name in research_catalog().feature
            if telemetry_dependent(name) and not metadata_dependent(name)
        )
    )
    weeks = []
    for cache in caches:
        with np.load(cache, allow_pickle=False) as saved:
            keys = saved["keys"].astype("<i8")
        weeks.append(
            {
                "week": cache.parent.name,
                "rows": len(keys),
                "ordered_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
            }
        )
    protocol = freeze_protocol(
        root,
        {
            "status": "prepared",
            "inputs": inputs,
            "contract": {
                "training_games": splits.loc[~splits.split.eq("holdout"), "game_id"].tolist(),
                "holdout_games": splits.loc[splits.split.eq("holdout"), "game_id"].tolist(),
                "availability_profile_features": {
                    "without_metadata": names,
                    "without_optional_inputs": [n for n in names if not telemetry_dependent(n)],
                },
                "estimator": {"settings": {"fixture": True}},
            },
            "training_inventory": {
                "training_rows": 210,
                "training_trajectories": 21,
                "weeks": weeks,
            },
        },
    )
    with Run(root, "preprocessing-fixture") as run:
        preprocess(root, run)
    monkeypatch.setattr("nfl_trajectory.final_fit.memory_budget_gib", lambda: 128)
    # The fixture needs little storage; leave the real materialization reserve intact.
    monkeypatch.setattr(
        "nfl_trajectory.final_features.shutil.disk_usage",
        lambda path: shutil._ntuple_diskusage(10**10, 0, 10**10),
    )
    return root, caches, protocol


def test_final_materialization_matches_raw_features_and_complete_partition(fitting_project):
    root, caches, protocol = fitting_project
    with Run(root, "matrix-test") as run:
        matrix, arrays = materialize(root, caches, protocol, "fixture", run)
    assert matrix.shape == (210, 13) and matrix.dtype == np.float32
    assert set(arrays["keys"][:, 0]) == set(protocol["contract"]["training_games"])
    assert not set(arrays["keys"][:, 0]) & set(protocol["contract"]["holdout_games"])
    np.testing.assert_allclose(
        arrays["residual"], (arrays["truth"] - arrays["baseline"]) * arrays["sign"]
    )
    bank, targets, _ = load_week(caches[-1])
    observed = pd.read_csv(root / "data/raw/train" / (caches[-1].parent.name + ".csv"))
    context = build_context_features(observed, bank.state)
    history = pd.read_csv(root / "artifacts/final/preprocessing/history.csv")
    routes = json.loads((root / "artifacts/final/preprocessing/routes.json").read_text())
    names = protocol["contract"]["availability_profile_features"]["without_metadata"]
    raw = feature_matrix(
        bank,
        targets,
        history_rows(history, targets),
        context,
        build_representation(bank, routes),
        names,
    )
    np.testing.assert_array_equal(matrix[-len(targets) :], raw)
    shuffled = list(reversed(names))
    np.testing.assert_array_equal(
        feature_matrix(
            bank,
            targets,
            history_rows(history, targets),
            context,
            build_representation(bank, routes),
            shuffled,
        ),
        raw[:, ::-1],
    )
    with Run(root, "raw-parity-test") as run:
        report = validate_observed_features(root, caches, run)
    assert report["rows"] == 210 and len(report["weeks"]) == 7
    assert all(max(row["max_absolute_feature_difference"].values()) == 0 for row in report["weeks"])


def test_final_matrix_rejects_inventory_order_changes(fitting_project):
    root, caches, protocol = fitting_project
    protocol["training_inventory"]["weeks"][0]["ordered_keys_sha256"] = "0" * 64
    with Run(root, "matrix-test") as run, pytest.raises(ValueError, match="row order"):
        materialize(root, caches, protocol, "fixture", run)
    assert json.loads((root / ".state/final-matrix.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("names", [[], ["unknown"], ["x", "x"]])
def test_invalid_feature_schemas_are_rejected(names):
    with pytest.raises(ValueError, match="feature names"):
        feature_groups(names)


def test_final_fallback_requires_frozen_subset_order(fitting_project):
    _, _, protocol = fitting_project
    profiles = protocol["contract"]["availability_profile_features"]
    validate_profiles(profiles)
    profiles["without_optional_inputs"].reverse()
    with pytest.raises(ValueError, match="subset and order"):
        validate_profiles(profiles)


def test_preprocessing_content_and_source_changes_fail_closed(fitting_project):
    root, _, protocol = fitting_project
    load_preprocessing(root, protocol)
    path = root / "artifacts/final/preprocessing/routes.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="output changed"):
        load_preprocessing(root, protocol)


def test_complete_final_fit_recovers_coordinate_and_reuses_without_matrix(
    fitting_project, monkeypatch
):
    root, _, _ = fitting_project
    attempts = 0

    def interrupted(matrix, target, settings):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise RuntimeError("Controlled second-coordinate interruption")
        return constant_fit(matrix, target, settings)

    with Run(root, "fit-test") as run, pytest.raises(RuntimeError, match="Controlled"):
        fit_models(root, run, interrupted, constant_convert, ENVIRONMENT)
    path = root / "artifacts/final/models/without_metadata/x.pkl"
    before = sha256(path), path.stat().st_mtime_ns
    with Run(root, "fit-test") as run:
        report = fit_models(root, run, interrupted, constant_convert, ENVIRONMENT)
    assert (sha256(path), path.stat().st_mtime_ns) == before
    assert attempts == 5 and report["final_fit"] is True
    assert report["holdout_evaluation"] == "not_run" and report["model_sealed"] is False
    assert [r["fitted_features"] for r in report["profiles"]] == [13, 12]
    assert all(
        r["training_rows"] == 210 and r["portable_max_absolute_difference"] == 0
        for r in report["profiles"]
    )

    def unnecessary(*args):
        raise AssertionError("Completed fits must reuse without materialization")

    monkeypatch.setattr("nfl_trajectory.final_fit.materialize", unnecessary)
    with Run(root, "fit-test") as run:
        assert fit_models(root, run, unnecessary, constant_convert, ENVIRONMENT) == report
    assert '"stage_started"' not in run.log_path.read_text()


def test_portable_mismatch_does_not_mark_export_complete(fitting_project):
    root, _, _ = fitting_project

    def changed(models, names):
        result = constant_convert(models, names)
        result["initial"][0] += 1
        return result

    with Run(root, "fit-test") as run, pytest.raises(AssertionError):
        fit_models(root, run, constant_fit, changed, ENVIRONMENT)
    receipt = root / ".state/final-export-without_metadata.json"
    assert json.loads(receipt.read_text())["status"] == "failed"
    assert (root / ".state/final-fit-without_metadata-1.json").exists()


def test_recovered_axis_invalidates_its_downstream_export(fitting_project):
    root, _, _ = fitting_project
    with Run(root, "fit-test") as run:
        before = fit_models(root, run, constant_fit, constant_convert, ENVIRONMENT)
    folder = root / "artifacts/final/models/without_metadata"
    (folder / "x.pkl").write_bytes(b"corrupted local checkpoint")
    y_before = sha256(folder / "y.pkl"), (folder / "y.pkl").stat().st_mtime_ns

    def changed(matrix, target, settings):
        return ConstantModel(matrix, target + 1)

    with Run(root, "fit-test") as run:
        after = fit_models(root, run, changed, constant_convert, ENVIRONMENT)
    assert (sha256(folder / "y.pkl"), (folder / "y.pkl").stat().st_mtime_ns) == y_before
    assert before["profiles"][0]["export_signature"] != after["profiles"][0]["export_signature"]
    assert before["profiles"][0]["tree_sha256"] != after["profiles"][0]["tree_sha256"]
    assert before["profiles"][1] == after["profiles"][1]


def test_final_fit_rejects_seal_environment_and_changed_plan(fitting_project):
    root, _, protocol = fitting_project
    with Run(root, "fit-test") as run, pytest.raises(ValueError, match="1.8.0"):
        fit_models(root, run, constant_fit, constant_convert, {"scikit-learn": "0"})
    with Run(root, "fit-test") as run:
        fit_models(root, run, constant_fit, constant_convert, ENVIRONMENT)
    plan = root / "artifacts/final/fit_plan.json"
    payload = json.loads(plan.read_text())
    payload["settings"] = {"changed": True}
    plan.write_text(json.dumps(payload))
    with Run(root, "fit-test") as run, pytest.raises(ValueError, match="plan changed"):
        fit_models(root, run, constant_fit, constant_convert, ENVIRONMENT)
    assert digest(load_protocol(root)) == digest(protocol)
    (root / "artifacts/final/model_seal.json").write_text("{}")
    with Run(root, "fit-test") as run, pytest.raises(ValueError, match="sealed"):
        fit_models(root, run, constant_fit, constant_convert, ENVIRONMENT)
