"""Validate generated submission code in a directory with no Git metadata."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from nfl_trajectory.motion import KEYS, constant_velocity
from nfl_trajectory.report import synthetic_play


@pytest.fixture
def exported_notebook(tmp_path: Path) -> Path:
    source_root = Path(__file__).resolve().parents[1]
    for relative in ["kaggle/export.py", "src/nfl_trajectory/motion.py", "pyproject.toml"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    subprocess.run(
        [sys.executable, "kaggle/export.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_path / "artifacts/kaggle/submission.ipynb"


def test_export_passes_lint_and_format_without_git(exported_notebook: Path) -> None:
    root = exported_notebook.parents[2]
    assert not (root / ".git").exists()
    for arguments in [["check"], ["format", "--check"]]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                *arguments,
                "--no-respect-gitignore",
                str(exported_notebook),
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_exported_predictor_matches_reference_and_preserves_order(exported_notebook: Path) -> None:
    notebook = nbformat.read(exported_notebook, as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    compile("\n".join(cells), "submission.py", "exec")
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    function = next(
        node
        for node in ast.parse(cells[-1]).body
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "predict.py", "exec"), namespace)
    observed, truth = synthetic_play()
    targets = truth[KEYS].sample(frac=1, random_state=3)
    actual = namespace["predict"](targets, observed)
    expected = constant_velocity(observed, targets)[["x", "y"]]
    pd.testing.assert_frame_equal(actual, expected)


def test_trained_export_matches_package_and_passes_lint(tmp_path: Path) -> None:
    import json

    from nfl_trajectory.models import design, fit_statistics, predict, sufficient_statistics

    source_root = Path(__file__).resolve().parents[1]
    for relative in [
        "kaggle/export.py",
        "src/nfl_trajectory/motion.py",
        "src/nfl_trajectory/models.py",
        "pyproject.toml",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    observed, truth = synthetic_play()
    observed["ball_land_x"] = 44.0
    observed["ball_land_y"] = 28.0
    observed["num_frames_output"] = 20
    observed["player_role"] = "Targeted Receiver"
    state, features = design(observed, truth[KEYS])
    fitted = fit_statistics([sufficient_statistics(state, features, truth)])
    weights = tmp_path / "model.json"
    weights.write_text(json.dumps(fitted))
    subprocess.run(
        [sys.executable, "kaggle/export.py", "--model", "role_ridge", "--weights", str(weights)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    path = tmp_path / "artifacts/kaggle/submission.ipynb"
    for command in [["check"], ["format", "--check"]]:
        subprocess.run(
            [sys.executable, "-m", "ruff", *command, "--no-respect-gitignore", str(path)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    notebook = nbformat.read(path, as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    compile("\n".join(cells), "submission.py", "exec")
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    interface = next(
        node
        for node in ast.parse(cells[-1]).body
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    exec(compile(ast.Module(body=[interface], type_ignores=[]), "predict.py", "exec"), namespace)
    targets = truth[KEYS].sample(frac=1, random_state=31)
    actual = namespace["predict"](targets, observed)
    expected = predict(observed, targets, "role_ridge", fitted)[["x", "y"]]
    pd.testing.assert_frame_equal(actual, expected)


def test_export_rejects_incompatible_model_without_overwriting_artifact(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for relative in [
        "kaggle/export.py",
        "src/nfl_trajectory/motion.py",
        "src/nfl_trajectory/models.py",
        "pyproject.toml",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    weights = tmp_path / "model.json"
    weights.write_text('{"format": 9, "basis": []}')
    destination = tmp_path / "artifacts/kaggle/submission.ipynb"
    destination.parent.mkdir(parents=True)
    destination.write_text("previous valid artifact")
    result = subprocess.run(
        [sys.executable, "kaggle/export.py", "--model", "role_ridge", "--weights", str(weights)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert destination.read_text() == "previous valid artifact"


def exporter_module():
    """Load the canonical exporter without invoking its CLI."""
    import runpy

    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "kaggle/export.py"))


def inference_namespace(tmp_path):
    """Exercise embedded runtime directly; no Kaggle connection or gateway replacement."""
    import hashlib
    import io
    import json
    import os
    import tempfile
    import threading
    import time
    from datetime import UTC, datetime

    import numpy as np

    from nfl_trajectory.motion import require_keys

    namespace = dict(
        hashlib=hashlib,
        io=io,
        json=json,
        os=os,
        tempfile=tempfile,
        threading=threading,
        time=time,
        UTC=UTC,
        datetime=datetime,
        Path=Path,
        np=np,
        pd=pd,
        KEYS=KEYS,
        require_keys=require_keys,
        MODEL_NAME="constant_velocity",
        MODEL_FINGERPRINT="tested-model",
        model_prediction=lambda target, observed: constant_velocity(observed, target)[["x", "y"]],
    )
    exec(exporter_module()["RUNTIME"], namespace)
    namespace.update(
        CACHE_ENABLED=True, CACHE_DIR=tmp_path / "cache", LOG_PATH=tmp_path / "events.jsonl"
    )
    return namespace


def test_play_cache_reuses_valid_predictions_and_rejects_corruption(tmp_path):
    ns = inference_namespace(tmp_path)
    observed, truth = synthetic_play()
    targets = truth[KEYS].sample(frac=1, random_state=5)
    original = ns["model_prediction"]
    calls = []

    def counted(target, observed):
        calls.append(1)
        return original(target, observed)

    ns["model_prediction"] = counted
    first = ns["cached_predict"](targets, observed)
    second = ns["cached_predict"](targets, observed)
    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1
    cached = next((tmp_path / "cache").glob("*/prediction.npy"))
    cached.write_bytes(b"interrupted write")
    third = ns["cached_predict"](targets, observed)
    pd.testing.assert_frame_equal(first, third)
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["targets", "observed", "model", "version"])
def test_play_cache_signature_includes_inputs_model_and_versions(tmp_path, monkeypatch, change):
    ns = inference_namespace(tmp_path)
    observed, truth = synthetic_play()
    targets = truth[KEYS]
    ns["cached_predict"](targets, observed)
    if change == "targets":
        targets = targets.iloc[::-1]
    elif change == "observed":
        observed = observed.copy()
        observed["x"] += 0.1
    elif change == "model":
        ns["MODEL_FINGERPRINT"] = "changed"
    else:
        monkeypatch.setattr(pd, "__version__", "test-only-version")
    ns["cached_predict"](targets, observed)
    assert len(list((tmp_path / "cache").glob("*/complete.json"))) == 2


@pytest.mark.parametrize("problem", ["rows", "columns", "nan", "inf"])
def test_inference_rejects_incomplete_or_nonfinite_predictions(tmp_path, problem):
    import numpy as np

    ns = inference_namespace(tmp_path)
    prediction = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    if problem == "rows":
        prediction = prediction.iloc[:1]
    elif problem == "columns":
        prediction = prediction.assign(id=[1, 2])
    else:
        prediction.loc[0, "x"] = np.nan if problem == "nan" else np.inf
    with pytest.raises(ValueError):
        ns["validate_prediction"](prediction, 2)


def test_failed_gateway_preserves_last_good_output_and_working_directory(tmp_path, monkeypatch):
    ns = inference_namespace(tmp_path)
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "submission.parquet"
    path.write_bytes(b"last good")

    class InterruptedGateway:
        def run_local_gateway(self, paths):
            Path("submission.parquet").write_bytes(b"partial")
            raise RuntimeError("Injected interruption")

    with pytest.raises(RuntimeError, match="Injected"):
        ns["run_gateway"](InterruptedGateway(), tmp_path)
    assert path.read_bytes() == b"last good"
    assert Path.cwd() == tmp_path
    assert not list(tmp_path.glob(".gateway-*"))


def test_gateway_output_must_match_organizer_play_order(tmp_path, monkeypatch):
    import json

    ns = inference_namespace(tmp_path)
    targets = pd.DataFrame({"id": ["a", "b", "c"], "game_id": [1, 1, 1], "play_id": [2, 1, 2]})
    targets.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "result.json").write_text(json.dumps({"Succeeded": True}))
    (tmp_path / "submission.parquet").write_bytes(b"test-only-placeholder")
    # Only serialization is mocked here. The official gateway is NOT reported as executed.
    output = pd.DataFrame({"id": ["a", "c", "b"], "x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    monkeypatch.setattr(pd, "read_parquet", lambda path: output)
    manifest = ns["validate_gateway_output"](tmp_path, tmp_path)
    assert manifest["rows"] == 3
    assert manifest["uploaded_to_kaggle"] is False
    assert manifest["leaderboard_score"] is None
    output.loc[1, "id"] = "wrong"
    with pytest.raises(ValueError, match="IDs/order"):
        ns["validate_gateway_output"](tmp_path, tmp_path)


def test_hidden_rerun_disables_preview_cache(tmp_path, monkeypatch):
    ns = inference_namespace(tmp_path)
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    calls = []

    class HiddenServer:
        def serve(self):
            calls.append(ns["CACHE_ENABLED"])

    ns["run_gateway"](HiddenServer(), tmp_path)
    assert calls == [False]
    assert not (tmp_path / "submission.parquet").exists()


@pytest.fixture
def residual_project(tmp_path):
    import json

    from nfl_trajectory.feature_experiment import numerical_sources
    from nfl_trajectory.models import design, fit_statistics, sufficient_statistics
    from nfl_trajectory.runtime import sha256

    root = Path(__file__).resolve().parents[1]
    for relative in [
        "kaggle/export.py",
        "pyproject.toml",
        *[
            f"src/nfl_trajectory/{name}.py"
            for name in ("motion", "models", "features", "feature_experiment", "benchmark")
        ],
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    observed, truth = synthetic_play()
    observed = observed.assign(
        ball_land_x=44.0,
        ball_land_y=28.0,
        num_frames_output=20,
        player_role="Targeted Receiver",
        player_side="Offense",
        play_direction="right",
    )
    state, x = design(observed, truth[KEYS])
    baseline = fit_statistics([sufficient_statistics(state, x, truth)])
    baseline.update(training_games=[1], split_sha256="synthetic-frozen-split")
    basepath = tmp_path / "baseline.json"
    basepath.write_text(json.dumps(baseline))
    residual = {
        "features": ["time_seconds", "time_squared"],
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "coefficients": [[0.01, -0.02], [0.03, 0.04]],
        "intercept": [0.1, -0.1],
    }
    bundle = {
        "format": 1,
        "baseline_sha256": sha256(basepath),
        "split_sha256": baseline["split_sha256"],
        "training_games": baseline["training_games"],
        "source_sha256": numerical_sources(),
        "models": {"landing_ridge": residual},
    }
    featurepath = tmp_path / "features.json"
    featurepath.write_text(json.dumps(bundle))
    return tmp_path, basepath, featurepath, observed, truth, baseline, residual


@pytest.mark.parametrize(
    "problem", ["baseline", "source", "split", "scale", "feature", "nonfinite"]
)
def test_residual_export_provenance_validation(residual_project, problem):
    import json

    root, baseline, featurepath, *_ = residual_project
    bundle = json.loads(featurepath.read_text())
    if problem in ("baseline", "split"):
        bundle[problem + "_sha256"] = "mismatch"
    elif problem == "source":
        bundle["source_sha256"]["features.py"] = "mismatch"
    elif problem == "scale":
        bundle["models"]["landing_ridge"]["scale"][0] = 0
    elif problem == "feature":
        bundle["models"]["landing_ridge"]["features"][0] = "future_target_x"
    else:
        bundle["models"]["landing_ridge"]["coefficients"][0][0] = float("nan")
    with pytest.raises(ValueError):
        exporter_module()["validate_residual"](bundle, baseline)


def test_residual_export_matches_package_and_is_byte_reproducible(residual_project):
    from nfl_trajectory.feature_experiment import predict_residual
    from nfl_trajectory.features import build_player_features
    from nfl_trajectory.models import predict

    root, baselinepath, featurepath, observed, truth, baseline, residual = residual_project
    export = exporter_module()["export_notebook"]
    path = export(root, "landing_ridge", baselinepath, featurepath)
    payload, mtime = path.read_bytes(), path.stat().st_mtime_ns
    export(root, "landing_ridge", baselinepath, featurepath)
    assert path.read_bytes() == payload
    assert path.stat().st_mtime_ns == mtime
    notebook = nbformat.read(path, 4)
    ns = {}
    exec(notebook.cells[1].source, ns)
    for direction in ("right", "left"):
        current = observed.assign(play_direction=direction)
        target = truth[KEYS].sample(frac=1, random_state=77)
        actual = ns["model_prediction"](target, current)
        expected = predict(current, target, "role_ridge", baseline)[["x", "y"]]
        expected += predict_residual(build_player_features(current), target, residual)
        pd.testing.assert_frame_equal(actual, expected)
    subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-respect-gitignore", str(path)],
        cwd=root,
        check=True,
        capture_output=True,
    )
