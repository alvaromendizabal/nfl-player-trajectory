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
    for relative in ["kaggle/export.py", "src/nfl_trajectory/motion.py", "src/nfl_trajectory/runtime.py", "pyproject.toml"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    subprocess.run(
        [sys.executable, "kaggle/export.py"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    return tmp_path / "artifacts/kaggle/submission.ipynb"


def test_export_passes_lint_and_format_without_git(exported_notebook: Path) -> None:
    root = exported_notebook.parents[2]
    assert not (root / ".git").exists()
    for arguments in [["check"], ["format", "--check"]]:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *arguments, "--no-respect-gitignore", str(exported_notebook)],
            cwd=root, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_exported_predictor_matches_reference_and_preserves_order(exported_notebook: Path) -> None:
    notebook = nbformat.read(exported_notebook, as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    compile("\n".join(cells), "submission.py", "exec")
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    function = next(
        node for node in ast.parse(cells[-1]).body
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
        "kaggle/export.py", "src/nfl_trajectory/motion.py", "src/nfl_trajectory/models.py",
        "src/nfl_trajectory/runtime.py", "pyproject.toml",
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
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    path = tmp_path / "artifacts/kaggle/submission.ipynb"
    for command in [["check"], ["format", "--check"]]:
        subprocess.run(
            [sys.executable, "-m", "ruff", *command, "--no-respect-gitignore", str(path)],
            cwd=tmp_path, check=True, capture_output=True,
        )
    notebook = nbformat.read(path, as_version=4)
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    compile("\n".join(cells), "submission.py", "exec")
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    interface = next(
        node for node in ast.parse(cells[-1]).body
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
        "kaggle/export.py", "src/nfl_trajectory/motion.py", "src/nfl_trajectory/models.py",
        "src/nfl_trajectory/runtime.py", "pyproject.toml",
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
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert destination.read_text() == "previous valid artifact"


def export_tools():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("nfl_export", root / "kaggle/export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def residual_fixture():
    from nfl_trajectory.models import design, fit_statistics, sufficient_statistics

    observed, truth = synthetic_play()
    observed["ball_land_x"], observed["ball_land_y"] = 44.0, 28.0
    observed["num_frames_output"] = 20
    observed["player_role"], observed["player_side"] = "Targeted Receiver", "Offense"
    observed["play_direction"] = "right"
    state, basis = design(observed, truth[KEYS])
    baseline = fit_statistics([sufficient_statistics(state, basis, truth)])
    residual = {"features": ["time_seconds", "lag00__vx", "fraction__ball_ux"],
                "mean": [0.1, 0.2, 0.3], "scale": [1.0, 2.0, 3.0],
                "coefficients": [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]],
                "intercept": [0.01, -0.02], "alpha": 0.01, "training_rows": len(truth)}
    return observed, truth, baseline, residual


def standalone_namespace(model):
    root = Path(__file__).resolve().parents[1]
    observed, truth, baseline, residual = residual_fixture()
    notebook = export_tools().build_notebook(root, model, baseline, residual, "fixture-signature")
    cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    compile("\n".join(cells), "standalone.py", "exec")
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    interface = next(node for node in ast.parse(cells[-1]).body
                     if isinstance(node, ast.FunctionDef) and node.name == "predict")
    exec(compile(ast.Module(body=[interface], type_ignores=[]), "interface.py", "exec"), namespace)
    return namespace, observed, truth, baseline, residual


@pytest.mark.parametrize("model", ["motion_ridge", "landing_ridge", "interaction_ridge"])
def test_standalone_residual_matches_tested_package(model):
    from nfl_trajectory.feature_experiment import predict_residual
    from nfl_trajectory.features import build_player_features
    from nfl_trajectory.models import predict
    from nfl_trajectory.motion import ENTITY

    namespace, observed, truth, baseline, residual = standalone_namespace(model)
    target = truth[KEYS].sample(frac=1, random_state=47)
    expected = predict(observed, target, "role_ridge", baseline)
    bank = build_player_features(observed, target[ENTITY])
    expected[["x", "y"]] += predict_residual(bank, target, residual)
    pd.testing.assert_frame_equal(namespace["predict"](target, observed), expected[["x", "y"]])
    target["x"], target["y"] = float("nan"), float("inf")
    pd.testing.assert_frame_equal(namespace["predict"](target, observed), expected[["x", "y"]])


def test_per_play_checkpoint_reuse_and_corruption_recovery(tmp_path, monkeypatch):
    from nfl_trajectory.runtime import Run

    namespace, observed, truth, _, _ = standalone_namespace("landing_ridge")
    monkeypatch.chdir(tmp_path)
    with Run(tmp_path, "gateway_test") as run:
        namespace["gateway_run"] = run
        expected = namespace["predict"](truth[KEYS], observed)
        original = namespace["build_player_features"]

        def cannot_recompute(*args, **kwargs):
            raise AssertionError("A verified completed prediction should be reused")

        namespace["build_player_features"] = cannot_recompute
        pd.testing.assert_frame_equal(namespace["predict"](truth[KEYS], observed), expected)
        namespace["build_player_features"] = original
        cached = next((tmp_path / "artifacts/inference").glob("*.json"))
        cached.write_text("interrupted")
        pd.testing.assert_frame_equal(namespace["predict"](truth[KEYS], observed), expected)
    assert '"stage_reused"' in run.log_path.read_text()
    assert '"elapsed_stage_seconds"' in run.log_path.read_text()


def test_changed_target_order_uses_a_distinct_inference_cache(tmp_path, monkeypatch):
    from nfl_trajectory.runtime import Run

    namespace, observed, truth, _, _ = standalone_namespace("landing_ridge")
    monkeypatch.chdir(tmp_path)
    with Run(tmp_path, "gateway_test") as run:
        namespace["gateway_run"] = run
        namespace["predict"](truth[KEYS], observed)
        namespace["predict"](truth[KEYS].iloc[::-1], observed)
    assert len(list((tmp_path / "artifacts/inference").glob("*.json"))) == 2


def test_same_inputs_make_identical_notebook_source():
    root = Path(__file__).resolve().parents[1]
    _, _, baseline, residual = residual_fixture()
    tool = export_tools()
    a = nbformat.writes(tool.build_notebook(root, "landing_ridge", baseline, residual, "same"))
    b = nbformat.writes(tool.build_notebook(root, "landing_ridge", baseline, residual, "same"))
    assert a == b
    assert "competition_submit" not in a
    assert "kernels_push" not in a


def test_feature_export_requires_matching_completion_evidence(tmp_path):
    import json

    from nfl_trajectory.feature_experiment import numerical_sources
    from nfl_trajectory.runtime import atomic_json, sha256

    _, _, baseline, residual = residual_fixture()
    baseline.update({"training_games": [2023090700], "split_sha256": "test-split"})
    weights = tmp_path / "artifacts/benchmark/model.json"
    atomic_json(weights, baseline)
    bundle = {"format": 1, "models": {"landing_ridge": residual},
              "training_games": baseline["training_games"], "split_sha256": "test-split",
              "source_sha256": numerical_sources(), "baseline_sha256": sha256(weights)}
    feature_weights = tmp_path / "artifacts/features/model.json"
    atomic_json(feature_weights, bundle)
    summary = {"status": "passed", "split": "validation", "screening_split": "train",
               "holdout_evaluation": "not_run", "numerical_signature": "test-signature",
               "source_sha256": numerical_sources(), "split_sha256": "test-split",
               "baseline_sha256": sha256(weights)}
    summary_path = tmp_path / "artifacts/features/summary.json"
    atomic_json(summary_path, summary)
    receipt = {"status": "completed", "signature": "test-signature", "outputs": {
        "artifacts/features/model.json": sha256(feature_weights),
        "artifacts/features/summary.json": sha256(summary_path),
    }}
    atomic_json(tmp_path / ".state/features-report.json", receipt)
    tool = export_tools()
    actual_baseline, actual_residual = tool.load_parameters(
        tmp_path, "landing_ridge", weights, feature_weights
    )
    assert actual_baseline == baseline and actual_residual == residual
    bundle["models"]["landing_ridge"]["mean"][0] += 1
    feature_weights.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="evidence"):
        tool.load_parameters(tmp_path, "landing_ridge", weights, feature_weights)
