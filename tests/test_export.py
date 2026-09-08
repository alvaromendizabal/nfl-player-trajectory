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
    for relative in [
        "kaggle/export.py", "src/nfl_trajectory/motion.py", "src/nfl_trajectory/runtime.py",
        "pyproject.toml",
    ]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    subprocess.run(
        [sys.executable, "kaggle/export.py"], cwd=tmp_path, check=True,
        capture_output=True, text=True,
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


def checkpoint_namespace():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("export_checkpoint_test", root / "kaggle/export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook, _ = module.build_notebook(root, "constant_velocity", root / "docs/results/model.json")
    cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    namespace = {}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    interface = next(n for n in ast.parse(cells[-1]).body
                     if isinstance(n, ast.FunctionDef) and n.name == "predict")
    exec(compile(ast.Module(body=[interface], type_ignores=[]), "predict.py", "exec"), namespace)
    return namespace


def test_inference_reuses_verified_predictions_without_recomputing(tmp_path, monkeypatch):
    from nfl_trajectory.runtime import Run

    namespace = checkpoint_namespace()
    observed, truth = synthetic_play()
    monkeypatch.chdir(tmp_path)
    with Run(tmp_path, "inference_test") as run:
        namespace["_gateway_run"] = run
        expected = namespace["predict"](truth[KEYS], observed)

        def prohibited(*args, **kwargs):
            raise AssertionError("Verified predictions must not be recomputed")

        namespace["constant_velocity"] = prohibited
        pd.testing.assert_frame_equal(namespace["predict"](truth[KEYS], observed), expected)
    assert '"stage_reused"' in run.log_path.read_text()


def test_corrupt_inference_cache_is_rebuilt(tmp_path, monkeypatch):
    from nfl_trajectory.runtime import Run

    namespace = checkpoint_namespace()
    observed, truth = synthetic_play()
    monkeypatch.chdir(tmp_path)
    with Run(tmp_path, "inference_test") as run:
        namespace["_gateway_run"] = run
        expected = namespace["predict"](truth[KEYS], observed)
        path = next((tmp_path / "artifacts/inference").glob("*.json"))
        path.write_text("truncated")
        pd.testing.assert_frame_equal(namespace["predict"](truth[KEYS], observed), expected)


@pytest.mark.parametrize("change", ["target_order", "observed_values"])
def test_changed_inference_inputs_get_distinct_checkpoints(tmp_path, monkeypatch, change):
    from nfl_trajectory.runtime import Run

    namespace = checkpoint_namespace()
    observed, truth = synthetic_play()
    target = truth[KEYS]
    monkeypatch.chdir(tmp_path)
    with Run(tmp_path, "inference_test") as run:
        namespace["_gateway_run"] = run
        namespace["predict"](target, observed)
        if change == "target_order":
            target = target.iloc[::-1]
        else:
            observed = observed.copy()
            observed["x"] += 1.0
        namespace["predict"](target, observed)
    assert len(list((tmp_path / "artifacts/inference").glob("*.json"))) == 2
