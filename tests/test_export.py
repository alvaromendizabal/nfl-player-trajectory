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
    # Compilation as one script catches misplaced future imports across cell boundaries.
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
