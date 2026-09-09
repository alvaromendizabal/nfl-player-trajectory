"""The owner-controlled notebook must embed the verified final model and preserve parity."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd
from test_feature_research import example_tracking
from test_feature_research import research_project as research_project
from test_final_fit import fitting_project as fitting_project
from test_final_inference import inference_project as inference_project

from nfl_trajectory.final_inference import final_bundle, predict_final
from nfl_trajectory.motion import KEYS
from nfl_trajectory.runtime import sha256


def test_final_notebook_export_matches_package_and_resumes(inference_project):
    root = inference_project
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(
        source / "src/nfl_trajectory",
        root / "src/nfl_trajectory",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "kaggle").mkdir(exist_ok=True)
    shutil.copyfile(source / "kaggle/export.py", root / "kaggle/export.py")
    shutil.copyfile(source / "kaggle/final_export.py", root / "kaggle/final_export.py")
    shutil.copyfile(source / "pyproject.toml", root / "pyproject.toml")
    command = [
        sys.executable,
        "kaggle/final_export.py",
        "--output",
        "artifacts/kaggle/submission.ipynb",
    ]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    path = root / "artifacts/kaggle/submission.ipynb"
    before = sha256(path), path.stat().st_mtime_ns
    notebook = nbformat.read(path, as_version=4)
    assert path.stat().st_size < 1_000_000
    assert notebook.metadata.nfl_export.model == "final"
    assert notebook.metadata.nfl_export.automatically_submitted is False
    namespace = {}
    cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    exec(compile(cells[0], "final-notebook-model.py", "exec"), namespace)
    assert namespace["FINAL_MODEL"] == final_bundle(root)
    callback = next(
        node
        for node in ast.parse(cells[-1]).body
        if isinstance(node, ast.FunctionDef) and node.name == "predict"
    )
    exec(compile(ast.Module(body=[callback], type_ignores=[]), "callback.py", "exec"), namespace)
    observed, targets = example_tracking(2025010100)
    targets = targets[KEYS].sample(frac=1, random_state=2026)
    pd.testing.assert_frame_equal(
        namespace["predict"](targets, observed),
        predict_final(observed, targets, final_bundle(root))[["x", "y"]],
    )
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    assert (sha256(path), path.stat().st_mtime_ns) == before
