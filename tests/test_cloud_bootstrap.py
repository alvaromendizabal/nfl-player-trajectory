"""Guard launchers that run before the locked Python 3.11 environment exists.

The SageMaker bootstrap image uses Python 3.9. Scientific modules may require
3.11, but the two launchers must parse on 3.9 and must not import datetime.UTC,
which caused the 2026-09-11 recovery job to fail before writing a status receipt.
These are static compatibility checks, not a claim of full cross-version tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    "scripts/cloud_motion_engineering.py",
    "scripts/cloud_motion_scientific.py",
)


@pytest.mark.parametrize("relative_path", LAUNCHERS)
def test_bootstrap_syntax_supports_base_python(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path, feature_version=(3, 9))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    ), "Postpone annotations until the locked runtime is available."


@pytest.mark.parametrize("relative_path", LAUNCHERS)
def test_bootstrap_avoids_python311_only_datetime_utc(relative_path: str) -> None:
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    datetime_modules = {"datetime"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            datetime_modules.update(
                alias.asname or alias.name for alias in node.names if alias.name == "datetime"
            )
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            assert all(alias.name not in {"UTC", "*"} for alias in node.names), (
                "Use datetime.timezone.utc in the base-image launcher."
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert not (node.value.id in datetime_modules and node.attr == "UTC"), (
                "datetime.UTC is unavailable in the Python 3.9 bootstrap image."
            )
