"""Exercise the actual nested epoch publisher without starting a scientific fit."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_motion_experiment.py"


def publisher_factory(store: Mock, event: Mock) -> Any:
    """Compile the production closure, not a test-side duplicate implementation."""
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    main = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    publisher = next(
        node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "publisher_for"
    )
    isolated = ast.Module(body=[publisher], type_ignores=[])
    namespace = {"store": store, "event": event, "Path": Path, "Any": Any}
    exec(compile(isolated, str(SOURCE), "exec"), namespace)
    return namespace["publisher_for"]


@pytest.mark.parametrize("arm", ["coordinate", "velocity"])
def test_epoch_publication_logs_returned_arm_once(arm: str, tmp_path: Path) -> None:
    remote = {"arm": arm, "step": 78, "independent_readback_verified": True}
    store = Mock()
    store.publish.return_value = remote
    event = Mock()
    published: list[dict[str, Any]] = []
    callback = publisher_factory(store, event)(arm, published)
    local = {"step": 78, "sha256": "a" * 64}
    callback(tmp_path, local)
    store.publish.assert_called_once_with(arm, tmp_path, local)
    event.assert_called_once_with("epoch_checkpoint_verified", **remote)
    assert published == [remote]


def test_failed_publication_cannot_emit_a_success_event(tmp_path: Path) -> None:
    store = Mock()
    store.publish.side_effect = ValueError("independent read-back failed")
    event = Mock()
    published: list[dict[str, Any]] = []
    callback = publisher_factory(store, event)("coordinate", published)
    with pytest.raises(ValueError, match="read-back failed"):
        callback(tmp_path, {"step": 78})
    event.assert_not_called()
    assert published == []
