"""Broader feature budgets preserve parents and screen only training evidence."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_script(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("budget_test", scripts / "feature_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_retains_parent_and_excludes_redundant_new_columns(monkeypatch):
    module = load_script(monkeypatch)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1000, 5))
    x[:, 3] = 2 * x[:, 0] + 3
    x[:, 4] = 1.0
    assert module.retain_columns(x, 2) == [0, 1, 2]
    np.testing.assert_array_equal(x[:, 4], 1)


def test_budget_pool_uses_only_fold_training_screen(monkeypatch, tmp_path):
    module = load_script(monkeypatch)
    for group in ("research", "context", "representation"):
        folder = tmp_path / "artifacts" / group / "inner_1"
        folder.mkdir(parents=True)
        screen = pd.DataFrame(
            {
                "feature": [group + "_strong", group + "_constant"],
                "family": [group] * 2,
                "training_association": [0.1, 999.0],
                "screen_status": ["eligible", "constant_or_near_constant"],
            }
        )
        file = "conditional.csv" if group == "research" else "screening.csv"
        screen.to_csv(folder / file, index=False)
    expected = ["parent", "context_strong", "representation_strong", "research_strong"]
    assert module.candidate_pool(tmp_path, "inner_1", ["parent"]) == expected
    # No development/holdout input exists: a selector that requests it cannot pass.
