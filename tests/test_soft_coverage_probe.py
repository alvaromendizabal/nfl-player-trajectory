"""Probe contracts for screening and official-metric accounting."""

import importlib.util
from pathlib import Path

import numpy as np


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/soft_coverage.py"
    spec = importlib.util.spec_from_file_location("coverage_probe", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_training_screen_and_origin_prediction():
    probe = module()
    rng = np.random.default_rng(8)
    first = rng.normal(size=100)
    x = np.column_stack([first, first, np.ones(100)])
    seconds = rng.uniform(0.1, 3, size=100)
    y = np.column_stack([first * seconds, -first * seconds])
    fitted = probe.fit_ridge(x, y, seconds)
    np.testing.assert_array_equal(fitted["retained"], [True, False, False])
    np.testing.assert_allclose(fitted["mean"], x.mean(0))
    np.testing.assert_array_equal(probe.predict(fitted, x[:2] + 999, np.zeros(2)), 0)
    assert np.sqrt(np.mean((probe.predict(fitted, x, seconds) - y) ** 2)) < 0.02


def test_metric_pools_coordinates_instead_of_averaging_games():
    probe = module()
    reference = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]])
    treatment = reference / 2
    report = probe.comparison(reference, treatment, np.array([1, 1, 2]))
    np.testing.assert_allclose(report["reference_rmse_yards"], np.sqrt(30 / 6))
    np.testing.assert_allclose(report["coordinate_rmse_yards"], np.sqrt(30 / 6) / 2)
    assert report["continue_gate"]
