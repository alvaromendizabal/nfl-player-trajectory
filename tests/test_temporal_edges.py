"""Counterfactual tests for observed temporal interaction features; no training."""

import copy

import numpy as np
import pytest

from nfl_trajectory.temporal_edges import NAMES, ODD_LATERAL, SCALES, build_edges, from_sample


def fixture():
    rng = np.random.default_rng(20260911)
    xy = rng.normal(size=(4, 20, 2)) + np.arange(4)[:, None, None] * 3
    velocity = rng.normal(size=xy.shape)
    return xy, velocity, np.ones((4, 20), dtype=bool), np.array([0, 1, 0, 1])


def test_units_closing_and_backward_separation_rate():
    xy = np.zeros((2, 3, 2))
    xy[1, :, 0] = [3, 2, 1]
    v = np.zeros_like(xy)
    v[1, :, 0] = -10
    result = build_edges(xy, v, np.ones((2, 3), bool), np.array([0, 1]))
    raw = result["values"] * SCALES
    np.testing.assert_allclose(raw[0, 1, :, NAMES.index("closing_speed")], 10)
    np.testing.assert_allclose(raw[0, 1, 1:, NAMES.index("separation_rate")], -10)
    assert not result["valid"][0, 1, 0, 9]
    assert not result["pair_valid"][0, 0].any()


def test_translation_and_player_permutation_equivariance():
    xy, v, seen, side = fixture()
    reference = build_edges(xy, v, seen, side)
    translated = build_edges(xy + [800, -900], v, seen, side)
    np.testing.assert_allclose(reference["values"], translated["values"], atol=1e-6)
    p = np.array([2, 0, 3, 1])
    permuted = build_edges(xy[p], v[p], seen[p], side[p])
    for key in reference:
        np.testing.assert_array_equal(permuted[key], reference[key][p][:, p])


def test_lateral_reflection_parity():
    xy, v, seen, side = fixture()
    expected = build_edges(xy, v, seen, side)
    actual = build_edges(xy * [1, -1], v * [1, -1], seen, side)
    values = expected["values"].copy()
    values[..., list(ODD_LATERAL)] *= -1
    np.testing.assert_allclose(actual["values"], values, atol=1e-6)
    np.testing.assert_array_equal(actual["valid"], expected["valid"])


def test_missing_values_never_enter_arithmetic_or_bridge_gaps():
    xy, v, seen, side = fixture()
    seen[1, 6:9] = False
    first = build_edges(xy, v, seen, side)
    xy[~seen], v[~seen] = np.nan, np.inf
    second = build_edges(xy, v, seen, side)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert not second["pair_valid"][0, 1, 6:9].any()
    assert not second["valid"][0, 1, 9, [8, 9]].any()
    assert second["valid"][0, 1, 10, [8, 9]].all()


def test_stationary_and_coincident_geometry_have_masks():
    xy = np.zeros((2, 2, 2))
    result = build_edges(xy, xy, np.ones((2, 2), bool), np.array([0, 1]))
    assert np.isfinite(result["values"]).all()
    assert not result["valid"][..., [5, 6, 7, 8]].any()
    assert result["valid"][0, 1, 1, 9]


def test_inputs_are_not_mutated_and_output_has_bounded_size():
    arguments = fixture()
    original = copy.deepcopy(arguments)
    result = build_edges(*arguments)
    for a, b in zip(arguments, original, strict=True):
        np.testing.assert_array_equal(a, b)
    assert result["values"].shape == (4, 4, 20, 11)
    maximum = build_edges(
        np.ones((22, 20, 2)), np.ones((22, 20, 2)), np.ones((22, 20), bool), np.zeros(22)
    )
    assert sum(value.nbytes for value in maximum.values()) < 600_000


def test_changing_later_observed_frames_does_not_change_earlier_edges():
    xy, v, seen, side = fixture()
    first = build_edges(xy, v, seen, side)
    xy[:, 10:] += 100
    v[:, 10:] += 30
    second = build_edges(xy, v, seen, side)
    np.testing.assert_array_equal(first["values"][..., :10, :], second["values"][..., :10, :])


def test_adapter_does_not_read_labels_or_queries():
    from nfl_trajectory.temporal_data import CHANNELS

    xy, v, seen, side = fixture()
    history = np.zeros((4, 20, len(CHANNELS)))
    anchor = xy[:, -1]
    for name, data in zip(
        ("relative_x", "relative_y", "vx", "vy"),
        (xy[..., 0] - anchor[:, None, 0], xy[..., 1] - anchor[:, None, 1], v[..., 0], v[..., 1]),
        strict=True,
    ):
        history[..., list(CHANNELS).index(name)] = data / CHANNELS[name]

    class InputsOnly(dict):
        def __getitem__(self, key):
            if key not in {"history", "observed", "xy", "side"}:
                raise AssertionError("Label or query field was accessed")
            return super().__getitem__(key)

    sample = InputsOnly(history=history, observed=seen, xy=anchor, side=side, truth=object())
    result = from_sample(sample)
    expected = build_edges(xy, v, seen, side)
    np.testing.assert_allclose(result["values"], expected["values"], atol=1e-6)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan, np.inf])
def test_invalid_time_interval_rejected(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        build_edges(*fixture(), seconds_per_frame=bad)


@pytest.mark.parametrize("case", ["mask", "shape", "side", "empty", "nonfinite"])
def test_invalid_observed_input_rejected(case):
    xy, v, seen, side = fixture()
    if case == "mask":
        seen = seen.astype(int)
    elif case == "shape":
        xy = xy[:, :, :1]
    elif case == "side":
        side[0] = 5
    elif case == "empty":
        seen[0] = False
    else:
        xy[0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        build_edges(xy, v, seen, side)


def test_notebook_cache_tracks_implementation_and_private_evidence(tmp_path):
    import nbformat

    from scripts.notebooks import execution_signature

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/notebooks.py").write_text("# executor")
    module = tmp_path / "src/nfl_trajectory/temporal_edges.py"
    module.parent.mkdir(parents=True)
    module.write_text("# version one")
    notebook = tmp_path / "03_interaction_research.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]), notebook)
    a = execution_signature(tmp_path, notebook)
    module.write_text("# version two")
    b = execution_signature(tmp_path, notebook)
    assert a != b
    report = tmp_path / "artifacts/interaction_milestone/inspection.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"new evidence"}')
    assert execution_signature(tmp_path, notebook) != b
