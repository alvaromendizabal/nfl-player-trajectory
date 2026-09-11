"""Independent physics, support, leakage, and symmetry tests; no scientific fits."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from nfl_trajectory.features import ROLES
from nfl_trajectory.relation_history import (
    CONTEXT_COLUMNS,
    GEOMETRY_COLUMNS,
    MOTION_COLUMNS,
    ODD_RELATIONS,
    RELATION_NAMES,
    RelationHistory,
    pair_history,
)
from nfl_trajectory.temporal_data import CHANNELS, HISTORY, play_features, reflect


def fixture(players: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = np.zeros((players, HISTORY, len(CHANNELS)), dtype=np.float32)
    channels = list(CHANNELS)
    # Player 1 is (3, 4) yards from player 0; relative velocity is (-1, 0) yd/s.
    for player in range(players):
        history[player, :, channels.index("ball_dx")] = (20 - 3 * player) / 20
        history[player, :, channels.index("ball_dy")] = (10 - 4 * player) / 20
        history[player, :, channels.index("vx")] = (2 - player) / 10
    observed = np.ones((players, HISTORY), dtype=bool)
    side = np.arange(players, dtype=np.int64) % 2
    role = np.arange(players, dtype=np.int64) % len(ROLES)
    return history, observed, side, role


def build(args: tuple[np.ndarray, ...]) -> RelationHistory:
    return pair_history(*args)


def test_directed_geometry_matches_independent_physics() -> None:
    result = build(fixture())
    values = result.values[0, 1, -1]
    np.testing.assert_allclose(
        values[:6], [3 / 20, 4 / 20, -1 / 10, 0, 5 / 20, 0.6 / 10], atol=1e-7
    )
    assert values[6] == values[7] == values[8] == 1
    assert values[9] == values[10] == values[11] == 0
    np.testing.assert_allclose(result.values[1, 0, -1, :4], -values[:4])
    np.testing.assert_allclose(result.values[1, 0, -1, 4:9], values[4:9])
    assert result.values[1, 0, -1, 10] == 1  # Neighbour, not focal, receiver flag.


def test_missing_frames_never_pair_independent_terminal_positions() -> None:
    args = fixture(2)
    args[1][0] = False
    args[1][1] = False
    args[1][0, :10] = True
    args[1][1, 10:] = True
    result = build(args)
    assert not result.observed.any()
    assert not result.values.any()
    assert not result.terminal_view().observed.any()


def test_terminal_view_preserves_true_age_and_does_not_retime() -> None:
    args = fixture(2)
    args[1][1, -3:] = False
    result = build(args)
    terminal = result.terminal_view()
    assert terminal.observed.sum() == 2
    assert terminal.observed[0, 1, -4]
    assert terminal.frame_seconds[-4] == pytest.approx(-0.3)
    assert not terminal.observed[..., -3:].any()
    np.testing.assert_array_equal(terminal.values[0, 1, -4], result.values[0, 1, -4])
    np.testing.assert_array_equal(terminal.terminal_view().values, terminal.values)


def test_missing_nonfinite_values_are_masked_before_arithmetic() -> None:
    args = fixture(2)
    args[1][1, 4:9] = False
    reference = build(args)
    args[0][1, 4:9, :] = np.nan
    args[0][1, 5, :] = np.inf
    actual = build(args)
    np.testing.assert_array_equal(actual.values, reference.values)
    assert np.isfinite(actual.values).all()


@pytest.mark.parametrize("name", ["ball_dx", "ball_dy", "vx", "vy"])
def test_nonfinite_supported_values_fail_closed(name: str) -> None:
    args = fixture()
    args[0][0, 0, list(CHANNELS).index(name)] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build(args)


def test_player_permutation_is_equivariant() -> None:
    args = fixture(4)
    args[1][2, :5] = False
    order = np.array([3, 1, 0, 2])
    expected = build(args)
    actual = build(tuple(value[order] for value in args))
    np.testing.assert_array_equal(actual.values, expected.values[order][:, order])
    np.testing.assert_array_equal(actual.observed, expected.observed[order][:, order])


def test_shared_landmark_translation_cancels() -> None:
    args = fixture()
    expected = build(args)
    args[0][..., list(CHANNELS).index("ball_dx")] += 2.0
    args[0][..., list(CHANNELS).index("ball_dy")] -= 3.0
    actual = build(args)
    np.testing.assert_allclose(actual.values, expected.values, atol=2e-6)


def test_player_specific_origins_and_unrelated_channels_are_not_geometry() -> None:
    args = fixture()
    expected = build(args)
    unused = set(CHANNELS) - {"ball_dx", "ball_dy", "vx", "vy"}
    for name in unused:
        args[0][..., list(CHANNELS).index(name)] = np.nan
    np.testing.assert_array_equal(build(args).values, expected.values)


def test_lateral_reflection_has_declared_parity() -> None:
    args = fixture()
    reference = build(args)
    for name in ("ball_dy", "vy"):
        args[0][..., list(CHANNELS).index(name)] *= -1
    expected = reference.values.copy()
    expected[..., list(ODD_RELATIONS)] *= -1
    np.testing.assert_array_equal(build(args).values, expected)


def test_zero_speed_and_coincident_positions_have_explicit_support() -> None:
    args = fixture(2)
    args[0][:] = 0
    result = build(args)
    assert result.observed.sum() == 2 * HISTORY
    assert not result.values[..., :9].any()
    assert np.isfinite(result.values).all()


def test_read_only_inputs_are_not_mutated_or_aliased() -> None:
    args = fixture()
    original = copy.deepcopy(args)
    for value in args:
        value.setflags(write=False)
    result = build(args)
    result.values[:] = 123
    result.observed[:] = False
    for actual, expected in zip(args, original, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("players", [1, 22])
def test_full_supported_slot_range_and_no_self_edges(players: int) -> None:
    result = build(fixture(players))
    assert result.values.shape == (players, players, HISTORY, len(RELATION_NAMES))
    assert result.observed.sum() == players * (players - 1) * HISTORY
    assert not result.observed[np.arange(players), np.arange(players)].any()
    assert np.count_nonzero(result.values[~result.observed]) == 0


@pytest.mark.parametrize("case", ["shape", "players", "integer_history", "mask", "side", "role"])
def test_invalid_contracts_fail_closed(case: str) -> None:
    args = list(fixture())
    if case == "shape":
        args[0] = args[0][:, :-1]
    elif case == "players":
        args = list(fixture(23))
    elif case == "integer_history":
        args[0] = args[0].astype(int)
    elif case == "mask":
        args[1] = args[1].astype(float)
    elif case == "side":
        args[2][0] = 2
    else:
        args[3][0] = -1
    with pytest.raises(ValueError):
        build(tuple(args))


def test_future_labels_are_not_an_accepted_argument() -> None:
    with pytest.raises(TypeError, match="truth"):
        pair_history(*fixture(), truth=np.zeros((5, 2)))


def test_family_columns_partition_the_schema() -> None:
    assert sorted((*GEOMETRY_COLUMNS, *MOTION_COLUMNS, *CONTEXT_COLUMNS)) == list(
        range(len(RELATION_NAMES))
    )


def test_raw_tracking_adapter_and_reflection_agree() -> None:
    from test_temporal_data import example

    raw, _, _ = example()
    sample = play_features(raw)
    result = pair_history(*(sample[k] for k in ("history", "observed", "side", "role")))
    reflected = reflect(sample)
    flipped = pair_history(*(reflected[k] for k in ("history", "observed", "side", "role")))
    expected = result.values.copy()
    expected[..., list(ODD_RELATIONS)] *= -1
    np.testing.assert_allclose(flipped.values, expected, atol=2e-6)
    # This independently known raw geometry survives each player's distinct origin.
    np.testing.assert_allclose(result.values[0, 1, -1, :2], [2 / 20, 1 / 20], atol=2e-6)


def test_unrepresentable_active_input_is_rejected_before_overflow() -> None:
    args = list(fixture())
    args[0] = args[0].astype(np.float64)
    args[0][0, 0, list(CHANNELS).index("vx")] = 1e308
    with pytest.raises(ValueError, match="finite"):
        build(tuple(args))


def test_all_missing_input_is_supported_without_fake_edges() -> None:
    args = fixture()
    args[1][:] = False
    args[0][:] = np.nan
    result = build(args)
    assert not result.observed.any()
    assert not result.values.any()
    np.testing.assert_allclose(result.frame_seconds[[0, -1]], [-1.9, 0.0], atol=1e-7)


def test_feature_changes_remain_at_their_actual_observed_frame() -> None:
    args = fixture(2)
    before = build(args)
    args[0][1, 6, list(CHANNELS).index("ball_dx")] += 0.5
    after = build(args)
    other_frames = np.arange(HISTORY) != 6
    np.testing.assert_array_equal(
        after.values[:, :, other_frames], before.values[:, :, other_frames]
    )
    assert not np.array_equal(after.values[0, 1, 6], before.values[0, 1, 6])
