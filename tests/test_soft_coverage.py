"""Physical, missingness and leakage contracts for uncertain matchup features."""

import numpy as np
import pytest
from test_temporal_data import sample

from nfl_trajectory.soft_coverage import candidates
from nfl_trajectory.temporal_data import CHANNELS, reflect


def test_outcomes_and_player_identity_do_not_change_features():
    source = sample(5)
    # Exercise read-only views under every supported pandas version.
    for name in ("truth", "ids", "keys"):
        source[name].setflags(write=False)
    expected, names, _ = candidates(source)
    source["truth"] = np.full_like(source["truth"], 1e9)
    # Pandas 3 may expose a read-only NumPy view; replace it without weakening
    # the identity-invariance assertion.
    source["ids"] = source["ids"] + 9000
    source["keys"] = np.full_like(source["keys"], -1)
    np.testing.assert_array_equal(candidates(source)[0], expected)
    assert len(names) == len(set(names)) == 83


def test_player_permutation_preserves_query_features():
    source = sample(5)
    expected = candidates(source)[0]
    order = np.array([4, 1, 3, 0, 2])
    for name in (
        "history",
        "observed",
        "static",
        "role",
        "side",
        "xy",
        "velocity",
        "ids",
        "horizon",
    ):
        source[name] = source[name][order]
    source["player"] = np.argsort(order)[source["player"]]
    np.testing.assert_allclose(candidates(source)[0], expected, atol=1e-7)


def test_reflection_has_known_scalar_and_vector_parity():
    source = sample(5)
    expected, names, _ = candidates(source)
    reflected = candidates(reflect(source))[0]
    parity = np.ones(len(names))
    for i, name in enumerate(names):
        token = name.split("__")[1]
        if token in ("dy", "dvy") or any(v in token for v in ("_dy_", "_dvy_")):
            parity[i] = -1
    np.testing.assert_allclose(reflected, expected * parity, atol=2e-6)


def test_unobserved_padding_is_ignored_before_arithmetic():
    source = sample(5)
    source["observed"][:, :7] = False
    expected = candidates(source)[0]
    source["history"][:, :7] = np.nan
    np.testing.assert_array_equal(candidates(source)[0], expected)


def test_no_opponent_produces_unmatched_state():
    source = sample(5)
    source["side"][:] = 1
    values, names, _ = candidates(source)
    assert np.isfinite(values).all()
    for name in ("match_mass", "entropy", "dx", "dy", "target_mass"):
        assert (values[:, names.index("soft_static__" + name)] == 0).all()


def test_distinct_histories_survive_equal_terminal_geometry():
    source = sample(5)
    expected, names, _ = candidates(source)
    channel = list(CHANNELS).index("relative_x")
    source["history"][2:, :12, channel] += 0.7
    changed = candidates(source)[0]
    static = [i for i, n in enumerate(names) if n.startswith("soft_static")]
    temporal = [i for i, n in enumerate(names) if n.startswith("soft_temporal")]
    np.testing.assert_array_equal(changed[:, static], expected[:, static])
    assert not np.allclose(changed[:, temporal], expected[:, temporal])


def test_actual_frame_gaps_do_not_become_adjacent_turnovers():
    source = sample(5)
    source["observed"][:, :-1] = False
    source["observed"][:, -3] = True
    values, names, _ = candidates(source)
    assert (values[:, [i for i, n in enumerate(names) if n.endswith("turnover")]] == 0).all()


def test_no_observed_state_is_rejected():
    source = sample()
    source["observed"][0] = False
    with pytest.raises(ValueError, match="Every player"):
        candidates(source)
