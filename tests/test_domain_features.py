"""Independent geometry, temporal identity and outcome-isolation contracts."""

import numpy as np
from test_temporal_data import example, sample

from nfl_trajectory.domain_features import FAMILIES, candidates
from nfl_trajectory.features import WIDTH
from nfl_trajectory.temporal_data import attach_targets, play_features, reflect


def test_domain_inputs_ignore_outcomes_and_identity_values():
    original = sample()
    expected, names, families = candidates(original)
    changed = {k: v.copy() for k, v in original.items()}
    changed["truth"][:] = 1e9
    changed["keys"][:] = 999999
    changed["ids"] += 100000
    actual, new_names, _ = candidates(changed)
    np.testing.assert_array_equal(actual, expected)
    assert names == new_names and len(names) == len(set(names))
    assert set(families) == set(FAMILIES)


def test_domain_player_permutation_preserves_keyed_features():
    original = sample(5)
    order = np.array([4, 2, 0, 3, 1])
    changed = {k: v.copy() for k, v in original.items()}
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
        changed[name] = changed[name][order]
    changed["player"] = np.argsort(order)[changed["player"]]
    np.testing.assert_allclose(candidates(changed)[0], candidates(original)[0], atol=1e-6)


def test_domain_reflection_matches_independent_raw_tracking_transform():
    raw, targets, baseline = example()
    original = attach_targets(play_features(raw), targets, baseline, baseline)
    raw["y"] = WIDTH - raw.y
    raw["ball_land_y"] = WIDTH - raw.ball_land_y
    raw["dir"], raw["o"] = (180 - raw.dir) % 360, (180 - raw.o) % 360
    baseline[:, 1] = WIDTH - baseline[:, 1]
    reflected = attach_targets(play_features(raw), targets, baseline, baseline)
    np.testing.assert_allclose(
        candidates(reflected)[0], candidates(reflect(original))[0], atol=2e-5
    )


def test_arrival_features_match_known_boundary_value_problem():
    source = sample()
    source["history"][:] = 0
    source["static"][:, 2:4] = [0.5, -0.25]  # Landing displacement (10, -5) yards.
    source["horizon"] = np.full_like(source["horizon"], 20)  # Two-second standing start.
    source["time"] = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    matrix, names, _ = candidates(source)
    for axis, expected in (("x", 5.0), ("y", -2.5)):
        np.testing.assert_allclose(
            matrix[:, names.index("arrival_constraints__required_acceleration_" + axis)] * 20,
            expected,
        )
    bridge = [
        names.index("arrival_constraints__quadratic_bridge_minus_cv_" + axis) for axis in ("x", "y")
    ]
    np.testing.assert_allclose(matrix[-1, bridge] * 10, [10, -5])


def test_synchronized_pair_history_detects_change_hidden_by_terminal_geometry():
    raw, targets, baseline = example()
    first = attach_targets(play_features(raw), targets, baseline)
    raw.loc[(raw.nfl_id == 3) & raw.frame_id.lt(20), "x"] += 3
    second = attach_targets(play_features(raw), targets, baseline)
    np.testing.assert_array_equal(first["xy"], second["xy"])
    np.testing.assert_array_equal(first["velocity"], second["velocity"])
    a, names, _ = candidates(first)
    b, _, _ = candidates(second)
    index = names.index("coverage_dynamics__opponent1_w20_distance_std")
    assert not np.allclose(a[:, index], b[:, index])


def test_gapped_observations_keep_physical_time_and_missing_peer_masks():
    raw, targets, baseline = example()
    raw = raw[~((raw.nfl_id == 1) & raw.frame_id.isin([20, 21, 23]))]
    source = attach_targets(play_features(raw), targets, baseline)
    values, names, _ = candidates(source)
    np.testing.assert_allclose(
        values[:, names.index("motion_state__w08_position_slope_x")] * 10, 3.0, atol=2e-5
    )
    raw = raw[raw.nfl_id.ne(3)]
    missing, names, _ = candidates(attach_targets(play_features(raw), targets, baseline))
    index = names.index("coverage_dynamics__opponent1_present")
    assert (missing[:, index] == 0).all()
    assert np.isfinite(missing).all()
