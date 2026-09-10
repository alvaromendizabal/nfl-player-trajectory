"""Reject feature drift and prove that paired motion ablations isolate their signals."""

from collections import Counter

import numpy as np
import pytest
from test_temporal_data import sample

from nfl_trajectory.domain_features import candidates
from nfl_trajectory.motion_research import GROUPS, feature_group, keep_columns


def test_partition_is_complete_and_add_remove_pairs_complement_each_other():
    _, names, _ = candidates(sample())
    names = ["control__000", *names]
    retained = [True] * len(names)
    counts = Counter(feature_group(n) for n in names)
    assert {g: counts[g] for g in GROUPS} == dict(zip(GROUPS, [5, 5, 30, 9, 8], strict=True))
    full = np.array([feature_group(n) in {"control", *GROUPS} for n in names])
    for group in GROUPS:
        added, removed = (keep_columns(names, retained, a) for a in (group, "without_" + group))
        np.testing.assert_array_equal(added | removed, full)
        np.testing.assert_array_equal(np.flatnonzero(added & removed), [0])
        retained[1] = False
        assert not keep_columns(names, retained, group)[1]
        retained[1] = True
    with pytest.raises(ValueError, match="unclassified"):
        feature_group("motion_state__unexpected_new_information")
