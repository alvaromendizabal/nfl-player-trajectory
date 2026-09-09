"""Wide ablations cover the complete catalog without changing the availability policy."""

import pandas as pd
import pytest

from nfl_trajectory.context_features import context_catalog
from nfl_trajectory.feature_candidates import research_catalog
from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
from nfl_trajectory.representation_features import representation_catalog
from nfl_trajectory.wide_ablation import GROUPS, retained_indices


def catalog():
    return pd.concat(
        [research_catalog(), context_catalog(), representation_catalog()], ignore_index=True
    )


def test_predeclared_groups_partition_every_catalog_family():
    covered = [family for group in GROUPS.values() for family in group]
    assert len(covered) == len(set(covered)) == 20
    assert set(covered) == set(catalog().family)


@pytest.mark.parametrize("profile", ["without_metadata", "without_optional_inputs"])
def test_group_removal_preserves_parent_order_and_cannot_refill(profile):
    c = catalog()
    names = list(reversed(c.feature.tolist()))
    families = dict(zip(c.feature, c.family, strict=True))
    parent = retained_indices(names, families, profile, set())
    for removed in GROUPS.values():
        kept = retained_indices(names, families, profile, removed)
        assert kept == [i for i in parent if families[names[i]] not in removed]
        assert all(not metadata_dependent(names[i]) for i in kept)
        if profile == "without_optional_inputs":
            assert all(not telemetry_dependent(names[i]) for i in kept)


def test_unknown_availability_profile_is_rejected():
    with pytest.raises(ValueError, match="availability profile"):
        retained_indices([], {}, "zero_fill", set())


def test_duplicate_or_unknown_columns_cannot_enter_a_refit():
    for names, families in [(["a", "a"], {"a": "motion"}), (["a"], {})]:
        with pytest.raises(ValueError, match="Unique features"):
            retained_indices(names, families, "without_metadata", set())
