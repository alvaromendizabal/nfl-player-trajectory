"""Optional-input dependencies are checked against actual feature values."""

import numpy as np
import pandas as pd
import pytest
from test_feature_research import example_tracking

from nfl_trajectory.context_features import build_context_features, context_catalog
from nfl_trajectory.feature_candidates import candidate_matrix, research_catalog
from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
from nfl_trajectory.features import build_player_features
from nfl_trajectory.representation_features import (
    build_representation,
    fit_routes,
    representation_catalog,
    route_inputs,
)
from nfl_trajectory.research_inference import (
    input_variant,
    linear_correction,
    split_fitted_blocks,
)


def test_safe_contract_covers_transitive_dependencies_across_all_candidates():
    observed, targets = example_tracking()
    observed = observed.assign(
        s=np.linspace(2, 7, len(observed)),
        a=np.linspace(-1, 2, len(observed)),
        dir=np.linspace(20, 280, len(observed)),
        o=np.linspace(40, 200, len(observed)),
        player_height="6-2",
        player_weight=220,
        player_birth_date="1998-01-01",
        player_position="WR",
    )
    full = build_player_features(observed)
    routes = fit_routes(route_inputs(full))
    history = np.zeros((len(targets), 10))

    def matrix(frame):
        bank = build_player_features(frame)
        return np.column_stack(
            [
                candidate_matrix(bank, targets, history),
                build_context_features(frame, bank.state).matrix(targets),
                build_representation(bank, routes).matrix(targets),
            ]
        )

    catalog = pd.concat([research_catalog(), context_catalog(), representation_catalog()])
    safe = np.array(
        [not telemetry_dependent(n) and not metadata_dependent(n) for n in catalog.feature]
    )
    reference = matrix(observed)
    changed = observed.drop(
        columns=[
            "s",
            "a",
            "dir",
            "o",
            "player_height",
            "player_weight",
            "player_birth_date",
            "player_position",
        ]
    )
    actual = matrix(changed)
    assert np.any(reference[:, ~safe] != actual[:, ~safe])
    np.testing.assert_array_equal(actual[:, safe], reference[:, safe])


def test_fallback_routing_requires_a_fitted_safe_representation():
    data, _ = example_tracking()
    data = data.assign(s=3.0, a=1.0, dir=90.0, o=90.0)
    model = {"blocks": [{"model": {"features": ["history__s__lag_00"]}}]}
    assert input_variant(data, model) == "complete_inputs"
    data.loc[0, "s"] = np.nan
    with pytest.raises(ValueError, match="not fitted"):
        input_variant(data, model)
    model["fallbacks"] = {"without_telemetry": {}}
    assert input_variant(data, model) == "without_telemetry"
    model["blocks"][0]["model"]["features"] = ["history__vx__lag_00"]
    assert input_variant(data.drop(columns=["s", "a", "dir", "o"]), model) == "complete_inputs"


def test_joint_fallback_partition_preserves_the_single_fitted_intercept():
    names = [
        research_catalog().feature.iloc[0],
        context_catalog().feature.iloc[0],
        representation_catalog().feature.iloc[0],
    ]
    fitted = {
        "features": names,
        "mean": [1.0, 2.0, 3.0],
        "scale": [2.0, 3.0, 4.0],
        "coefficients": [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
        "intercept": [4.0, -5.0],
    }
    x = np.arange(18, dtype=float).reshape(6, 3)
    parts = split_fitted_blocks(fitted)
    actual = sum(
        linear_correction(x[:, [names.index(n) for n in b["model"]["features"]]], b["model"])
        for b in parts
    )
    np.testing.assert_allclose(actual, linear_correction(x, fitted), rtol=1e-14)
