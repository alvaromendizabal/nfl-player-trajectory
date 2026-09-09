"""Leakage and schema contracts for learned and deterministic route representations."""

import copy

import numpy as np
import pytest
from test_feature_research import example_tracking

from nfl_trajectory.features import build_player_features
from nfl_trajectory.representation_features import (
    ROUTE_INPUTS,
    build_representation,
    fit_routes,
    representation_catalog,
    route_inputs,
    route_transform,
)


def test_representation_schema_target_poison_and_batch_parity():
    inputs, targets = example_tracking()
    bank = build_player_features(inputs)
    routes = fit_routes(route_inputs(bank))
    representation = build_representation(bank, routes)
    catalog = representation_catalog()
    assert len(catalog) == 1144
    assert catalog.feature.is_unique
    assert catalog[["rationale", "availability", "provenance"]].notna().all().all()
    targets = targets.sample(frac=1, random_state=2)
    expected = representation.matrix(targets)
    targets[["x", "y"]] = np.nan
    np.testing.assert_array_equal(representation.matrix(targets), expected)
    chosen = catalog.feature.iloc[[0, 185, 400, 1143]].tolist()
    actual = np.concatenate(
        [
            representation.matrix(targets.iloc[:13], chosen),
            representation.matrix(targets.iloc[13:], chosen),
        ]
    )
    np.testing.assert_array_equal(actual, expected[:, [0, 185, 400, 1143]])


def test_route_encoder_is_frozen_deterministic_and_rejects_invalid_input():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(45, len(ROUTE_INPUTS)))
    fitted = fit_routes(train)
    assert fitted == fit_routes(train)
    before = copy.deepcopy(fitted)
    first = route_transform(train[:3], fitted)
    heldout = train[3:6] + 1000
    route_transform(heldout, fitted)
    assert fitted == before
    np.testing.assert_array_equal(first, route_transform(train[:3], fitted))
    assert fitted["training_entities"] == 45
    with pytest.raises(ValueError, match="finite"):
        fit_routes(np.full_like(train, np.nan))
    fitted["inputs"] = []
    with pytest.raises(ValueError, match="schema"):
        route_transform(train, fitted)


def test_absent_neighbours_have_zero_pooling_and_destination_masks():
    inputs, targets = example_tracking()
    full = build_player_features(inputs)
    fitted = fit_routes(route_inputs(full))
    bank = build_player_features(inputs[inputs.nfl_id.eq(2)])
    targets = targets[targets.nfl_id.eq(2)]
    representation = build_representation(bank, fitted)
    catalog = representation_catalog()
    columns = catalog.loc[
        catalog.family.eq("graph_pool") | catalog.feature.str.contains("destination__receiver"),
        "feature",
    ].tolist()
    np.testing.assert_array_equal(representation.matrix(targets, columns), 0)
