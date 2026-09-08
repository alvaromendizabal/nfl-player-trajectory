"""Predeclared, disjoint football feature groups for the deployed representation."""

from __future__ import annotations

from collections.abc import Mapping

from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent

GROUPS = {
    "observed_histories": {"history_lags", "history_summaries", "robust_history"},
    "multiscale_transforms": {"multiscale", "nonlinear"},
    "coverage_geometry": {"interactions", "projected_geometry", "matched_history", "peer_context"},
    "forecast_crosses": {"forecast_interactions"},
    "role_responses": {"role_response"},
    "arrival_feasibility": {"reachability"},
    "destination_coupling": {"role_destination"},
    "player_set_pools": {"graph_pool"},
    "route_representation": {"route_representation"},
    "historical_priors": {"player_history"},
    "task_context_masks": {"context", "forecast", "availability"},
    "body_position_metadata": {"metadata"},
}


def retained_indices(
    names: list[str], families: Mapping[str, str], profile: str, removed: set[str]
) -> list[int]:
    """Keep original column order and availability policy; never refill a dropped group."""
    if profile not in {"without_metadata", "without_optional_inputs"}:
        raise ValueError("The ablation requires the selected, fitted availability profile.")
    if len(set(names)) != len(names) or not set(names) <= families.keys():
        raise ValueError("Unique features and a complete family dictionary are required.")
    return [
        i
        for i, name in enumerate(names)
        if families[name] not in removed
        and not metadata_dependent(name)
        and (profile != "without_optional_inputs" or not telemetry_dependent(name))
    ]
