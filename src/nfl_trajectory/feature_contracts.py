"""Explicit dependencies for optional telemetry and player metadata."""

from __future__ import annotations

import pandas as pd

TELEMETRY_COLUMNS = ("s", "a", "dir", "o")
METADATA_COLUMNS = ("player_height", "player_weight", "player_birth_date", "player_position")
TELEMETRY_TERMS = {
    "s",
    "a",
    "dir_sin",
    "dir_cos",
    "o_sin",
    "o_cos",
    "orientation_alignment",
    "telemetry",
    "reported_vx_gap",
    "reported_vy_gap",
    "heading_cross",
}


def telemetry_dependent(feature: str) -> bool:
    """Catalog names retain primitive channel tokens through every deterministic transform."""
    return bool(set(feature.split("__")) & TELEMETRY_TERMS)


def metadata_dependent(feature: str) -> bool:
    return "metadata" in feature.split("__")


def family_missing(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    """Missing fields or wholly absent channels are outside the trained full-input contract."""
    return any(column not in frame or not frame[column].notna().any() for column in columns)
