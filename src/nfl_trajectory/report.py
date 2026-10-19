"""Small authored tracking fixture for independent numerical contract tests."""

from __future__ import annotations

import pandas as pd


def synthetic_play() -> tuple[pd.DataFrame, pd.DataFrame]:
    past, future = [], []
    for player, lateral in [(1, 20), (2, 25)]:
        for frame in range(1, 11):
            past.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 30 + 0.5 * frame,
                    "y": lateral + 0.1 * frame,
                }
            )
        for frame in range(1, 21):
            t = frame / 10
            future.append(
                {
                    "game_id": 2023090700,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 35 + 5 * t - 0.6 * t**2,
                    "y": lateral + 1 + t + (0.6 if player == 1 else -0.7) * t**2,
                }
            )
    return pd.DataFrame(past), pd.DataFrame(future)
