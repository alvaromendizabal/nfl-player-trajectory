"""Label-independent, resumable play order and reflection for matched training."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nfl_trajectory.temporal_data import reflect


class TrainingBatches:
    """Map a global optimizer cursor to the same examples in either arm."""

    def __init__(self, samples: list[dict[str, Any]], batch_plays: int, seed: int = 2026) -> None:
        if not samples or batch_plays < 1 or seed < 0:
            raise ValueError("Nonempty training data, positive batch size and seed are required.")
        if any(str(s["split"]) != "train" for s in samples):
            raise ValueError("Training batches reject evaluation samples before label access.")
        self.samples = sorted(samples, key=lambda s: tuple(s["keys"][0, :2]))
        ids = [tuple(int(i) for i in s["keys"][0, :2]) for s in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate play identity in training batches.")
        self.batch_plays, self.seed = batch_plays, seed
        self.batches_per_epoch = math.ceil(len(samples) / batch_plays)
        self._epoch = -1
        self._order = np.empty(0, dtype=np.int64)
        self._reflections = np.empty(0, dtype=bool)

    def selection(self, cursor: int) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical indices and stateless, play-indexed reflection flags."""
        if cursor < 0:
            raise ValueError("Negative optimizer cursor.")
        epoch, batch = divmod(cursor, self.batches_per_epoch)
        if epoch != self._epoch:
            self._order = np.random.default_rng(
                np.random.SeedSequence([self.seed, epoch])
            ).permutation(len(self.samples))
            self._reflections = (
                np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 1]))
                .integers(0, 2, len(self.samples), dtype=np.int8)
                .astype(bool)
            )
            self._epoch = epoch
        start = batch * self.batch_plays
        indices = self._order[start : start + self.batch_plays]
        return indices.copy(), self._reflections[indices].copy()

    def batch(self, cursor: int) -> list[dict[str, Any]]:
        indices, flags = self.selection(cursor)
        return [
            reflect(self.samples[int(i)]) if flip else self.samples[int(i)]
            for i, flip in zip(indices, flags, strict=True)
        ]


def scheduled_learning_rate(
    cursor: int, total_steps: int, warmup_steps: int, peak: float = 0.001, floor: float = 0.00001
) -> float:
    """One fixed warmup/cosine schedule, indexed by the saved optimizer step."""
    if not 0 <= cursor < total_steps or not 1 <= warmup_steps < total_steps - 1:
        raise ValueError("Invalid cursor or warmup/total exposure.")
    if not 0 < floor <= peak or not math.isfinite(peak):
        raise ValueError("Finite positive learning rates are required.")
    if cursor < warmup_steps:
        return peak * (cursor + 1) / warmup_steps
    progress = (cursor - warmup_steps) / (total_steps - warmup_steps - 1)
    return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2
