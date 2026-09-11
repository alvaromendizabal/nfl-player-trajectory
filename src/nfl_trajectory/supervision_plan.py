"""Training-only batching and normalization contracts for matched motion research."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from nfl_trajectory.motion_targets import motion_targets


def training_plan(
    samples: list[dict[str, Any]], batch_plays: int = 64, seed: int = 2026, epoch: int = 0
) -> dict[str, Any]:
    """Plan equal-exposure arms without looking at validation labels or fitting a model.

    Input order is canonicalized by play identity. The returned order is a
    training-play permutation, not a data-dependent feature-selection ranking.
    Fixed denominators keep a uniformly sampled batch's expected gradient
    proportional to the global coordinate objective despite unequal play lengths.
    """
    if not samples or batch_plays < 1 or seed < 0 or epoch < 0:
        raise ValueError("Nonempty training samples and valid batch/seed/epoch are required.")
    if any(str(s["split"]) != "train" for s in samples):
        raise ValueError("Only training samples may determine the plan.")
    ordered = sorted(samples, key=lambda s: tuple(s["keys"][0, :2]))
    plays = [tuple(int(x) for x in s["keys"][0, :2]) for s in ordered]
    if len(set(plays)) != len(plays):
        raise ValueError("Duplicate training play identity.")
    rows, supported, velocity_sse, long_rows = [], [], 0.0, 0
    for sample in ordered:
        labels = motion_targets(sample)
        valid = labels["velocity_mask"]
        velocity_sse += float(np.square(labels["velocity"][valid]).sum())
        rows.append(len(sample["keys"]))
        supported.append(int(valid.sum()))
        long_rows += int((sample["keys"][:, 3] > 48).sum())
    coordinates, velocity_coordinates = 2 * sum(rows), 2 * sum(supported)
    if not coordinates or not velocity_coordinates:
        raise ValueError("Training coordinates and supported velocity labels are required.")
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(ordered))
    batches = math.ceil(len(ordered) / batch_plays)
    batch_counts = [
        2 * sum(rows[int(i)] for i in order[start : start + batch_plays])
        for start in range(0, len(order), batch_plays)
    ]
    return {
        "training_plays": len(ordered),
        "training_games": len({p[0] for p in plays}),
        "training_rows": sum(rows),
        "training_rows_after_frame_48": long_rows,
        "velocity_coordinates": velocity_coordinates,
        "velocity_support_fraction": sum(supported) / sum(rows),
        "training_only_velocity_rms": max(0.1, math.sqrt(velocity_sse / velocity_coordinates)),
        "batch_plays": batch_plays,
        "batches_per_epoch": batches,
        "coordinate_loss_denominator": coordinates / batches,
        "velocity_loss_denominator": velocity_coordinates / batches,
        "batch_coordinate_counts": batch_counts,
        "canonical_play_ids": [list(p) for p in plays],
        "permutation": order.tolist(),
        "permutation_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
        "seed": seed,
        "epoch": epoch,
        "scientific_fits": 0,
        "validation_statistics_used": False,
        "feature_completion_gate": "open",
        "epochs_and_lr_schedule_frozen": False,
        "training_ready": False,
    }
