"""Matched temporal feature experiments with fold-local label-derived inputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_candidates import HISTORY_NAMES
from nfl_trajectory.motion import ENTITY, KEYS
from nfl_trajectory.runtime import sha256
from nfl_trajectory.temporal_data import STATIC_NAMES, attach_targets

TARGET_STATISTICS = [12, 13, 14, 17, 18, 19]
VARIANTS = ("full", "without_target_statistics")


def verified_samples(root: Path) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Read only tensors sealed by the completed temporal experiment."""
    folder = root / "artifacts/temporal/development"
    trial = json.loads((folder / "summary.json").read_text())
    if trial["status"] != "completed":
        raise ValueError("A completed temporal reference is required.")
    for name, digest in trial["artifacts"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Temporal reference artifact changed: {name}")
    plan = json.loads((folder / "plan.json").read_text())
    for name, digest in plan["inputs"].items():
        if name.startswith(("src/", "scripts/")) and sha256(root / name) != digest:
            raise ValueError(f"Temporal numerical source changed: {name}")
    samples = []
    for name in sorted(trial["artifacts"]):
        if name.endswith(".pkl") and "/weeks/" in name:
            samples.extend(pickle.loads((root / name).read_bytes()))
    return samples, trial


def ablate(sample: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    """Remove outcomes' means/dispersion; retain counts, cold flags and dimensions."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown temporal feature variant: {variant}")
    result = dict(sample)
    result["static"] = sample["static"].copy()
    if variant == "without_target_statistics":
        result["static"][:, TARGET_STATISTICS] = 0
    return result


def validate_fold(fold: dict[str, Any], splits: pd.DataFrame) -> None:
    """Inner research must stay inside the original training dates."""
    training, evaluation = set(fold["training_games"]), set(fold["evaluation_games"])
    permitted = set(splits.loc[splits.split.eq("train"), "game_id"])
    if not training or not evaluation or training & evaluation:
        raise ValueError("A fold requires nonempty disjoint training and evaluation games.")
    if not (training | evaluation).issubset(permitted):
        raise ValueError("Inner folds cannot include development or holdout games.")
    if max(training) // 100 >= min(evaluation) // 100:
        raise ValueError("Fold evaluation must follow training on strictly later dates.")


def rebase_sample(
    sample: dict[str, np.ndarray],
    targets: pd.DataFrame,
    baseline: np.ndarray,
    truth: np.ndarray,
    priors: pd.DataFrame,
    split: str,
) -> dict[str, np.ndarray]:
    """Replace every full-training label-derived field before using an inner fold."""
    if not np.array_equal(targets[KEYS].to_numpy(), sample["keys"]):
        raise ValueError("Fold rebase requires identical ordered forecast keys.")
    if split not in {"train", "validation"}:
        raise ValueError("Only training and evaluation examples may enter fold fitting.")
    result = dict(sample)
    static = sample["static"].copy()
    static[:, 10:] = 0
    static[:, [11, 16]] = 1
    entities = pd.DataFrame(
        {
            "game_id": int(targets.game_id.iloc[0]),
            "play_id": int(targets.play_id.iloc[0]),
            "nfl_id": sample["ids"],
        }
    )
    joined = entities.merge(priors, on=ENTITY, how="left", sort=False, validate="one_to_one")
    present = joined[HISTORY_NAMES].notna().all(axis=1).to_numpy()
    static[present, 10:] = joined.loc[present, HISTORY_NAMES].to_numpy(np.float32)
    if not np.isfinite(static).all():
        raise ValueError("Fold-local historical features must be finite.")
    result["static"] = static
    result = attach_targets(result, targets, baseline, truth)
    result["split"] = np.array(split)
    return result


def channel_audit() -> dict[str, Any]:
    return {
        "variants": list(VARIANTS),
        "target_derived_channels": [STATIC_NAMES[i] for i in TARGET_STATISTICS],
        "retained_count_and_cold_channels": [STATIC_NAMES[i] for i in (10, 11, 15, 16)],
        "static_channels_full": len(STATIC_NAMES),
        "static_channels_after_ablation": len(STATIC_NAMES) - len(TARGET_STATISTICS),
        "parameter_count": "identical; removed input channels are fixed to zero",
    }
