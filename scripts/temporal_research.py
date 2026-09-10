# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "plotly==7.0.0", "matplotlib==3.10.8",
#   "filelock==3.32.5", "torch==2.8.0", "pytest==9.1.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Confirm the temporal representation and isolate target statistics on inner folds.

Run with: uv run --locked scripts/temporal_research.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_temporal as trainer  # noqa: E402

from nfl_trajectory.benchmark import error_metrics  # noqa: E402
from nfl_trajectory.feature_research import (  # noqa: E402
    baseline_prediction,
    chronological_folds,
    fit_baseline,
    fold_history,
    verify_inputs,
    weeks,
)
from nfl_trajectory.motion import KEYS  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage  # noqa: E402
from nfl_trajectory.temporal_research import (  # noqa: E402
    VARIANTS,
    ablate,
    channel_audit,
    rebase_sample,
    validate_fold,
    verified_samples,
)


def prepare_fold(
    fold: dict[str, Any],
    samples: list[dict[str, np.ndarray]],
    caches: list[Path],
    folder: Path,
    signature: str,
    run: Run,
) -> list[dict[str, np.ndarray]]:
    training, evaluation = set(fold["training_games"]), set(fold["evaluation_games"])
    fitted_path = folder / "baseline.json"
    history_path = folder / "history.csv"
    encoder_path = folder / "history_model.json"

    def fit() -> None:
        fitted = fit_baseline(ROOT, caches, training)
        history, encoder = fold_history(ROOT, caches, training, evaluation)
        atomic_json(fitted_path, fitted)
        atomic_bytes(history_path, history.to_csv(index=False).encode())
        atomic_json(encoder_path, encoder)

    stage(
        ROOT,
        fold["name"] + "-temporal-priors",
        signature,
        [fitted_path, history_path, encoder_path],
        fit,
        run,
    )
    fitted = json.loads(fitted_path.read_text())
    if fitted["training_games"] != fold["training_games"]:
        raise ValueError("Fold baseline was not fitted on the exact training games.")
    priors = pd.read_csv(history_path)
    prior_groups = dict(tuple(priors.groupby(KEYS[:2], sort=False)))
    observed = {tuple(s["keys"][0, :2]): s for s in samples}
    destination = folder / "samples.pkl"

    def build() -> None:
        result = []
        for _, targets, arrays, state, basis in weeks(ROOT, caches, training | evaluation):
            baseline = baseline_prediction(state, basis, fitted)
            for key, indices in targets.groupby(KEYS[:2], sort=True).indices.items():
                result.append(
                    rebase_sample(
                        observed[key],
                        targets.iloc[indices],
                        baseline[indices],
                        arrays["truth"][indices],
                        prior_groups[key],
                        "train" if key[0] in training else "validation",
                    )
                )
        if {int(s["keys"][0, 0]) for s in result} != training | evaluation:
            raise ValueError("Fold tensors must cover exactly the declared games.")
        atomic_bytes(destination, pickle.dumps(result, protocol=5))

    stage(ROOT, fold["name"] + "-temporal-samples", signature, [destination], build, run)
    return pickle.loads(destination.read_bytes())


def execute_arm(
    fold: dict[str, Any],
    variant: str,
    samples: list[dict[str, np.ndarray]],
    folder: Path,
    signature: str,
    run: Run,
) -> dict[str, Any]:
    arm = folder / variant
    arm.mkdir(parents=True, exist_ok=True)
    arm_signature = hashlib.sha256((signature + variant).encode()).hexdigest()
    result_path = arm / "summary.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["signature"] == arm_signature and old["status"] == "completed":
            if all(sha256(arm / name) == digest for name, digest in old["artifacts"].items()):
                run.event("arm_reused", fold=fold["name"], variant=variant)
                return old
            raise ValueError("Completed ablation checkpoint or prediction hash changed.")
    changed = [ablate(s, variant) for s in samples]
    model, state = trainer.train(arm, changed, arm_signature, run)
    if state["epoch"] != trainer.SETTINGS["epochs"]:
        raise RuntimeError("Bounded arm stopped before all predeclared epochs; no result selected.")
    errors = trainer.evaluate(model, [s for s in changed if s["split"] == "validation"])
    atomic_bytes(arm / "errors.csv", errors.to_csv(index=False).encode())
    result = {
        "status": "completed",
        "signature": arm_signature,
        "fold": fold,
        "variant": variant,
        "settings": trainer.SETTINGS,
        "run_id": run.run_id,
        "training_seconds": state["seconds"],
        "epochs_completed": state["epoch"],
        "curve": state["curve"],
        "evaluation_rows": len(errors),
        **error_metrics(errors),
        "artifacts": {
            name: sha256(arm / name) for name in ("checkpoint.pt", "checkpoint.json", "errors.csv")
        },
    }
    atomic_json(result_path, result)
    run.event("arm_completed", fold=fold["name"], variant=variant, **error_metrics(errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["all", "inner_1", "inner_2", "inner_3"], default="all")
    args = parser.parse_args()
    folder = ROOT / "artifacts/temporal/research"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1), Run(ROOT, "temporal-research") as run:
        samples, trial = verified_samples(ROOT)
        caches, inputs = verify_inputs(ROOT)
        splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
        folds = chronological_folds(splits)
        sources = [
            Path(__file__),
            Path(__file__).with_suffix(".py.lock"),
            ROOT / "src/nfl_trajectory/temporal_research.py",
        ]
        specification = {
            "protocol_version": 1,
            "folds": folds,
            "settings": trainer.SETTINGS,
            "reference_signature": trial["source_signature"],
            "inputs": inputs,
            "sources": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
            "feature_ablation": channel_audit(),
            "selection": "final epoch EMA; identical fixed schedule and seed; no fold tuning",
            "ensemble": "fixed equal weights; compare all three folds before any promotion",
            "scope": "original training games only; no development, holdout or Kaggle selection",
            "limitation": "one seed; historical folds previously inspected during tree research",
        }
        signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        plan_path = folder / "plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text())["signature"] != signature:
            raise ValueError(
                "Research specification changed; do not overwrite existing experiments."
            )
        atomic_json(plan_path, {"signature": signature, **specification})
        for fold in folds:
            if args.fold != "all" and args.fold != fold["name"]:
                continue
            validate_fold(fold, splits)
            destination = folder / fold["name"]
            destination.mkdir(parents=True, exist_ok=True)
            prepared = prepare_fold(fold, samples, caches, destination, signature, run)
            for variant in VARIANTS:
                execute_arm(fold, variant, prepared, destination, signature, run)


if __name__ == "__main__":
    main()
