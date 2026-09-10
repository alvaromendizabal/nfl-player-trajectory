"""Bounded chronological motion-history screen with separately checkpointed arms."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soft_coverage import comparison, fit_ridge, predict, save_arrays  # noqa: E402

from nfl_trajectory.conditional_history import (  # noqa: E402
    NAMES,
    fit_transform,
    observed_rows,
    training_targets,
)
from nfl_trajectory.runtime import Run, atomic_json, sha256, stage  # noqa: E402

ARMS = ("control_motion", "coarse_history", "context_history")


def execute(source_commit: str, phase: str, run: Run) -> None:
    folder = ROOT / "artifacts/conditional_history/inner_1"
    parent = ROOT / "artifacts/domain_research/inner_1"
    old = ROOT / "artifacts/soft_coverage"
    parent_plan = json.loads((old / "plan.json").read_text())
    for name, expected in parent_plan["inputs"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError("Parent input differs from its preserved hash.")
    source_names = [
        "scripts/conditional_history.py",
        "src/nfl_trajectory/conditional_history.py",
        "scripts/soft_coverage.py",
        "src/nfl_trajectory/runtime.py",
        "tests/test_conditional_history.py",
        "docs/CONDITIONAL_HISTORY_EXPERIMENT.md",
    ]
    old_report = json.loads((ROOT / "docs/results/soft_coverage.json").read_text())
    control_paths = [old / "inner_1/control_motion.npz", old / "inner_1/control_motion_errors.npz"]
    for path in control_paths:
        if sha256(path) != old_report["artifacts"][str(path.relative_to(ROOT))]:
            raise ValueError("Saved control checkpoint or predictions changed.")
    packages = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "pandas", "scipy", "filelock", "threadpoolctl")
    }
    if packages != parent_plan["packages"]:
        raise ValueError("Control replay requires the preserved numerical runtime.")
    spec = {
        "source_commit": source_commit,
        "inputs": parent_plan["inputs"],
        "control": {str(p.relative_to(ROOT)): sha256(p) for p in control_paths},
        "sources": {p: sha256(ROOT / p) for p in source_names},
        "arms": ARMS,
        "l2": 0.01,
        "threads": 2,
        "max_seconds_per_phase": 300,
        "python": platform.python_version(),
        "packages": packages,
    }
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    plan = folder.parent / "plan.json"
    if plan.exists() and json.loads(plan.read_text())["signature"] != signature:
        raise ValueError("Existing study has different code, data or runtime.")
    if not plan.exists():
        atomic_json(plan, {**spec, "signature": signature})
    data = dict(np.load(parent / "features.npz"))
    train = data["train"]
    keys = data["keys"]
    train_games, eval_games = np.unique(keys[train, 0]), np.unique(keys[~train, 0])
    if len(train_games) != 94 or len(eval_games) != 41 or int((~train).sum()) != 83938:
        raise ValueError("Unexpected fold membership or support.")
    if max(train_games // 100) >= min(eval_games // 100):
        raise ValueError("Dates overlap.")
    fold = json.loads((ROOT / "artifacts/temporal/research/plan.json").read_text())["folds"][0]
    if set(map(int, fold["training_games"])) != set(train_games):
        raise ValueError("Training games differ from the frozen chronological plan.")
    schema = json.loads((parent / "schema.json").read_text())
    selected = np.array([f in ("control", "motion_state") for f in schema["families"]])
    control = data["x"][:, selected]
    saved_control = dict(np.load(control_paths[0]))
    saved_errors = dict(np.load(control_paths[1]))
    np.testing.assert_array_equal(saved_errors["keys"], keys[~train])
    replay = predict(saved_control, control[~train], data["time"][~train]) - data["y"][~train]
    np.testing.assert_allclose(replay, saved_errors["errors"], rtol=0, atol=1e-10)
    run.event("control_verified", reused_fit=True, rmse=float(np.sqrt(np.mean(replay**2))))
    feature_path, history_path = folder / "features.npz", folder / "history.json"

    def prepare() -> None:
        samples = pickle.loads(
            (ROOT / "artifacts/temporal/research/inner_1/samples.pkl").read_bytes()
        )
        parts, labels, flags, ordered_keys = [], [], [], []
        for split in ("train", "validation"):
            for sample in samples:
                if sample["split"] != split:
                    continue
                rows = observed_rows(sample)
                # No evaluation outcome is passed to the history builder.
                target = (
                    training_targets(sample, rows)
                    if split == "train"
                    else np.full((len(rows), 2), np.nan)
                )
                parts.append(rows)
                labels.append(target)
                ordered_keys.append(sample["keys"])
                flags.append(np.full(len(rows), split == "train"))
                if len(parts) % 500 == 0:
                    run.event("history_covariates", plays=len(parts), total=len(samples))
        np.testing.assert_array_equal(np.concatenate(ordered_keys), keys)
        np.testing.assert_array_equal(np.concatenate(flags), train)
        rows = pd.concat(parts, ignore_index=True)
        features, history = fit_transform(rows, np.concatenate(labels), train)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite conditional features.")
        save_arrays(feature_path, x=features, names=np.array(NAMES))
        atomic_json(history_path, history)

    stage(
        ROOT, "conditional-history-features", signature, [feature_path, history_path], prepare, run
    )
    if phase == "prepare":
        return
    extra = dict(np.load(feature_path))
    results = {ARMS[0]: saved_errors["errors"]}
    counts = {
        ARMS[0]: {"candidates": control.shape[1], "retained": int(saved_control["retained"].sum())}
    }
    for arm, width in zip(ARMS[1:], (5, 20), strict=True):
        model_path, error_path = folder / (arm + ".npz"), folder / (arm + "_errors.npz")
        if phase in ("report", "replay") and not (model_path.exists() and error_path.exists()):
            raise ValueError("Complete and durably save each arm before reporting or replay.")
        if phase not in (arm, "report", "replay"):
            continue
        x = np.column_stack([control, extra["x"][:, :width]])

        def fit(x: np.ndarray = x, path: Path = model_path) -> None:
            if phase in ("report", "replay"):
                raise ValueError("Reporting/replay must never perform model fitting.")
            save_arrays(path, **fit_ridge(x[train], data["y"][train], data["time"][train]))

        stage(ROOT, "conditional-history-fit-" + arm, signature, [model_path], fit, run)
        model = dict(np.load(model_path))

        def evaluate(x: np.ndarray = x, model: Any = model, path: Path = error_path) -> None:
            errors = predict(model, x[~train], data["time"][~train]) - data["y"][~train]
            save_arrays(path, errors=errors, keys=keys[~train])

        stage(ROOT, "conditional-history-eval-" + arm, signature, [error_path], evaluate, run)
        results[arm] = np.load(error_path)["errors"]
        counts[arm] = {"candidates": x.shape[1], "retained": int(model["retained"].sum())}
        run.event("arm_completed", arm=arm, rmse=float(np.sqrt(np.mean(results[arm] ** 2))))
    if phase not in ("report", "replay"):
        return
    report_path = ROOT / "docs/results/conditional_history.json"

    def report() -> None:
        primary = comparison(results[ARMS[0]], results[ARMS[2]], keys[~train, 0])
        contextual = comparison(results[ARMS[1]], results[ARMS[2]], keys[~train, 0])
        keep = primary["continue_gate"] and contextual["relative_reduction"] > 0
        samples = pickle.loads(
            (ROOT / "artifacts/temporal/research/inner_1/samples.pkl").read_bytes()
        )
        role = np.concatenate(
            [s["role"][s["player"]] for s in samples if s["split"] == "validation"]
        )
        slices = {}
        for name, selector in {
            **{f"role_{r}": role == r for r in np.unique(role)},
            "horizon_le_1s": data["time"][~train] <= 1,
            "horizon_1_2s": (data["time"][~train] > 1) & (data["time"][~train] <= 2),
            "horizon_gt_2s": data["time"][~train] > 2,
        }.items():
            slices[name] = {
                "rows": int(selector.sum()),
                "rmse": {a: float(np.sqrt(np.mean(e[selector] ** 2))) for a, e in results.items()},
            }
        atomic_json(
            report_path,
            {
                "status": "completed",
                "signature": signature,
                "source_commit": source_commit,
                "training_games": 94,
                "evaluation_games": 41,
                "training_rows": int(train.sum()),
                "evaluation_rows": int((~train).sum()),
                "candidate_counts": counts,
                "scores": {a: float(np.sqrt(np.mean(e**2))) for a, e in results.items()},
                "primary": primary,
                "context_vs_coarse": contextual,
                "decision": "continue_to_folds" if keep else "stop_this_probe",
                "new_fit_count": 2,
                "reused_fit_count": 1,
                "slices": slices,
                "artifacts": {
                    str(p.relative_to(ROOT)): sha256(p) for p in folder.iterdir() if p.is_file()
                },
                "kaggle_private_rmse_unchanged": 0.7009,
                "feature_gate": "open",
                "limitations": [
                    "one reused chronological fold",
                    "linear correction interface",
                    "in-sample parent residuals",
                    "single season",
                    "adaptive research hypothesis",
                ],
            },
        )

    stage(ROOT, "conditional-history-report", signature, [report_path], report, run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--phase", choices=("prepare", *ARMS[1:], "report", "replay"), required=True
    )
    args = parser.parse_args()

    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(
            "Conditional history phase exceeded 300 seconds; saved stages are reusable."
        )

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    with FileLock(str(ROOT / ".conditional-history.lock"), timeout=1), threadpool_limits(limits=2):
        with Run(ROOT, "conditional-history-" + args.phase) as run:
            execute(args.source_commit, args.phase, run)


if __name__ == "__main__":
    main()
