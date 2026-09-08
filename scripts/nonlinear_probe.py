# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Fixed-capacity nonlinear feature checks; no tuning, final training or holdout score."""

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

SETTINGS = {
    "loss": "squared_error",
    "learning_rate": 0.07,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "max_depth": 4,
    "min_samples_leaf": 60,
    "l2_regularization": 1.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": 2026,
}


def train_pair(x: np.ndarray, y: np.ndarray) -> list[Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    if not np.isfinite(x).all() or not np.isfinite(y).all() or y.shape != (len(x), 2):
        raise ValueError("Nonlinear probe requires finite aligned two-coordinate training rows.")
    return [HistGradientBoostingRegressor(**SETTINGS).fit(x, y[:, axis]) for axis in range(2)]


def probe_summary(errors: pd.DataFrame, variants: dict[str, list[str]]) -> dict[str, Any]:
    from nfl_trajectory.benchmark import bootstrap_scores, summarize

    summary = summarize(errors)
    reference = bootstrap_scores(errors[errors.model.eq("landing_features")])
    for row in summary["models"]:
        row["feature_count"] = len(variants.get(row["model"], []))
        row["delta_vs_landing_ci95"] = np.quantile(
            bootstrap_scores(errors[errors.model.eq(row["model"])]) - reference, [0.025, 0.975]
        ).tolist()
    return summary


def feature_sets(
    core: dict[str, Any], context: dict[str, Any], rep: dict[str, Any]
) -> dict[str, list[str]]:
    landing = core["landing"]["features"]
    balanced = list(dict.fromkeys(landing + core["additions"]["plus_balanced"]["features"]))
    context_names = context["additions"]["plus_context"]["features"]
    representation_names = rep["additions"]["plus_representation"]["features"]
    return {
        "landing_features": landing,
        "engineered_core": balanced,
        "core_plus_context": balanced + context_names,
        "core_plus_representation": balanced + representation_names,
        "all_engineered": balanced + context_names + representation_names,
        "core_plus_noise": balanced + [f"negative_control_{i:02d}" for i in range(64)],
    }


def materialize(
    root: Path, caches: list[Path], fold: dict[str, Any], names: list[str], training: bool
) -> tuple[Any, ...]:
    from nfl_trajectory.context_experiment import context_weeks
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import candidate_matrix, research_catalog
    from nfl_trajectory.feature_experiment import target_state
    from nfl_trajectory.feature_research import BATCH, baseline_prediction, history_rows
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.representation_features import RepresentationBank, representation_catalog

    core_folder = root / "artifacts/research" / fold["name"]
    core = json.loads((core_folder / "models.json").read_text())
    history = pd.read_csv(core_folder / "history.csv")
    groups = {
        "research": set(research_catalog().feature),
        "context": set(context_catalog().feature),
        "representation": set(representation_catalog().feature),
    }
    positions = {name: i for i, name in enumerate(names)}
    requested = {kind: [n for n in names if n in group] for kind, group in groups.items()}
    noise = [n for n in names if n.startswith("negative_control_")]
    if len(set().union(*(set(v) for v in requested.values()), set(noise))) != len(names):
        raise ValueError("Unknown nonlinear probe feature.")
    rng = np.random.default_rng(1234 if training else 9876)
    matrices, residuals, baselines, truths, states, signs = [], [], [], [], [], []
    games = set(fold["training_games"] if training else fold["evaluation_games"])
    for weekly, bank, targets, arrays, state, basis, context in (
        (cache, *values) for cache in caches for values in context_weeks(root, [cache], games)
    ):
        with np.load(
            root / "artifacts/representation" / fold["name"] / (weekly.parent.name + ".npz"),
            allow_pickle=False,
        ) as saved:
            representation = RepresentationBank(bank.state, saved["values"])
        x = np.empty((len(targets), len(names)), dtype=np.float32)
        h = history_rows(history, targets)
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            for kind, columns in requested.items():
                if not columns:
                    continue
                values = (
                    candidate_matrix(bank, targets.iloc[start:end], h[start:end], columns)
                    if kind == "research"
                    else (context if kind == "context" else representation).matrix(
                        targets.iloc[start:end], columns
                    )
                )
                x[start:end, [positions[c] for c in columns]] = values
        if noise:
            # A diagnostic only: independent noise in training and evaluation, never selected.
            x[:, [positions[n] for n in noise]] = rng.normal(size=(len(x), len(noise)))
        baseline = baseline_prediction(state, basis, core["baseline"])
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        matrices.append(x)
        residuals.append((arrays["truth"] - baseline) * sign)
        baselines.append(baseline)
        truths.append(arrays["truth"])
        signs.append(sign)
        states.append(state[KEYS + ["player_role", "num_frames_output"]])
    return (
        np.concatenate(matrices),
        np.concatenate(residuals),
        np.concatenate(baselines),
        np.concatenate(truths),
        np.concatenate(signs),
        pd.concat(states, ignore_index=True),
    )


def nonlinear_probe(root: Path, run: Any) -> None:
    import sklearn
    from filelock import FileLock

    from nfl_trajectory.feature_research import verify_inputs, weeks
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.runtime import atomic_bytes, atomic_json, sha256, stage

    destination = root / "artifacts/nonlinear_probe"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        snapshot = feature_research_snapshot(root)
        caches, _ = verify_inputs(root)
        core = json.loads((root / "artifacts/research/summary.json").read_text())
        folds = [r["fold"] for r in core["inner_folds"]] + [core["fold"]]
        plan = {
            "representation_sources": snapshot["source_signatures"],
            "settings": SETTINGS,
            "script": sha256(Path(__file__)),
            "lock": sha256(Path(__file__).with_suffix(".py.lock")),
            "sklearn_version": sklearn.__version__,
            "folds": folds,
            "selection": "pooled inner RMSE; noise is excluded",
            "holdout_evaluation": "not_run",
        }
        source = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        plan["source_signature"] = source
        atomic_json(destination / "plan.json", plan)
        inner = []
        for fold in folds:
            folder = destination / fold["name"]
            core_models = json.loads(
                (root / "artifacts/research" / fold["name"] / "models.json").read_text()
            )
            context = json.loads(
                (root / "artifacts/context" / fold["name"] / "models.json").read_text()
            )
            rep = json.loads(
                (root / "artifacts/representation" / fold["name"] / "models.json").read_text()
            )
            variants = feature_sets(core_models, context, rep)
            names = sorted({c for v in variants.values() for c in v})
            positions = {name: i for i, name in enumerate(names)}
            training = materialize(root, caches, fold, names, True)
            evaluation = materialize(root, caches, fold, names, False)
            results = []
            for name, columns in variants.items():
                indices = [positions[c] for c in columns]
                model_path = folder / (name + ".pkl")
                error_path = folder / (name + ".csv")

                def fit_action(
                    columns: list[int] = indices,
                    path: Path = model_path,
                    training: tuple[Any, ...] = training,
                ) -> None:
                    fitted = train_pair(training[0][:, columns], training[1])
                    atomic_bytes(path, pickle.dumps(fitted, protocol=5))

                stage(
                    root,
                    f"nonlinear-{fold['name']}-{name}-fit",
                    source,
                    [model_path],
                    fit_action,
                    run,
                )

                def evaluate_action(
                    columns: list[int] = indices,
                    path: Path = model_path,
                    output: Path = error_path,
                    model_name: str = name,
                    evaluation: tuple[Any, ...] = evaluation,
                ) -> None:
                    # Only our own SHA-verified checkpoint is deserialized.
                    fitted = pickle.loads(path.read_bytes())
                    correction = np.column_stack(
                        [m.predict(evaluation[0][:, columns]) for m in fitted]
                    )
                    errors = evaluation[5].copy()
                    errors[["dx", "dy"]] = (
                        evaluation[2] + evaluation[4] * correction - evaluation[3]
                    )
                    errors["model"] = model_name
                    atomic_bytes(output, errors.to_csv(index=False).encode())

                stage(
                    root,
                    f"nonlinear-{fold['name']}-{name}-evaluate",
                    source,
                    [error_path],
                    evaluate_action,
                    run,
                )
                results.append(pd.read_csv(error_path))
            for _, _, arrays, state, _ in weeks(root, caches, set(fold["evaluation_games"])):
                reference = state[KEYS + ["player_role", "num_frames_output"]].copy()
                reference[["dx", "dy"]] = arrays["velocity"] - arrays["truth"]
                reference["model"] = "constant_velocity"
                results.append(reference)
            errors = pd.concat(results, ignore_index=True)
            summary = probe_summary(errors, variants)
            summary.update(fold=fold, source_signature=source, settings=SETTINGS)
            atomic_json(folder / "summary.json", summary)
            if fold["name"] != "development":
                inner.append(summary)
                if len(inner) == 3:
                    scores = {}
                    for variant in variants:
                        if variant == "core_plus_noise":
                            continue
                        values = [
                            (
                                next(
                                    r["coordinate_rmse_yards"]
                                    for r in s["models"]
                                    if r["model"] == variant
                                ),
                                s["validation_rows_per_model"],
                            )
                            for s in inner
                        ]
                        scores[variant] = float(
                            np.sqrt(sum(r * r * n for r, n in values) / sum(n for _, n in values))
                        )
                    atomic_json(
                        destination / "selection.json",
                        {
                            "inner_scores": scores,
                            "selected_model": min(scores, key=lambda n: (scores[n], n)),
                            "outer_validation_used_for_selection": False,
                            "source_signature": source,
                        },
                    )
            else:
                selection = json.loads((destination / "selection.json").read_text())
                summary.update(
                    inner_folds=inner,
                    selection=selection,
                    holdout_evaluation="not_run",
                    feature_gate="open",
                    purpose="Fixed-capacity representation diagnostic.",
                )
                atomic_json(destination / "summary.json", summary)
        run.event("nonlinear_probe_completed", holdout_evaluation="not_run")


def self_test() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1000, 4))
    y = np.column_stack([x[:, 0] * x[:, 1], np.sin(x[:, 2])])
    first = train_pair(x[:750], y[:750])
    second = train_pair(x[:750], y[:750])
    a = np.column_stack([m.predict(x[750:]) for m in first])
    b = np.column_stack([m.predict(x[750:]) for m in second])
    np.testing.assert_array_equal(a, b)
    if not np.mean((a - y[750:]) ** 2) < np.mean(y[750:] ** 2):
        raise ValueError("Nonlinear probe failed its independent synthetic signal check.")
    if any(m.n_iter_ != SETTINGS["max_iter"] for m in first):
        raise ValueError("Random-row early stopping must remain disabled.")
    observed = pd.DataFrame(
        {
            "game_id": [2024010100] * len(a),
            "play_id": 1,
            "nfl_id": 1,
            "frame_id": np.arange(1, len(a) + 1),
            "num_frames_output": len(a),
            "player_role": "Defensive Coverage",
            "dx": (a - y[750:])[:, 0],
            "dy": (a - y[750:])[:, 1],
            "model": "landing_features",
        }
    )
    reference = observed.assign(model="constant_velocity", dx=-y[750:, 0], dy=-y[750:, 1])
    summary = probe_summary(
        pd.concat([observed, reference], ignore_index=True), {"landing_features": list("abcd")}
    )
    if summary["status"] != "passed" or summary["selected_model"] != "landing_features":
        raise ValueError("Nonlinear probe metric integration failed.")
    print(json.dumps({"self_test": "passed", "fixed_iterations": SETTINGS["max_iter"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    if args.self_test:
        self_test()
    else:
        from nfl_trajectory.runtime import Run

        with Run(root, "nonlinear-probe") as run:
            nonlinear_probe(root, run)
