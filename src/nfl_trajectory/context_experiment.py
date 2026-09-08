"""Controlled raw-context additions to the fixed, balanced feature representation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.benchmark import bootstrap_scores, summarize
from nfl_trajectory.context_features import (
    ContextBank,
    build_context_features,
    context_catalog,
    load_context,
    save_context,
)
from nfl_trajectory.feature_candidates import candidate_matrix
from nfl_trajectory.feature_experiment import load_week, target_state
from nfl_trajectory.feature_research import (
    BATCH,
    add_moments,
    associations,
    baseline_prediction,
    empty_moments,
    history_rows,
    predict_linear,
    ranked_names,
    solve_ridge,
    verify_inputs,
    weeks,
)
from nfl_trajectory.features import PlayerFeatures
from nfl_trajectory.motion import KEYS
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

CONTEXT_GROUPS = ("metadata", "robust_history", "matched_history", "peer_context", "reachability")


def context_signature(files: dict[str, str], parameters: Any) -> str:
    sources = {
        name: sha256(Path(__file__).with_name(name + ".py"))
        for name in (
            "context_features",
            "context_experiment",
            "feature_research",
            "feature_candidates",
            "features",
        )
    }
    return hashlib.sha256(
        json.dumps(
            {"sources": sources, "inputs": files, "parameters": parameters}, sort_keys=True
        ).encode()
    ).hexdigest()


def prepare_context(root: Path, run: Run) -> list[Path]:
    caches, _ = verify_inputs(root)
    manifest_path = root / "artifacts/research/input_manifest.json"
    expected = {}
    if manifest_path.exists():
        expected = {e["path"]: e["sha256"] for e in json.loads(manifest_path.read_text())["files"]}
    for i, cache in enumerate(caches, 1):
        raw = root / "data/raw/train" / (cache.parent.name + ".csv")
        if not raw.exists():
            raise ValueError("Restore the verified observed input weeks before context research.")
        digest = sha256(raw)
        if expected and expected.get(str(raw.relative_to(root))) != digest:
            raise ValueError("Raw context input disagrees with the verified source snapshot.")
        output = root / "artifacts/context/weeks" / cache.parent.name / "context.npz"
        lock = sha256(root / "uv.lock") if (root / "uv.lock").exists() else "test"
        signature = context_signature(
            {"raw": digest, "bank": sha256(cache), "lock": lock}, {"phase": "prepare"}
        )

        def prepare(
            source: Path = raw, numerical: Path = cache, destination: Path = output
        ) -> None:
            bank, _, _ = load_week(numerical)
            observed = pd.read_csv(source)
            observed = observed[observed.game_id.isin(bank.state.game_id.unique())]
            save_context(destination, build_context_features(observed, bank.state))

        stage(root, f"context-prepare-{cache.parent.name}", signature, [output], prepare, run)
        run.event("context_preparation_progress", completed_weeks=i, total_weeks=len(caches))
    return caches


def context_weeks(
    root: Path, caches: list[Path], games: set[int]
) -> Iterator[
    tuple[
        PlayerFeatures, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, np.ndarray, ContextBank
    ]
]:
    for cache in caches:
        for bank, targets, arrays, state, basis in weeks(root, [cache], games):
            context = load_context(
                root / "artifacts/context/weeks" / cache.parent.name / "context.npz"
            )
            yield bank, targets, arrays, state, basis, context


def parent_prediction(
    bank: PlayerFeatures,
    targets: pd.DataFrame,
    state: pd.DataFrame,
    basis: np.ndarray,
    models: dict[str, Any],
    history: pd.DataFrame,
) -> np.ndarray:
    predicted = baseline_prediction(state, basis, models["baseline"])
    sign = target_state(bank, targets).sign.to_numpy()[:, None]
    h = history_rows(history, targets)
    landing, balanced = models["landing"], models["additions"]["plus_balanced"]
    for start in range(0, len(targets), BATCH):
        end = start + BATCH
        correction = predict_linear(
            bank.matrix(targets.iloc[start:end], landing["features"]).astype(float), landing
        )
        if balanced["features"]:
            x = candidate_matrix(
                bank, targets.iloc[start:end], h[start:end], balanced["features"]
            ).astype(float)
            correction += predict_linear(x, balanced)
        predicted[start:end] += sign[start:end] * correction
    return predicted


def context_screen(
    root: Path,
    caches: list[Path],
    training: set[int],
    models: dict[str, Any],
    history: pd.DataFrame,
    run: Run,
) -> pd.DataFrame:
    catalog = context_catalog()
    totals = empty_moments(len(catalog))
    for bank, targets, arrays, state, basis, context in context_weeks(root, caches, training):
        predicted = parent_prediction(bank, targets, state, basis, models, history)
        y = (arrays["truth"] - predicted) * target_state(bank, targets).sign.to_numpy()[:, None]
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            x = context.matrix(targets.iloc[start:end]).astype(float)
            add_moments(totals, x, y[start:end])
        run.event("context_screen_progress", training_rows=totals["n"])
    return associations(totals, catalog)


def context_candidates(catalog: pd.DataFrame) -> dict[str, tuple[list[str], int]]:
    chosen = {
        group: ranked_names(catalog, catalog.family.eq(group), 64) for group in CONTEXT_GROUPS
    }
    balanced = [columns[i] for i in range(12) for columns in chosen.values() if len(columns) > i]
    candidates = {f"plus_{group}": (columns, 32) for group, columns in chosen.items()}
    candidates["plus_context"] = (balanced, 64)
    for group in CONTEXT_GROUPS:
        candidates[f"context_without_{group}"] = (
            [c for c in balanced if c not in chosen[group]],
            64,
        )
    return candidates


def context_fit(
    root: Path,
    caches: list[Path],
    training: set[int],
    models: dict[str, Any],
    history: pd.DataFrame,
    candidates: dict[str, tuple[list[str], int]],
    run: Run,
) -> dict[str, Any]:
    names = sorted({c for columns, _ in candidates.values() for c in columns})
    positions = {name: i for i, name in enumerate(names)}
    gram, rhs = np.zeros((len(names), len(names))), np.zeros((len(names), 2))
    sx, sy, count = np.zeros(len(names)), np.zeros(2), 0
    for bank, targets, arrays, state, basis, context in context_weeks(root, caches, training):
        y = (
            arrays["truth"] - parent_prediction(bank, targets, state, basis, models, history)
        ) * target_state(bank, targets).sign.to_numpy()[:, None]
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            x = context.matrix(targets.iloc[start:end], names).astype(float)
            gram += x.T @ x
            rhs += x.T @ y[start:end]
            sx += x.sum(0)
            sy += y[start:end].sum(0)
            count += len(x)
        run.event("context_fit_progress", training_rows=count, union_features=len(names))
    fitted: dict[str, Any] = {}
    for name, (columns, limit) in candidates.items():
        if name.startswith("context_without_"):
            columns = [c for c in columns if c in fitted["plus_context"]["features"]]
        selected = [positions[c] for c in columns]
        fitted[name] = solve_ridge(
            gram[np.ix_(selected, selected)], rhs[selected], sx[selected], sy, count, columns, limit
        )
    return fitted


def context_evaluate(
    root: Path,
    caches: list[Path],
    evaluation: set[int],
    parent: dict[str, Any],
    history: pd.DataFrame,
    additions: dict[str, Any],
) -> pd.DataFrame:
    frames = []
    names = sorted({c for model in additions.values() for c in model["features"]})
    positions = {name: i for i, name in enumerate(names)}
    for bank, targets, arrays, state, basis, context in context_weeks(root, caches, evaluation):
        core = parent_prediction(bank, targets, state, basis, parent, history)
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        predictions = {name: core.copy() for name in additions}
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            x = context.matrix(targets.iloc[start:end], names).astype(float)
            for name, model in additions.items():
                if model["features"]:
                    selected = [positions[c] for c in model["features"]]
                    predictions[name][start:end] += sign[start:end] * predict_linear(
                        x[:, selected], model
                    )
        predictions.update(core_balanced=core, constant_velocity=arrays["velocity"])
        for name, prediction in predictions.items():
            errors = state[KEYS + ["player_role", "num_frames_output"]].copy()
            errors[["dx", "dy"]] = prediction - arrays["truth"]
            errors["model"] = name
            frames.append(errors)
    return pd.concat(frames, ignore_index=True)


def context_fold(
    root: Path, caches: list[Path], fold: dict[str, Any], source: str, run: Run
) -> None:
    destination = root / "artifacts/context" / fold["name"]
    parent_folder = root / "artifacts/research" / fold["name"]
    models = json.loads((parent_folder / "models.json").read_text())
    if models["fold"] != fold:
        raise ValueError("Context and parent folds disagree.")
    history = pd.read_csv(parent_folder / "history.csv")
    training, evaluation = set(fold["training_games"]), set(fold["evaluation_games"])

    def screen_action() -> None:
        catalog = context_screen(root, caches, training, models, history, run)
        atomic_bytes(destination / "screening.csv", catalog.to_csv(index=False).encode())

    stage(
        root,
        f"context-{fold['name']}-screen",
        source,
        [destination / "screening.csv"],
        screen_action,
        run,
    )
    catalog = pd.read_csv(destination / "screening.csv")

    def fit_action() -> None:
        fitted = context_fit(
            root, caches, training, models, history, context_candidates(catalog), run
        )
        atomic_json(
            destination / "models.json",
            {
                "additions": fitted,
                "parent_sha256": sha256(parent_folder / "models.json"),
                "fold": fold,
            },
        )

    stage(
        root, f"context-{fold['name']}-fit", source, [destination / "models.json"], fit_action, run
    )
    additions = json.loads((destination / "models.json").read_text())["additions"]

    def evaluate_action() -> None:
        errors = context_evaluate(root, caches, evaluation, models, history, additions)
        summary = summarize(errors)
        reference = bootstrap_scores(errors[errors.model.eq("core_balanced")])
        for row in summary["models"]:
            row["delta_vs_core_ci95"] = np.quantile(
                bootstrap_scores(errors[errors.model.eq(row["model"])]) - reference, [0.025, 0.975]
            ).tolist()
            row["retained_additions"] = len(additions.get(row["model"], {}).get("features", []))
        summary.update(fold=fold, candidate_features=len(catalog), source_signature=source)
        atomic_bytes(destination / "errors.csv", errors.to_csv(index=False).encode())
        atomic_json(destination / "summary.json", summary)

    stage(
        root,
        f"context-{fold['name']}-evaluate",
        source,
        [destination / "errors.csv", destination / "summary.json"],
        evaluate_action,
        run,
    )


def context_research(root: Path, run: Run, checkpoint: Callable[[], object] | None = None) -> None:
    destination = root / "artifacts/context"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        caches = prepare_context(root, run)
        parent = json.loads((root / "artifacts/research/summary.json").read_text())
        parent_plan = json.loads((root / "artifacts/research/plan.json").read_text())
        if parent["source_signature"] != parent_plan["source_signature"]:
            raise ValueError("Parent research is incomplete or stale.")
        folds = [r["fold"] for r in parent["inner_folds"]] + [parent["fold"]]
        input_paths = [
            root / "artifacts/context/weeks" / c.parent.name / "context.npz" for c in caches
        ]
        input_paths += [
            root / "artifacts/research" / f["name"] / name
            for f in folds
            for name in ("models.json", "history.csv")
        ]
        input_hashes = {str(p.relative_to(root)): sha256(p) for p in input_paths}
        source = context_signature(
            input_hashes,
            {
                "parent_source": parent_plan["source_signature"],
                "parent_variant": "plus_balanced",
                "folds": folds,
            },
        )
        atomic_json(
            destination / "plan.json",
            {
                "source_signature": source,
                "inputs": input_hashes,
                "parent_variant": "plus_balanced",
                "parent_source_signature": parent_plan["source_signature"],
                "folds": folds,
                "candidate_features": len(context_catalog()),
                "holdout_evaluation": "not_run",
            },
        )
        inner = []
        for fold in folds[:-1]:
            context_fold(root, caches, fold, source, run)
            inner.append(json.loads((destination / fold["name"] / "summary.json").read_text()))
            if checkpoint:
                checkpoint()
        scores: dict[str, list[float]] = {}
        for result in inner:
            for row in result["models"]:
                if row["model"].startswith("plus_") or row["model"] == "core_balanced":
                    values = scores.setdefault(row["model"], [0.0, 0])
                    n = result["validation_rows_per_model"]
                    values[0] += row["coordinate_rmse_yards"] ** 2 * n
                    values[1] += n
        pooled = {name: float(np.sqrt(sse / count)) for name, (sse, count) in scores.items()}
        selected = min(pooled, key=lambda name: (pooled[name], name))
        atomic_json(
            destination / "selection.json",
            {
                "selected_model": selected,
                "inner_scores": pooled,
                "outer_validation_used_for_selection": False,
                "source_signature": source,
            },
        )
        context_fold(root, caches, folds[-1], source, run)
        summary = json.loads((destination / "development/summary.json").read_text())
        summary.update(
            inner_folds=inner,
            inner_scores=pooled,
            selected_by_inner_folds=selected,
            source_signature=source,
            parent_variant="plus_balanced",
            holdout_evaluation="not_run",
        )
        atomic_json(destination / "summary.json", summary)
        if checkpoint:
            checkpoint()
        run.event(
            "context_research_completed",
            selected_model=selected,
            inner_rmse=pooled[selected],
            candidate_features=len(context_catalog()),
            holdout_evaluation="not_run",
        )
