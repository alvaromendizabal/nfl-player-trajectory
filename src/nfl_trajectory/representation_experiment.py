"""Fixed-capacity ablations of pooled geometry, destinations and route embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.benchmark import bootstrap_scores, summarize
from nfl_trajectory.context_experiment import parent_prediction
from nfl_trajectory.feature_experiment import load_week, target_state
from nfl_trajectory.feature_research import (
    BATCH,
    add_moments,
    associations,
    empty_moments,
    predict_linear,
    ranked_names,
    solve_ridge,
    verify_inputs,
    weeks,
)
from nfl_trajectory.motion import ENTITY, KEYS
from nfl_trajectory.representation_features import (
    RepresentationBank,
    build_representation,
    fit_routes,
    representation_catalog,
    route_inputs,
)
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

GROUPS = ("graph_pool", "role_destination", "route_representation")


def representation_signature(inputs: dict[str, str], parameters: Any) -> str:
    sources = {
        name: sha256(Path(__file__).with_name(name + ".py"))
        for name in (
            "representation_features",
            "representation_experiment",
            "context_experiment",
            "feature_research",
            "feature_candidates",
            "features",
        )
    }
    return hashlib.sha256(
        json.dumps(
            {"sources": sources, "inputs": inputs, "parameters": parameters}, sort_keys=True
        ).encode()
    ).hexdigest()


def prepare_fold(root: Path, caches: list[Path], fold: dict[str, Any], destination: Path) -> None:
    frames, arrays = [], []
    for cache in caches:
        bank, _, _ = load_week(cache)
        selected = bank.state.game_id.isin(fold["training_games"]).to_numpy()
        if selected.any():
            frames.append(bank.state.loc[selected, ENTITY])
            arrays.append(route_inputs(bank)[selected])
    entities = pd.concat(frames, ignore_index=True)
    values = np.concatenate(arrays)
    order = entities.sort_values(ENTITY).index.to_numpy()
    fitted = fit_routes(values[order])
    fitted["training_games"] = fold["training_games"]
    atomic_json(destination / "routes.json", fitted)
    for cache in caches:
        bank, _, _ = load_week(cache)
        representation = build_representation(bank, fitted)
        from io import BytesIO

        buffer = BytesIO()
        np.savez_compressed(buffer, values=representation.values)
        atomic_bytes(destination / (cache.parent.name + ".npz"), buffer.getvalue())


def feature_rows(root: Path, caches: list[Path], games: set[int], destination: Path) -> Any:
    for cache in caches:
        for bank, targets, arrays, state, basis in weeks(root, [cache], games):
            with np.load(destination / (cache.parent.name + ".npz"), allow_pickle=False) as saved:
                rep = RepresentationBank(bank.state, saved["values"])
            yield bank, targets, arrays, state, basis, rep


def candidates(catalog: pd.DataFrame) -> dict[str, tuple[list[str], int]]:
    chosen = {family: ranked_names(catalog, catalog.family.eq(family), 64) for family in GROUPS}
    balanced = [columns[i] for i in range(24) for columns in chosen.values() if len(columns) > i]
    result = {f"plus_{family}": (columns, 32) for family, columns in chosen.items()}
    result["plus_representation"] = (balanced, 64)
    for family in GROUPS:
        result[f"representation_without_{family}"] = (
            [c for c in balanced if c not in chosen[family]],
            64,
        )
    return result


def run_fold(root: Path, caches: list[Path], fold: dict[str, Any], source: str, run: Run) -> None:
    destination = root / "artifacts/representation" / fold["name"]
    parent_folder = root / "artifacts/research" / fold["name"]
    parent = json.loads((parent_folder / "models.json").read_text())
    history = pd.read_csv(parent_folder / "history.csv")
    if parent["fold"] != fold:
        raise ValueError("Representation and parent fold differ.")
    training, evaluation = set(fold["training_games"]), set(fold["evaluation_games"])
    stage(
        root,
        f"representation-{fold['name']}-prepare",
        source,
        [destination / "routes.json"] + [destination / (c.parent.name + ".npz") for c in caches],
        lambda: prepare_fold(root, caches, fold, destination),
        run,
    )

    def screen_action() -> None:
        catalog = representation_catalog()
        moments = empty_moments(len(catalog))
        for bank, targets, arrays, state, basis, rep in feature_rows(
            root, caches, training, destination
        ):
            residual = (
                arrays["truth"] - parent_prediction(bank, targets, state, basis, parent, history)
            ) * target_state(bank, targets).sign.to_numpy()[:, None]
            for start in range(0, len(targets), BATCH):
                end = start + BATCH
                add_moments(
                    moments, rep.matrix(targets.iloc[start:end]).astype(float), residual[start:end]
                )
            run.event("representation_screen_progress", training_rows=moments["n"])
        screened = associations(moments, catalog)
        atomic_bytes(destination / "screening.csv", screened.to_csv(index=False).encode())

    stage(
        root,
        f"representation-{fold['name']}-screen",
        source,
        [destination / "screening.csv"],
        screen_action,
        run,
    )
    variants = candidates(pd.read_csv(destination / "screening.csv"))

    def fit_action() -> None:
        names = sorted({c for columns, _ in variants.values() for c in columns})
        positions = {name: i for i, name in enumerate(names)}
        gram, rhs = np.zeros((len(names), len(names))), np.zeros((len(names), 2))
        sx, sy, count = np.zeros(len(names)), np.zeros(2), 0
        for bank, targets, arrays, state, basis, rep in feature_rows(
            root, caches, training, destination
        ):
            residual = (
                arrays["truth"] - parent_prediction(bank, targets, state, basis, parent, history)
            ) * target_state(bank, targets).sign.to_numpy()[:, None]
            for start in range(0, len(targets), BATCH):
                end = start + BATCH
                x = rep.matrix(targets.iloc[start:end], names).astype(float)
                gram += x.T @ x
                rhs += x.T @ residual[start:end]
                sx += x.sum(0)
                sy += residual[start:end].sum(0)
                count += len(x)
            run.event("representation_fit_progress", training_rows=count)
        fitted: dict[str, Any] = {}
        for name, (columns, limit) in variants.items():
            if name.startswith("representation_without_"):
                columns = [c for c in columns if c in fitted["plus_representation"]["features"]]
            selected = [positions[c] for c in columns]
            fitted[name] = solve_ridge(
                gram[np.ix_(selected, selected)],
                rhs[selected],
                sx[selected],
                sy,
                count,
                columns,
                limit,
            )
        atomic_json(
            destination / "models.json",
            {
                "fold": fold,
                "additions": fitted,
                "parent_sha256": sha256(parent_folder / "models.json"),
                "routes_sha256": sha256(destination / "routes.json"),
            },
        )

    stage(
        root,
        f"representation-{fold['name']}-fit",
        source,
        [destination / "models.json"],
        fit_action,
        run,
    )

    def evaluate_action() -> None:
        additions = json.loads((destination / "models.json").read_text())["additions"]
        frames = []
        names = sorted({c for model in additions.values() for c in model["features"]})
        positions = {name: i for i, name in enumerate(names)}
        for bank, targets, arrays, state, basis, rep in feature_rows(
            root, caches, evaluation, destination
        ):
            core = parent_prediction(bank, targets, state, basis, parent, history)
            sign = target_state(bank, targets).sign.to_numpy()[:, None]
            predictions = {name: core.copy() for name in additions}
            for start in range(0, len(targets), BATCH):
                end = start + BATCH
                x = rep.matrix(targets.iloc[start:end], names).astype(float)
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
        errors = pd.concat(frames, ignore_index=True)
        summary = summarize(errors)
        reference = bootstrap_scores(errors[errors.model.eq("core_balanced")])
        for row in summary["models"]:
            row["delta_vs_core_ci95"] = np.quantile(
                bootstrap_scores(errors[errors.model.eq(row["model"])]) - reference, [0.025, 0.975]
            ).tolist()
            row["retained_additions"] = len(additions.get(row["model"], {}).get("features", []))
        summary.update(
            fold=fold, candidate_features=len(representation_catalog()), source_signature=source
        )
        atomic_bytes(destination / "errors.csv", errors.to_csv(index=False).encode())
        atomic_json(destination / "summary.json", summary)

    stage(
        root,
        f"representation-{fold['name']}-evaluate",
        source,
        [destination / "errors.csv", destination / "summary.json"],
        evaluate_action,
        run,
    )


def representation_research(root: Path, run: Run) -> None:
    from nfl_trajectory.research import research_stage_evidence

    destination = root / "artifacts/representation"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        caches, inputs = verify_inputs(root)
        parent = research_stage_evidence(root, "research")
        folds = [r["fold"] for r in parent["inner_folds"]] + [parent["fold"]]
        inputs.update(
            {
                str(p.relative_to(root)): sha256(p)
                for f in folds
                for name in ("models.json", "history.csv")
                for p in [root / "artifacts/research" / f["name"] / name]
            }
        )
        parameters = {
            "parent_source": parent["source_signature"],
            "parent_variant": "plus_balanced",
            "folds": folds,
        }
        source = representation_signature(inputs, parameters)
        atomic_json(
            destination / "plan.json",
            {
                "inputs": inputs,
                "parameters": parameters,
                "source_signature": source,
                "candidate_features": len(representation_catalog()),
                "holdout_evaluation": "not_run",
            },
        )
        inner = []
        for fold in folds[:-1]:
            run_fold(root, caches, fold, source, run)
            inner.append(json.loads((destination / fold["name"] / "summary.json").read_text()))
        totals: dict[str, list[float]] = {}
        for result in inner:
            for row in result["models"]:
                if row["model"].startswith("plus_") or row["model"] == "core_balanced":
                    values = totals.setdefault(row["model"], [0.0, 0.0])
                    n = result["validation_rows_per_model"]
                    values[0] += row["coordinate_rmse_yards"] ** 2 * n
                    values[1] += n
        pooled = {name: float(np.sqrt(sse / count)) for name, (sse, count) in totals.items()}
        selected = min(pooled, key=lambda n: (pooled[n], n))
        atomic_json(
            destination / "selection.json",
            {
                "selected_model": selected,
                "inner_scores": pooled,
                "outer_validation_used_for_selection": False,
                "source_signature": source,
            },
        )
        run_fold(root, caches, folds[-1], source, run)
        summary = json.loads((destination / "development/summary.json").read_text())
        summary.update(
            inner_folds=inner,
            inner_scores=pooled,
            selected_by_inner_folds=selected,
            source_signature=source,
            holdout_evaluation="not_run",
            feature_gate="open",
        )
        atomic_json(destination / "summary.json", summary)
        run.event(
            "representation_research_completed",
            selected_model=selected,
            candidate_features=len(representation_catalog()),
            holdout_evaluation="not_run",
        )
