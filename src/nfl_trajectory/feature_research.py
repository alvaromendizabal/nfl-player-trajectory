"""Chronological feature research with fold-local baselines and protected additions.

This experiment consumes verified completed numerical caches. It never overwrites
the original feature/model artifacts or evaluates the reserved holdout. Fixed
ridge regularization is held constant so this is a representation experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.benchmark import bootstrap_scores, load_design, protocol, summarize
from nfl_trajectory.feature_candidates import (
    HISTORY_NAMES,
    candidate_matrix,
    historical_encodings,
    research_catalog,
)
from nfl_trajectory.feature_experiment import load_week, numerical_sources, target_state
from nfl_trajectory.features import PlayerFeatures, feature_catalog
from nfl_trajectory.models import fit_statistics, predict_from_design, sufficient_statistics
from nfl_trajectory.motion import ENTITY, KEYS
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

BATCH = 1024
ALPHA = 0.01
CORRELATION_LIMIT = 0.9995
GROUPS = (
    "interactions",
    "multiscale",
    "nonlinear",
    "role_response",
    "projected_geometry",
    "player_history",
    "remaining_history",
)


def chronological_folds(splits: pd.DataFrame) -> list[dict[str, Any]]:
    train = splits[splits.split.eq("train")]
    dates = sorted(train.game_date.unique())
    if len(dates) < 6:
        raise ValueError("At least six training dates are required for chronological research.")
    boundaries = [len(dates) // 2, 2 * len(dates) // 3, 5 * len(dates) // 6, len(dates)]
    folds = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        folds.append(
            {
                "name": f"inner_{i}",
                "training_games": sorted(
                    train.loc[train.game_date.isin(dates[:start]), "game_id"].astype(int)
                ),
                "evaluation_games": sorted(
                    train.loc[train.game_date.isin(dates[start:end]), "game_id"].astype(int)
                ),
            }
        )
    for fold in folds:
        if (
            not fold["evaluation_games"]
            or max(fold["training_games"]) // 100 >= min(fold["evaluation_games"]) // 100
        ):
            raise ValueError("Chronological folds must have strictly ordered, disjoint dates.")
    return folds


def verify_inputs(root: Path) -> tuple[list[Path], dict[str, str]]:
    """Require externally verified restore hashes or original successful stage receipts."""
    model_path = root / "artifacts/features/model.json"
    model = json.loads(model_path.read_text())
    if model.get("source_sha256") != numerical_sources():
        raise ValueError("Completed feature caches do not match their numerical source.")
    baseline = root / "artifacts/benchmark/model.json"
    if model.get("baseline_sha256") != sha256(baseline):
        raise ValueError("Completed baseline differs from feature-cache provenance.")
    splits, audit = protocol(root)
    if model.get("training_games") != sorted(
        splits.loc[splits.split.eq("train"), "game_id"].astype(int)
    ):
        raise ValueError("Completed feature training games do not match the frozen split.")
    if model.get("split_sha256") != sha256(root / "artifacts/game_splits.csv"):
        raise ValueError("Completed feature split fingerprint differs.")
    manifest_path = root / "artifacts/research/input_manifest.json"
    expected = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected.update({item["path"]: item["sha256"] for item in manifest["files"]})
    else:
        for receipt in sorted((root / ".state").glob("*.json")):
            value = json.loads(receipt.read_text())
            if value.get("status") == "completed":
                expected.update(value.get("outputs", {}))
    paths = []
    development = set(splits.loc[~splits.split.eq("holdout"), "game_id"])
    for pair in audit["pairs"]:
        if not set(pair["games"]) & development:
            continue
        week = Path(pair["file"]).stem
        for kind, filename in (("features", "features.npz"), ("benchmark", "design.npz")):
            path = root / "artifacts" / kind / "weeks" / week / filename
            relative = str(path.relative_to(root))
            if not path.exists() or expected.get(relative) != sha256(path):
                raise ValueError(f"Missing or unverified completed research input: {relative}")
            paths.append(path)
    inputs = paths + [
        model_path,
        baseline,
        root / "artifacts/game_splits.csv",
        root / "artifacts/audit_summary.json",
    ]
    return [p for p in paths if p.name == "features.npz"], {
        str(p.relative_to(root)): sha256(p) for p in inputs
    }


def weeks(
    root: Path, caches: list[Path], games: set[int]
) -> Iterator[tuple[PlayerFeatures, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, np.ndarray]]:
    for cache in caches:
        bank, targets, arrays = load_week(cache)
        selected = targets.game_id.isin(games).to_numpy()
        if not selected.any():
            continue
        design_path = root / "artifacts/benchmark/weeks" / cache.parent.name / "design.npz"
        state, basis, truth, labels = load_design(design_path)
        if not state[KEYS].equals(targets[KEYS]) or not np.array_equal(labels, arrays["labels"]):
            raise ValueError("Baseline and feature caches have different row keys or partitions.")
        if not np.array_equal(truth[["x", "y"]].to_numpy(), arrays["truth"]):
            raise ValueError("Baseline and feature caches have different targets.")
        if not set(labels).issubset({"train", "validation"}):
            raise ValueError("Holdout data must not enter research caches.")
        yield (
            bank,
            targets.loc[selected].reset_index(drop=True),
            {k: v[selected] for k, v in arrays.items()},
            state.loc[selected].reset_index(drop=True),
            basis[selected],
        )


def fit_baseline(
    root: Path, caches: list[Path], training: set[int], no_landing: bool = False
) -> dict[str, Any]:
    statistics = []
    for _, targets, arrays, state, basis in weeks(root, caches, training):
        if no_landing:
            basis = basis.copy()
            basis[:, :, 3:] = 0.0
        truth = targets[KEYS].copy()
        truth[["x", "y"]] = arrays["truth"]
        statistics.append(sufficient_statistics(state, basis, truth))
    fitted = fit_statistics(statistics)
    fitted["training_games"] = sorted(training)
    fitted["no_landing"] = no_landing
    return fitted


def baseline_prediction(
    state: pd.DataFrame, basis: np.ndarray, fitted: dict[str, Any]
) -> np.ndarray:
    if fitted["no_landing"]:
        basis = basis.copy()
        basis[:, :, 3:] = 0.0
    return predict_from_design(state, basis, "role_ridge", fitted)[["x", "y"]].to_numpy()


def fold_history(
    root: Path, caches: list[Path], training: set[int], evaluation: set[int]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for bank, targets, arrays, state, _ in weeks(root, caches, training | evaluation):
        frame = targets[KEYS].copy()
        frame["player_role"] = state.player_role.to_numpy()
        is_train = frame.game_id.isin(training).to_numpy()
        errors = np.zeros((len(frame), 2))
        errors[is_train] = (
            (arrays["truth"][is_train] - arrays["velocity"][is_train])
            * target_state(bank, targets).sign.to_numpy()[is_train, None]
            / (targets.frame_id.to_numpy()[is_train, None] / 10)
        )
        frame[["error_x", "error_y"]] = errors
        frames.append(
            frame.groupby(ENTITY, sort=True)
            .agg(
                player_role=("player_role", "first"),
                error_x=("error_x", "mean"),
                error_y=("error_y", "mean"),
            )
            .reset_index()
        )
    observations = pd.concat(frames, ignore_index=True).sort_values(ENTITY).reset_index(drop=True)
    encoded, fitted = historical_encodings(observations, training)
    table = observations[ENTITY].copy()
    table[HISTORY_NAMES] = encoded
    return table, fitted


def history_rows(table: pd.DataFrame, targets: pd.DataFrame) -> np.ndarray:
    values = (
        targets[ENTITY]
        .merge(table, on=ENTITY, how="left", sort=False, validate="many_to_one")[HISTORY_NAMES]
        .to_numpy(float)
    )
    if not np.isfinite(values).all():
        raise ValueError("Historical feature rows are missing or nonfinite.")
    return values


def predict_linear(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(
        model["coefficients"]
    ) + np.asarray(model["intercept"])


def model_correction(x: np.ndarray, positions: dict[str, int], model: dict[str, Any]) -> np.ndarray:
    if not model["features"]:
        return np.zeros((len(x), 2))
    return predict_linear(x[:, [positions[c] for c in model["features"]]], model)


def research_signature(root: Path, inputs: dict[str, str], settings: Any) -> str:
    sources = numerical_sources()
    for name in ("feature_candidates", "feature_research"):
        sources[f"{name}.py"] = sha256(Path(__file__).with_name(f"{name}.py"))
    sources["uv.lock"] = sha256(root / "uv.lock") if (root / "uv.lock").exists() else "absent"
    payload = {"sources": sources, "inputs": inputs, "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def empty_moments(width: int) -> dict[str, Any]:
    return {
        "n": 0,
        "sx": np.zeros(width),
        "sxx": np.zeros(width),
        "sxy": np.zeros((width, 2)),
        "sy": np.zeros(2),
        "syy": np.zeros(2),
    }


def add_moments(total: dict[str, Any], x: np.ndarray, y: np.ndarray) -> None:
    total["n"] += len(x)
    total["sx"] += x.sum(0)
    total["sxx"] += np.einsum("nf,nf->f", x, x)
    total["sxy"] += x.T @ y
    total["sy"] += y.sum(0)
    total["syy"] += (y * y).sum(0)


def associations(total: dict[str, Any], catalog: pd.DataFrame) -> pd.DataFrame:
    n = total["n"]
    if n < 2:
        raise ValueError("At least two training rows are required.")
    variance = np.maximum(total["sxx"] - total["sx"] ** 2 / n, 0)
    target_variance = np.maximum(total["syy"] - total["sy"] ** 2 / n, 0)
    covariance = total["sxy"] - np.outer(total["sx"], total["sy"]) / n
    denominator = np.sqrt(variance[:, None] * target_variance[None, :])
    correlation = np.divide(
        covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12
    )
    result = catalog.copy()
    result["training_variance"] = variance / n
    result["training_association"] = np.max(np.abs(np.clip(correlation, -1, 1)), axis=1)
    result["screen_status"] = np.where(
        result.training_variance > 1e-10, "eligible", "constant_or_near_constant"
    )
    return result


def ranked_names(catalog: pd.DataFrame, mask: pd.Series, limit: int) -> list[str]:
    rows = catalog[mask & catalog.screen_status.eq("eligible")]
    return (
        rows.sort_values(["training_association", "feature"], ascending=[False, True])
        .head(limit)
        .feature.tolist()
    )


def screen(
    root: Path,
    caches: list[Path],
    training: set[int],
    baseline: dict[str, Any],
    history: pd.DataFrame,
    landing: dict[str, Any] | None,
    run: Run,
) -> pd.DataFrame:
    catalog = feature_catalog() if landing is None else research_catalog()
    totals = empty_moments(len(catalog))
    for bank, targets, arrays, state, basis in weeks(root, caches, training):
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        y = (arrays["truth"] - baseline_prediction(state, basis, baseline)) * sign
        h = history_rows(history, targets)
        if landing is not None:
            for start in range(0, len(targets), BATCH):
                end = start + BATCH
                x = bank.matrix(targets.iloc[start:end], landing["features"]).astype(float)
                y[start:end] -= predict_linear(x, landing)
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            x = (
                bank.matrix(targets.iloc[start:end])
                if landing is None
                else candidate_matrix(bank, targets.iloc[start:end], h[start:end])
            ).astype(float)
            add_moments(totals, x, y[start:end])
        run.event(
            "research_screen_progress",
            phase="conditional" if landing else "baseline",
            training_rows=totals["n"],
            training_games=len(training),
        )
    return associations(totals, catalog)


def solve_ridge(
    gram: np.ndarray,
    rhs: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    count: int,
    names: list[str],
    limit: int,
) -> dict[str, Any]:
    mean, intercept = sx / count, sy / count
    covariance = gram / count - np.outer(mean, mean)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    normalized = covariance / np.outer(scale, scale)
    retained: list[int] = []
    rejected: dict[str, str] = {}
    for i, name in enumerate(names):
        if covariance[i, i] <= 1e-10:
            rejected[name] = "constant_or_near_constant"
        elif retained and np.abs(normalized[i, retained]).max() >= CORRELATION_LIMIT:
            rejected[name] = "training_redundancy"
        elif len(retained) >= limit:
            rejected[name] = "feature_budget"
        else:
            retained.append(i)
    if not retained:
        return {
            "features": [],
            "mean": [],
            "scale": [],
            "coefficients": [],
            "intercept": [0.0, 0.0],
            "rejected": rejected,
            "alpha": ALPHA,
        }
    lhs = normalized[np.ix_(retained, retained)] + ALPHA * np.eye(len(retained))
    response = (rhs[retained] / count - np.outer(mean[retained], intercept)) / scale[retained, None]
    coefficients = np.linalg.solve(lhs, response)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite research coefficients.")
    return {
        "features": [names[i] for i in retained],
        "mean": mean[retained].tolist(),
        "scale": scale[retained].tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept.tolist(),
        "rejected": rejected,
        "alpha": ALPHA,
    }


def fit_additions(
    root: Path,
    caches: list[Path],
    training: set[int],
    baseline: dict[str, Any],
    history: pd.DataFrame,
    landing: dict[str, Any] | None,
    candidates: dict[str, tuple[list[str], int]],
    run: Run,
) -> dict[str, Any]:
    union = sorted({name for names, _ in candidates.values() for name in names})
    positions = {name: i for i, name in enumerate(union)}
    gram = np.zeros((len(union), len(union)))
    rhs, sx, sy = np.zeros((len(union), 2)), np.zeros(len(union)), np.zeros(2)
    count = 0
    for bank, targets, arrays, state, basis in weeks(root, caches, training):
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        y = (arrays["truth"] - baseline_prediction(state, basis, baseline)) * sign
        h = history_rows(history, targets)
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            response = y[start:end].copy()
            if landing is not None:
                response -= predict_linear(
                    bank.matrix(targets.iloc[start:end], landing["features"]).astype(float), landing
                )
            x = candidate_matrix(bank, targets.iloc[start:end], h[start:end], union).astype(float)
            gram += x.T @ x
            rhs += x.T @ response
            sx += x.sum(0)
            sy += response.sum(0)
            count += len(x)
        run.event("research_fit_progress", training_rows=count, union_features=len(union))
    fitted: dict[str, Any] = {}
    for name, (columns, limit) in candidates.items():
        if name.startswith("balanced_without_"):
            # A removal ablation cannot refill freed slots with new features.
            # Keep precisely the surviving columns of the already fitted union.
            protected_union = set(fitted["plus_balanced"]["features"])
            columns = [column for column in columns if column in protected_union]
        indices = [positions[c] for c in columns]
        fitted[name] = solve_ridge(
            gram[np.ix_(indices, indices)], rhs[indices], sx[indices], sy, count, columns, limit
        )
    return fitted


def correction_candidates(
    catalog: pd.DataFrame, landing: dict[str, Any]
) -> dict[str, tuple[list[str], int]]:
    protected = set(landing["features"])
    masks = {
        "interactions": catalog.signal.eq("interaction"),
        "remaining_history": catalog.family.isin(
            [
                "history_lags",
                "history_summaries",
                "forecast_interactions",
                "context",
                "forecast",
                "availability",
            ]
        )
        & ~catalog.signal.eq("interaction"),
        **{
            group: catalog.family.eq(group)
            for group in GROUPS
            if group not in ("interactions", "remaining_history")
        },
    }
    chosen = {
        group: ranked_names(catalog, mask & ~catalog.feature.isin(protected), 64)
        for group, mask in masks.items()
    }
    # Round-robin family quotas prevent one correlated family displacing all others.
    balanced = [columns[i] for i in range(12) for columns in chosen.values() if len(columns) > i]
    candidates = {f"plus_{group}": (columns, 32) for group, columns in chosen.items()}
    candidates["plus_balanced"] = (balanced, 64)
    for group in GROUPS:
        excluded = set(chosen[group])
        candidates[f"balanced_without_{group}"] = ([c for c in balanced if c not in excluded], 64)
    return candidates


def evaluate_fold(
    root: Path,
    caches: list[Path],
    evaluation: set[int],
    history: pd.DataFrame,
    baseline: dict[str, Any],
    motion_baseline: dict[str, Any],
    landing: dict[str, Any],
    motion: dict[str, Any],
    additions: dict[str, Any],
) -> pd.DataFrame:
    frames = []
    for bank, targets, arrays, state, basis in weeks(root, caches, evaluation):
        h = history_rows(history, targets)
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        ridge = baseline_prediction(state, basis, baseline)
        motion_prediction = baseline_prediction(state, basis, motion_baseline)
        union = sorted(
            {name for model in [landing, motion, *additions.values()] for name in model["features"]}
        )
        positions = {name: i for i, name in enumerate(union)}
        predicted = {
            name: np.empty_like(ridge) for name in ("landing_refit", "without_landing", *additions)
        }
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            x = candidate_matrix(bank, targets.iloc[start:end], h[start:end], union).astype(float)

            core = ridge[start:end] + sign[start:end] * model_correction(x, positions, landing)
            predicted["landing_refit"][start:end] = core
            predicted["without_landing"][start:end] = motion_prediction[start:end] + sign[
                start:end
            ] * model_correction(x, positions, motion)
            for name, model in additions.items():
                predicted[name][start:end] = core + sign[start:end] * model_correction(
                    x, positions, model
                )
        predicted.update({"constant_velocity": arrays["velocity"], "role_ridge": ridge})
        for name, prediction in predicted.items():
            errors = state[KEYS + ["player_role", "num_frames_output"]].copy()
            errors[["dx", "dy"]] = prediction - arrays["truth"]
            errors["model"] = name
            frames.append(errors)
    return pd.concat(frames, ignore_index=True)


def run_fold(
    root: Path, caches: list[Path], fold: dict[str, Any], destination: Path, run: Run
) -> None:
    training, evaluation = set(fold["training_games"]), set(fold["evaluation_games"])
    source = json.loads((root / "artifacts/research/plan.json").read_text())["source_signature"]

    def phase(name: str, files: list[str], action: Callable[[], None]) -> None:
        recipe = hashlib.sha256(
            (source + json.dumps(fold, sort_keys=True) + name).encode()
        ).hexdigest()
        stage(
            root,
            f"research-{fold['name']}-{name}",
            recipe,
            [destination / p for p in files],
            action,
            run,
        )

    def prepare() -> None:
        baseline = fit_baseline(root, caches, training)
        motion_baseline = fit_baseline(root, caches, training, no_landing=True)
        history, history_model = fold_history(root, caches, training, evaluation)
        atomic_json(
            destination / "baselines.json", {"landing": baseline, "motion": motion_baseline}
        )
        atomic_json(destination / "history.json", history_model)
        atomic_bytes(destination / "history.csv", history.to_csv(index=False).encode())

    phase("prepare", ["baselines.json", "history.json", "history.csv"], prepare)
    baselines = json.loads((destination / "baselines.json").read_text())
    baseline, motion_baseline = baselines["landing"], baselines["motion"]
    history = pd.read_csv(destination / "history.csv")
    history_model = json.loads((destination / "history.json").read_text())

    def fit_core(name: str, fitted: dict[str, Any], motion_only: bool) -> None:
        catalog = screen(root, caches, training, fitted, history, None, run)
        mask = catalog.signal.eq("motion") if motion_only else ~catalog.signal.eq("interaction")
        names = ranked_names(catalog, mask, 96)
        model = fit_additions(
            root, caches, training, fitted, history, None, {name: (names, 64)}, run
        )[name]
        atomic_json(destination / f"{name}.json", model)
        atomic_bytes(destination / f"{name}_screen.csv", catalog.to_csv(index=False).encode())

    phase(
        "landing",
        ["landing.json", "landing_screen.csv"],
        partial(fit_core, "landing", baseline, False),
    )
    phase(
        "motion",
        ["motion.json", "motion_screen.csv"],
        partial(fit_core, "motion", motion_baseline, True),
    )
    landing = json.loads((destination / "landing.json").read_text())
    motion = json.loads((destination / "motion.json").read_text())

    def conditional_screen() -> None:
        catalog = screen(root, caches, training, baseline, history, landing, run)
        atomic_bytes(destination / "conditional.csv", catalog.to_csv(index=False).encode())

    phase("conditional", ["conditional.csv"], conditional_screen)
    conditional = pd.read_csv(destination / "conditional.csv")
    candidates = correction_candidates(conditional, landing)

    def fit_corrections() -> None:
        additions = fit_additions(
            root, caches, training, baseline, history, landing, candidates, run
        )
        atomic_json(destination / "additions.json", additions)

    phase("additions", ["additions.json"], fit_corrections)
    additions = json.loads((destination / "additions.json").read_text())
    models = {
        "baseline": baseline,
        "motion_baseline": motion_baseline,
        "landing": landing,
        "motion": motion,
        "history": history_model,
        "additions": additions,
        "fold": fold,
    }
    atomic_json(destination / "models.json", models)
    errors = evaluate_fold(
        root, caches, evaluation, history, baseline, motion_baseline, landing, motion, additions
    )
    summary = summarize(errors)
    reference = bootstrap_scores(errors[errors.model.eq("landing_refit")])
    for row in summary["models"]:
        delta = bootstrap_scores(errors[errors.model.eq(row["model"])]) - reference
        row["delta_vs_landing_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
        row["retained_additions"] = len(additions.get(row["model"], {}).get("features", []))
    summary["fold"] = fold
    summary["candidate_features"] = len(conditional)
    selected = {name: model["features"] for name, model in additions.items()}
    for name, (names, _) in candidates.items():
        conditional[f"{name}_status"] = [
            "retained"
            if feature in selected[name]
            else additions[name]["rejected"].get(
                feature, "below_training_screen" if feature not in names else "not_retained"
            )
            for feature in conditional.feature
        ]
    atomic_bytes(destination / "screening.csv", conditional.to_csv(index=False).encode())
    atomic_bytes(destination / "errors.csv", errors.to_csv(index=False).encode())
    atomic_json(destination / "summary.json", summary)


def feature_research(root: Path, run: Run, checkpoint: Callable[[], object] | None = None) -> None:
    destination = root / "artifacts/research"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        caches, inputs = verify_inputs(root)
        splits, _ = protocol(root)
        folds = chronological_folds(splits)
        source_signature = research_signature(
            root, inputs, {"method": 1, "alpha": ALPHA, "folds": folds}
        )
        # A fingerprinted plan is written before any outer-validation comparison.
        plan = {
            "format": 1,
            "source_signature": source_signature,
            "inputs": inputs,
            "folds": folds,
            "alpha": ALPHA,
            "candidate_features": len(research_catalog()),
            "selection": "pooled inner-fold coordinate RMSE; ties by model name",
            "holdout_evaluation": "not_run",
        }
        atomic_json(destination / "plan.json", plan)
        results = []
        for fold in folds:
            folder = destination / fold["name"]
            outputs = [
                folder / name
                for name in ("models.json", "screening.csv", "errors.csv", "summary.json")
            ]
            sig = hashlib.sha256(
                (source_signature + json.dumps(fold, sort_keys=True)).encode()
            ).hexdigest()
            stage(
                root,
                f"research-{fold['name']}",
                sig,
                outputs,
                partial(run_fold, root, caches, fold, folder, run),
                run,
            )
            results.append(json.loads((folder / "summary.json").read_text()))
            if checkpoint:
                checkpoint()
        pooled: dict[str, list[float]] = {}
        for result in results:
            for row in result["models"]:
                if row["model"] in {
                    "landing_refit",
                    "plus_balanced",
                    *(f"plus_{g}" for g in GROUPS),
                }:
                    previous = pooled.setdefault(row["model"], [0.0, 0])
                    n = result["validation_rows_per_model"]
                    previous[0] += row["coordinate_rmse_yards"] ** 2 * n
                    previous[1] += n
        inner_scores = {name: float(np.sqrt(sse / n)) for name, (sse, n) in pooled.items()}
        chosen = min(inner_scores, key=lambda name: (inner_scores[name], name))
        atomic_json(
            destination / "selection.json",
            {
                "selected_model": chosen,
                "inner_scores": inner_scores,
                "source_signature": source_signature,
                "outer_validation_used_for_selection": False,
            },
        )
        fold = {
            "name": "development",
            "training_games": sorted(splits.loc[splits.split.eq("train"), "game_id"].astype(int)),
            "evaluation_games": sorted(
                splits.loc[splits.split.eq("validation"), "game_id"].astype(int)
            ),
        }
        folder = destination / fold["name"]
        outputs = [
            folder / name for name in ("models.json", "screening.csv", "errors.csv", "summary.json")
        ]
        stage(
            root,
            "research-development",
            source_signature,
            outputs,
            partial(run_fold, root, caches, fold, folder, run),
            run,
        )
        outer = json.loads((folder / "summary.json").read_text())
        outer.update(
            {
                "format": 1,
                "selected_by_inner_folds": chosen,
                "inner_scores": inner_scores,
                "inner_folds": results,
                "source_signature": source_signature,
                "feature_gate": "open",
                "final_training_ready": False,
                "limitations": [
                    "One season only; no across-season stability claim.",
                    "Previously inspected development validation is not an untouched test set.",
                    "Raw metadata and observed interaction-history avenues remain open.",
                    "Linear-probe rejection does not establish absence of nonlinear signal.",
                    "No final-model retraining, reserved holdout score or official gateway run.",
                ],
            }
        )
        atomic_json(destination / "summary.json", outer)
        if checkpoint:
            checkpoint()
        run.event(
            "feature_research_completed",
            selected_model=chosen,
            inner_rmse=inner_scores[chosen],
            candidate_features=len(research_catalog()),
            feature_gate="open",
            holdout_evaluation="not_run",
        )
