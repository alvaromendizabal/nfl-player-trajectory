"""Bounded, resumable feature-family ablations against the preserved motion baseline.

Candidates are screened using training residuals only. Fixed regularization and
training-correlation pruning make this an interpretable challenger, not a claim
that thousands of features are useful or that ridge is a competition-winning model.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.benchmark import bootstrap_scores, protocol, summarize
from nfl_trajectory.features import REQUIRED, PlayerFeatures, build_player_features, feature_catalog
from nfl_trajectory.models import design, predict_from_design
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

BATCH_ROWS = 1024
SCREEN_LIMIT = 96
SELECT_LIMIT = 64
RIDGE_ALPHA = 0.01
ABLATIONS = {
    "motion_ridge": {"motion"},
    "landing_ridge": {"motion", "landing"},
    "interaction_ridge": {"motion", "landing", "interaction"},
}


def numerical_sources() -> dict[str, str]:
    return {f"{name}.py": sha256(Path(__file__).with_name(f"{name}.py"))
            for name in ("feature_experiment", "features", "models", "motion", "benchmark")}


def signature(root: Path, files: list[Path], settings: Any) -> str:
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    if (root / "uv.lock").exists():
        hashes["uv.lock"] = sha256(root / "uv.lock")
    payload = {"files": hashes, "source": numerical_sources(), "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_week(root: Path, pair: dict[str, Any], splits: pd.DataFrame,
              baseline: dict[str, Any], path: Path) -> None:
    x_path = root / "data/raw/train" / pair["file"]
    allowed = REQUIRED + ["s", "a", "dir", "o", "absolute_yardline_number"]
    observed = pd.read_csv(x_path, usecols=lambda c: c in allowed)
    truth = pd.read_csv(x_path.with_name(x_path.name.replace("input_", "output_")),
                        usecols=KEYS + ["x", "y"])
    games = set(splits.loc[splits.split != "holdout", "game_id"])
    observed = observed[observed.game_id.isin(games)]
    truth = truth[truth.game_id.isin(games)].reset_index(drop=True)
    require_keys(truth)
    if not np.isfinite(truth[["x", "y"]].to_numpy(float)).all():
        raise ValueError("Training/validation targets must be finite.")
    bank = build_player_features(observed, truth[ENTITY])
    state, basis = design(observed, truth[KEYS])
    ridge = predict_from_design(state, basis, "role_ridge", baseline)[["x", "y"]].to_numpy()
    velocity = predict_from_design(state, basis, "constant_velocity")[["x", "y"]].to_numpy()
    labels = truth[["game_id"]].merge(splits[["game_id", "split"]], on="game_id",
                                     how="left", sort=False, validate="many_to_one").split
    if labels.isna().any() or not labels.isin(["train", "validation"]).all():
        raise ValueError("Only explicitly partitioned development rows may enter a feature cache.")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, keys=truth[KEYS].to_numpy(np.int64),
                        truth=truth[["x", "y"]].to_numpy(float), ridge=ridge, velocity=velocity,
                        labels=labels.to_numpy(str), bank_keys=bank.state[ENTITY].to_numpy(np.int64),
                        bank_state=bank.state[["x", "y", "sign", "num_frames_output"]].to_numpy(float),
                        roles=bank.state.player_role.to_numpy(str), features=bank.values)
    atomic_bytes(path, buffer.getvalue())


def load_week(path: Path) -> tuple[PlayerFeatures, pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as stored:
        state = pd.DataFrame(stored["bank_keys"], columns=ENTITY)
        state[["x", "y", "sign", "num_frames_output"]] = stored["bank_state"]
        state["player_role"] = stored["roles"]
        bank = PlayerFeatures(state, stored["features"])
        targets = pd.DataFrame(stored["keys"], columns=KEYS)
        arrays = {name: stored[name] for name in ("truth", "ridge", "velocity", "labels")}
    return bank, targets, arrays


def target_state(bank: PlayerFeatures, targets: pd.DataFrame) -> pd.DataFrame:
    return targets[KEYS].merge(bank.state, on=ENTITY, how="left", sort=False, validate="many_to_one")


def screen_week(cache: Path, output: Path) -> None:
    bank, targets, arrays = load_week(cache)
    mask = arrays["labels"] == "train"
    targets = targets.loc[mask].reset_index(drop=True)
    y = (arrays["truth"][mask] - arrays["ridge"][mask])
    if len(targets):
        y *= target_state(bank, targets).sign.to_numpy()[:, None]
    n = len(feature_catalog())
    sx, sxx, sxy = np.zeros(n), np.zeros(n), np.zeros((n, 2))
    for start in range(0, len(targets), BATCH_ROWS):
        x = bank.matrix(targets.iloc[start:start + BATCH_ROWS]).astype(float)
        response = y[start:start + BATCH_ROWS]
        sx += x.sum(0)
        sxx += np.einsum("nf,nf->f", x, x)
        sxy += x.T @ response
    atomic_json(output, {"count": len(targets), "sx": sx.tolist(), "sxx": sxx.tolist(),
                         "sxy": sxy.tolist(), "sy": y.sum(0).tolist(),
                         "syy": (y**2).sum(0).tolist()})


def screening(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = [json.loads(p.read_text()) for p in paths]
    count = sum(r["count"] for r in rows)
    if count <= 1:
        raise ValueError("At least two training target rows are required for feature screening.")
    sums = {key: sum(np.asarray(r[key], dtype=float) for r in rows)
            for key in ("sx", "sxx", "sxy", "sy", "syy")}
    vx = np.maximum(sums["sxx"] - sums["sx"]**2 / count, 0.0)
    vy = np.maximum(sums["syy"] - sums["sy"]**2 / count, 0.0)
    covariance = sums["sxy"] - np.outer(sums["sx"], sums["sy"]) / count
    denominator = np.sqrt(vx[:, None] * vy[None, :])
    correlations = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                             where=denominator > 1e-12)
    catalog = feature_catalog()
    catalog["training_residual_correlation"] = np.max(np.abs(np.clip(correlations, -1, 1)), axis=1)
    catalog["training_variance"] = vx / count
    selections = {}
    for name, families in ABLATIONS.items():
        eligible = catalog[catalog.signal.isin(families) & (catalog.training_variance > 1e-10)]
        ranked = eligible.sort_values(["training_residual_correlation", "feature"],
                                      ascending=[False, True])
        chosen = ranked.head(SCREEN_LIMIT).feature.tolist()
        if not chosen:
            raise ValueError("No nonconstant training features remain.")
        selections[name] = chosen
    return catalog, selections


def regression_week(cache: Path, names: list[str], output: Path) -> None:
    bank, targets, arrays = load_week(cache)
    mask = arrays["labels"] == "train"
    targets = targets.loc[mask].reset_index(drop=True)
    y = arrays["truth"][mask] - arrays["ridge"][mask]
    if len(targets):
        y *= target_state(bank, targets).sign.to_numpy()[:, None]
    gram, rhs, sx = np.zeros((len(names), len(names))), np.zeros((len(names), 2)), np.zeros(len(names))
    for start in range(0, len(targets), BATCH_ROWS):
        x = bank.matrix(targets.iloc[start:start + BATCH_ROWS], names).astype(float)
        gram += x.T @ x
        rhs += x.T @ y[start:start + BATCH_ROWS]
        sx += x.sum(0)
    atomic_json(output, {"count": len(targets), "gram": gram.tolist(), "rhs": rhs.tolist(),
                         "sx": sx.tolist(), "sy": y.sum(0).tolist()})


def fit_regression(paths: list[Path], names: list[str]) -> dict[str, Any]:
    records = [json.loads(p.read_text()) for p in paths]
    count = sum(r["count"] for r in records)
    if count <= 1:
        raise ValueError("Training-only regression statistics are empty.")
    total = {key: sum(np.asarray(r[key], float) for r in records) for key in ("gram", "rhs", "sx", "sy")}
    mean, intercept = total["sx"] / count, total["sy"] / count
    covariance = total["gram"] / count - np.outer(mean, mean)
    scales = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    normalized = covariance / np.outer(scales, scales)
    kept: list[int] = []
    for index in range(len(names)):
        if covariance[index, index] > 1e-10 and (not kept or np.abs(normalized[index, kept]).max() < 0.9995):
            kept.append(index)
        if len(kept) == SELECT_LIMIT:
            break
    if not kept:
        raise ValueError("Training-correlation pruning removed every feature.")
    lhs = normalized[np.ix_(kept, kept)] + RIDGE_ALPHA * np.eye(len(kept))
    rhs = (total["rhs"][kept] / count - np.outer(mean[kept], intercept)) / scales[kept, None]
    coefficients = np.linalg.solve(lhs, rhs)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite ridge coefficients.")
    return {"features": [names[i] for i in kept], "mean": mean[kept].tolist(),
            "scale": scales[kept].tolist(), "coefficients": coefficients.tolist(),
            "intercept": intercept.tolist(), "alpha": RIDGE_ALPHA, "training_rows": count}


def predict_residual(bank: PlayerFeatures, targets: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    output = np.empty((len(targets), 2), dtype=float)
    names = model["features"]
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    weights, intercept = np.asarray(model["coefficients"]), np.asarray(model["intercept"])
    if mean.shape != (len(names),) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Invalid feature scaler.")
    if weights.shape != (len(names), 2) or intercept.shape != (2,):
        raise ValueError("Invalid residual coefficient shape.")
    for start in range(0, len(targets), BATCH_ROWS):
        x = bank.matrix(targets.iloc[start:start + BATCH_ROWS], names).astype(float)
        output[start:start + len(x)] = ((x - mean) / scale) @ weights + intercept
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite residual predictions.")
    return output * target_state(bank, targets).sign.to_numpy()[:, None]


def evaluate(cache: Path, models: dict[str, Any], output: Path) -> None:
    bank, targets, arrays = load_week(cache)
    mask = arrays["labels"] == "validation"
    targets = targets.loc[mask].reset_index(drop=True)
    state = target_state(bank, targets)
    records = []
    if len(targets):
        predictions = {"role_ridge": arrays["ridge"][mask], "constant_velocity": arrays["velocity"][mask]}
        for name, model in models.items():
            predictions[name] = arrays["ridge"][mask] + predict_residual(bank, targets, model)
        for name, prediction in predictions.items():
            errors = state[KEYS + ["player_role", "num_frames_output"]].copy()
            errors[["dx", "dy"]] = prediction - arrays["truth"][mask]
            errors["model"] = name
            records.append(errors)
    columns = KEYS + ["player_role", "num_frames_output", "dx", "dy", "model"]
    errors = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=columns)
    atomic_bytes(output, errors.to_csv(index=False).encode())


def report(summary: dict[str, Any], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["models"]
    values = np.array([r["coordinate_rmse_yards"] for r in rows])
    errors = np.array([[r["coordinate_rmse_yards"] - r["rmse_ci95_low"] for r in rows],
                       [r["rmse_ci95_high"] - r["coordinate_rmse_yards"] for r in rows]])
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    positions = np.arange(len(rows))
    ax.barh(positions, values, xerr=np.maximum(errors, 0), capsize=4)
    ax.set_yticks(positions, [r["model"].replace("_", " ") for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Coordinate RMSE (yards) — lower is better")
    ax.set_title("Feature-family ablation | frozen temporal validation", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for i, value in enumerate(values):
        ax.text(value + max(errors[1, i], 0) + 0.015, i, f"{value:.4f}", va="center")
    ax.set_xlim(0, float((values + np.maximum(errors[1], 0)).max()) * 1.2)
    fig.supxlabel("95% game-cluster bootstrap intervals. Holdout remains unscored.", fontsize=9)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    atomic_bytes(destination, buffer.getvalue())


def feature_experiment(root: Path, run: Run, checkpoint: Callable[[], object] | None = None) -> None:
    destination = root / "artifacts/features"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / "pipeline.lock"), timeout=1):
        splits, audit = protocol(root)
        baseline_path = root / "artifacts/benchmark/model.json"
        if not baseline_path.exists():
            raise ValueError("Complete nfl benchmark once before the feature experiment.")
        baseline = json.loads(baseline_path.read_text())
        training = sorted(int(g) for g in splits.loc[splits.split == "train", "game_id"])
        if baseline.get("training_games") != training or baseline.get("split_sha256") != sha256(root / "artifacts/game_splits.csv"):
            raise ValueError("Baseline training games do not match the frozen protocol.")
        development = set(splits.loc[splits.split != "holdout", "game_id"])
        pairs = [p for p in audit["pairs"] if set(p["games"]) & development]
        caches, screens = [], []
        run.event("feature_schema", candidate_features=len(feature_catalog()), batch_rows=BATCH_ROWS,
                  screen_limit=SCREEN_LIMIT, selected_limit=SELECT_LIMIT, holdout_evaluation="not_run")
        for i, pair in enumerate(pairs, 1):
            name = Path(pair["file"]).stem
            cache = destination / "weeks" / name / "features.npz"
            source = root / "data/raw/train" / pair["file"]
            inputs = [source, source.with_name(source.name.replace("input_", "output_")),
                      baseline_path, root / "artifacts/game_splits.csv"]
            stage(root, f"features-prepare-{name}", signature(root, inputs, {"prepare": 1}), [cache],
                  partial(save_week, root, pair, splits, baseline, cache), run)
            screen = cache.with_name("screening.json")
            stage(root, f"features-screen-{name}", signature(root, [cache], {"screen": "train_only"}),
                  [screen], partial(screen_week, cache, screen), run)
            caches.append(cache)
            screens.append(screen)
            if checkpoint is not None:
                checkpoint()
            run.event("feature_progress", phase="prepare_and_screen", completed_weeks=i, total_weeks=len(pairs))
        catalog, selections = screening(screens)
        models = {}
        for name, columns in selections.items():
            statistics = []
            for i, cache in enumerate(caches, 1):
                stats = cache.with_name(f"{name}.json")
                stage(root, f"features-statistics-{name}-{cache.parent.name}",
                      signature(root, [cache], {"features": columns}), [stats],
                      partial(regression_week, cache, columns, stats), run)
                statistics.append(stats)
                run.event("feature_progress", phase="fit_statistics", model=name,
                          completed_weeks=i, total_weeks=len(caches))
            model_path = destination / "models" / f"{name}.json"
            stage(root, f"features-fit-{name}", signature(root, statistics, {"alpha": RIDGE_ALPHA, "features": columns}),
                  [model_path], lambda p=model_path, s=statistics, c=columns: atomic_json(p, fit_regression(s, c)), run)
            models[name] = json.loads(model_path.read_text())
            if checkpoint is not None:
                checkpoint()
            run.event("features_selected", model=name, selected_features=len(models[name]["features"]))
        model_path = destination / "model.json"
        atomic_json(model_path, {"format": 1, "models": models, "training_games": training,
                                "split_sha256": baseline["split_sha256"], "baseline_sha256": sha256(baseline_path),
                                "source_sha256": numerical_sources()})
        errors_paths = []
        for i, cache in enumerate(caches, 1):
            output = cache.with_name("errors.csv")
            stage(root, f"features-evaluate-{cache.parent.name}", signature(root, [cache, model_path], {"evaluate": 1}),
                  [output], partial(evaluate, cache, models, output), run)
            errors_paths.append(output)
            run.event("feature_progress", phase="evaluate", completed_weeks=i, total_weeks=len(caches))
        summary_path, figure_path = destination / "summary.json", destination / "benchmark.png"
        report_signature = signature(root, errors_paths + [model_path], {"report": 1})

        def aggregate() -> None:
            frames = [pd.read_csv(p) for p in errors_paths]
            errors = pd.concat([f for f in frames if len(f)], ignore_index=True)
            summary = summarize(errors)
            if summary["validation_games"] != int(splits.split.eq("validation").sum()):
                raise ValueError("Validation must cover every frozen validation game.")
            reference = bootstrap_scores(errors[errors.model == "role_ridge"])
            for row in summary["models"]:
                delta = bootstrap_scores(errors[errors.model == row["model"]]) - reference
                row["delta_vs_role_ridge_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
                row["selected_features"] = len(models[row["model"]]["features"]) if row["model"] in models else (6 if row["model"] == "role_ridge" else 1)
            summary.update({"candidate_features": len(catalog), "screening_split": "train",
                            "split_sha256": baseline["split_sha256"], "numerical_signature": report_signature,
                            "source_sha256": numerical_sources(), "baseline_sha256": sha256(baseline_path),
                            "family_counts": {str(k): int(v) for k, v in catalog.groupby("family").size().items()},
                            "top_training_associations": catalog.sort_values("training_residual_correlation", ascending=False)
                            .head(20).to_dict(orient="records"),
                            "limitation": ("Training-only univariate screening and fixed ridge; "
                                           "associations are not causal importance. No leaderboard claim."),
                            "official_gateway_status": "not_run"})
            report(summary, figure_path)
            atomic_json(summary_path, summary)

        stage(root, "features-report", report_signature, [summary_path, figure_path, model_path], aggregate, run)
        if checkpoint is not None:
            checkpoint()
        result = json.loads(summary_path.read_text())
        run.event("feature_experiment_completed", selected_model=result["selected_model"],
                  validation_rmse=result["models"][0]["coordinate_rmse_yards"],
                  candidate_features=len(catalog), holdout_evaluation="not_run",
                  elapsed_total_seconds=round(time.monotonic() - run.started, 3))
