"""Auditable notebook analysis of completed experiments; never fit or submit silently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.feature_experiment import numerical_sources
from nfl_trajectory.runtime import sha256


def trajectory_permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Shuffle complete trajectories within role/horizon; retain forecast-frame alignment."""
    from nfl_trajectory.motion import ENTITY, KEYS, require_keys

    require_keys(frame)
    metadata = ["player_role", "num_frames_output"]
    if (frame.groupby(ENTITY)[metadata].nunique() > 1).any().any():
        raise ValueError("Permutation metadata must be constant within a trajectory.")
    entities = frame[ENTITY + metadata].drop_duplicates(ENTITY).reset_index(drop=True)
    entities["entity"] = np.arange(len(entities))
    assignment = np.arange(len(entities))
    rng = np.random.default_rng(seed)
    for _, group in entities.groupby(metadata, sort=True):
        indices = group.entity.to_numpy()
        assignment[indices] = rng.permutation(indices)
    aligned = frame[KEYS].merge(entities[ENTITY + ["entity"]], on=ENTITY, how="left", sort=False)
    lookup = aligned[["entity", "frame_id"]].assign(source_row=np.arange(len(frame)))
    queries = aligned[["entity", "frame_id"]].copy()
    queries["entity"] = assignment[queries.entity.to_numpy()]
    result = queries.merge(
        lookup, on=["entity", "frame_id"], how="left", sort=False, validate="one_to_one"
    )
    if result.source_row.isna().any():
        raise ValueError("Permutation requires complete matching forecast frame sets.")
    return result.source_row.to_numpy(int)


def research_stage_evidence(root: Path, folder_name: str) -> dict[str, Any]:
    """Read a complete research run only after checking all numerical output receipts."""
    from nfl_trajectory.context_experiment import context_signature
    from nfl_trajectory.feature_research import research_signature, verify_inputs
    from nfl_trajectory.representation_experiment import representation_signature

    if folder_name not in {"research", "context", "representation"}:
        raise ValueError("Unknown research stage.")
    folder = root / "artifacts" / folder_name
    plan = json.loads((folder / "plan.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if (
        summary.get("status") != "passed"
        or summary.get("source_signature") != plan["source_signature"]
    ):
        raise ValueError("Research result is incomplete or differs from its plan.")
    if summary.get("holdout_evaluation") != "not_run":
        raise ValueError("Feature research must leave the reserved holdout unscored.")
    if folder_name == "research":
        _, inputs = verify_inputs(root)
        signature = research_signature(
            root, inputs, {"method": 1, "alpha": 0.01, "folds": plan["folds"]}
        )
    elif folder_name == "representation":
        hashes = {path: sha256(root / path) for path in plan["inputs"]}
        signature = representation_signature(hashes, plan["parameters"])
    else:
        hashes = {path: sha256(root / path) for path in plan["inputs"]}
        signature = context_signature(
            hashes,
            {
                "parent_source": plan["parent_source_signature"],
                "parent_variant": "plus_balanced",
                "folds": plan["folds"],
            },
        )
    if signature != plan["source_signature"]:
        raise ValueError("Research code or inputs changed since the completed experiment.")
    folds = [r["fold"] for r in summary["inner_folds"]] + [summary["fold"]]
    for fold in folds:
        prefix = folder_name
        suffixes = ("",) if prefix == "research" else ("-screen", "-fit", "-evaluate")
        if prefix == "representation":
            suffixes = ("-prepare", *suffixes)
        for suffix in suffixes:
            receipt_name = f"{prefix}-{fold['name']}{suffix}.json"
            receipt = json.loads((root / ".state" / receipt_name).read_text())
            if receipt.get("status") != "completed" or not receipt.get("outputs"):
                raise ValueError("A research fold has no completed receipt.")
            for relative, digest in receipt["outputs"].items():
                if sha256(root / relative) != digest:
                    raise ValueError("Research output checksum failed.")
        fold_summary = json.loads((folder / fold["name"] / "summary.json").read_text())
        expected_summary = next((r for r in summary["inner_folds"] if r["fold"] == fold), summary)
        if fold_summary["models"] != expected_summary["models"]:
            raise ValueError("Research aggregate differs from verified fold metrics.")
    selection = json.loads((folder / "selection.json").read_text())
    if selection.get("outer_validation_used_for_selection") is not False:
        raise ValueError("Research variant must be chosen using inner folds only.")
    if selection.get("inner_scores") != summary["inner_scores"] or selection.get(
        "selected_model"
    ) != min(summary["inner_scores"], key=lambda name: (summary["inner_scores"][name], name)):
        raise ValueError("Research selection disagrees with its training-fold evidence.")
    return summary


def feature_research_snapshot(root: Path) -> dict[str, Any]:
    """Compact aggregate publication; private coefficients and tracking stay out of Git."""
    from collections import Counter

    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.representation_features import representation_catalog

    core = research_stage_evidence(root, "research")
    context = research_stage_evidence(root, "context")
    representation = research_stage_evidence(root, "representation")
    stages = {"research": core, "context": context, "representation": representation}
    family_counts = []
    retention = []
    stability = []
    fold_rows = []
    models_for_stage = {}
    for label, result in stages.items():
        folder = root / "artifacts" / label
        screen = pd.read_csv(folder / "development" / "screening.csv")
        for family, rows in screen.groupby("family", sort=True):
            family_counts.append(
                {
                    "stage": label,
                    "family": family,
                    "candidates": len(rows),
                    "nonconstant": int(rows.screen_status.eq("eligible").sum()),
                    "constant_or_near_constant": int(rows.screen_status.ne("eligible").sum()),
                }
            )
        models = json.loads((folder / "development/models.json").read_text())
        models_for_stage[label] = models
        sets = []
        for item in result["inner_folds"]:
            fold = item["fold"]
            bundle = json.loads((folder / fold["name"] / "models.json").read_text())
            sets.append(bundle)
            baseline_name = "landing_refit" if label == "research" else "core_balanced"
            reference = next(
                r["coordinate_rmse_yards"] for r in item["models"] if r["model"] == baseline_name
            )
            for row in item["models"]:
                fold_rows.append(
                    {
                        "stage": label,
                        "fold": fold["name"],
                        "training_games": len(fold["training_games"]),
                        "validation_games": len(fold["evaluation_games"]),
                        "model": row["model"],
                        "coordinate_rmse_yards": row["coordinate_rmse_yards"],
                        "improvement_vs_parent_percent": 100
                        * (1 - row["coordinate_rmse_yards"] / reference),
                    }
                )
        for name, model in models["additions"].items():
            retention.append(
                {
                    "stage": label,
                    "model": name,
                    "retained": len(model["features"]),
                    "screened": len(model["features"]) + len(model["rejected"]),
                    "rejection_reasons": dict(Counter(model["rejected"].values())),
                    "features": model["features"],
                }
            )
            counts = Counter(
                feature for bundle in sets for feature in bundle["additions"][name]["features"]
            )
            stability.append(
                {
                    "stage": label,
                    "model": name,
                    "all_three_folds": sorted(
                        feature for feature, count in counts.items() if count == 3
                    ),
                    "selected_in_any_fold": len(counts),
                    "pairwise_jaccard": [
                        len(
                            set(a["additions"][name]["features"])
                            & set(b["additions"][name]["features"])
                        )
                        / max(
                            len(
                                set(a["additions"][name]["features"])
                                | set(b["additions"][name]["features"])
                            ),
                            1,
                        )
                        for i, a in enumerate(sets)
                        for b in sets[i + 1 :]
                    ],
                }
            )
    choices = [
        (value, stage_name, name)
        for stage_name, result in stages.items()
        for name, value in result["inner_scores"].items()
    ]
    _, chosen_stage, chosen_name = min(choices)
    chosen_summary = stages[chosen_stage]
    selected_row = next(row for row in chosen_summary["models"] if row["model"] == chosen_name)
    core_models = models_for_stage["research"]
    core_addition = chosen_name if chosen_stage == "research" else "plus_balanced"
    retained = len(core_models["landing"]["features"]) + len(
        core_models["additions"].get(core_addition, {}).get("features", [])
    )
    if chosen_stage != "research":
        retained += len(
            models_for_stage[chosen_stage]["additions"].get(chosen_name, {}).get("features", [])
        )
    baseline = next(row for row in core["models"] if row["model"] == "role_ridge")
    return {
        "format": 1,
        "status": "passed",
        "candidate_features": len(research_catalog())
        + len(context_catalog())
        + len(representation_catalog()),
        "core_candidate_features": len(research_catalog()),
        "context_candidate_features": len(context_catalog()),
        "representation_candidate_features": len(representation_catalog()),
        "selected_stage": chosen_stage,
        "selected_model": chosen_name,
        "selected_metrics": selected_row,
        "selected_retained_features": retained,
        "original_role_ridge_rmse": baseline["coordinate_rmse_yards"],
        "improvement_vs_role_ridge_percent": 100
        * (1 - selected_row["coordinate_rmse_yards"] / baseline["coordinate_rmse_yards"]),
        "models": {label: result["models"] for label, result in stages.items()},
        "inner_scores": {label: result["inner_scores"] for label, result in stages.items()},
        "families": family_counts,
        "retention": retention,
        "stability": stability,
        "fold_results": fold_rows,
        "validation_games": core["validation_games"],
        "validation_rows_per_model": core["validation_rows_per_model"],
        "holdout_evaluation": "not_run",
        "feature_gate": "open",
        "final_training_ready": False,
        "source_signatures": {
            label: result["source_signature"] for label, result in stages.items()
        },
        "limitations": [
            "Fixed linear probes; conditional importance is not causal importance.",
            "Three chronological folds from one season; no across-season claim.",
            "The previously inspected development set is not an untouched holdout.",
            "Research additions are not yet promoted to the official inference exporter.",
        ],
    }


def permutation_study(root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Frozen-model family importance; shuffle whole trajectories within role and horizon."""
    from nfl_trajectory.context_experiment import context_weeks
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import candidate_matrix, research_catalog
    from nfl_trajectory.feature_experiment import target_state
    from nfl_trajectory.feature_research import (
        BATCH,
        baseline_prediction,
        history_rows,
        verify_inputs,
    )
    from nfl_trajectory.motion import KEYS
    from nfl_trajectory.representation_features import build_representation, representation_catalog

    caches, _ = verify_inputs(root)
    core = json.loads((root / "artifacts/research/development/models.json").read_text())
    context = json.loads((root / "artifacts/context/development/models.json").read_text())
    history = pd.read_csv(root / "artifacts/research/development/history.csv")
    names = snapshot["selected_model"], snapshot["selected_stage"]
    core_name = names[0] if names[1] == "research" else "plus_balanced"
    blocks = [("research", core["landing"])]
    if core_name in core["additions"]:
        blocks.append(("research", core["additions"][core_name]))
    if names[1] == "context" and names[0] in context["additions"]:
        blocks.append(("context", context["additions"][names[0]]))
    route_model = None
    if names[1] == "representation":
        rep_folder = root / "artifacts/representation/development"
        rep = json.loads((rep_folder / "models.json").read_text())
        if names[0] in rep["additions"]:
            blocks.append(("representation", rep["additions"][names[0]]))
        route_model = json.loads((rep_folder / "routes.json").read_text())
    catalog = pd.concat(
        [research_catalog(), context_catalog(), representation_catalog()], ignore_index=True
    )
    families = dict(zip(catalog.feature, catalog.family, strict=True))
    active = sorted({families[f] for _, model in blocks for f in model["features"]})
    components: dict[str, list[np.ndarray]] = {family: [] for family in active}
    errors, states, signs = [], [], []
    evaluation = set(core["fold"]["evaluation_games"])
    for bank, targets, arrays, state, basis, ctx in context_weeks(root, caches, evaluation):
        representations: dict[str, Any] = {"context": ctx}
        if route_model is not None:
            representations["representation"] = build_representation(bank, route_model)
        sign = target_state(bank, targets).sign.to_numpy()[:, None]
        prediction = baseline_prediction(state, basis, core["baseline"])
        h = history_rows(history, targets)
        current = {family: np.zeros((len(targets), 2)) for family in active}
        for start in range(0, len(targets), BATCH):
            end = start + BATCH
            for kind, model in blocks:
                features = model["features"]
                if not features:
                    continue
                x = (
                    candidate_matrix(bank, targets.iloc[start:end], h[start:end], features)
                    if kind == "research"
                    else representations[kind].matrix(targets.iloc[start:end], features)
                ).astype(float)
                standardized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
                coefficients = np.asarray(model["coefficients"])
                prediction[start:end] += sign[start:end] * (
                    standardized @ coefficients + np.asarray(model["intercept"])
                )
                for family in {families[f] for f in features}:
                    selected = [i for i, f in enumerate(features) if families[f] == family]
                    current[family][start:end] += standardized[:, selected] @ coefficients[selected]
        states.append(state[KEYS + ["player_role", "num_frames_output"]])
        signs.append(sign)
        errors.append(prediction - arrays["truth"])
        for family in active:
            components[family].append(current[family])
    frame = pd.concat(states, ignore_index=True)
    error = np.concatenate(errors)
    orientation = np.concatenate(signs)
    expected_rmse = snapshot["selected_metrics"]["coordinate_rmse_yards"]
    observed_rmse = float(np.sqrt(np.mean(error**2)))
    if not math.isclose(observed_rmse, expected_rmse, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("Attribution predictions differ from the selected experiment.")
    permutations = [trajectory_permutation(frame, seed) for seed in range(5)]
    rows = []
    for family in active:
        contribution = np.concatenate(components[family])
        deltas = [
            float(
                np.sqrt(np.mean((error + orientation * (contribution[order] - contribution)) ** 2))
            )
            - observed_rmse
            for order in permutations
        ]
        rows.append(
            {
                "family": family,
                "retained_features": sum(
                    families[f] == family for _, model in blocks for f in model["features"]
                ),
                "mean_absolute_contribution_yards": float(np.abs(contribution).mean()),
                "mean_rmse_increase_yards": float(np.mean(deltas)),
                "shuffle_sd_yards": float(np.std(deltas, ddof=1)),
                "seed_deltas_yards": deltas,
            }
        )
    return {
        "status": "passed",
        "selected_stage": names[1],
        "selected_model": names[0],
        "unpermuted_coordinate_rmse_yards": observed_rmse,
        "source_signatures": snapshot["source_signatures"],
        "rows": sorted(rows, key=lambda r: -r["mean_rmse_increase_yards"]),
        "method": (
            "Joint canonical family contribution permutation; whole trajectories within "
            "role/horizon, restored using the recipient's field direction."
        ),
        "seeds": list(range(5)),
        "baseline_and_intercepts": "fixed",
        "scope": "Selected residual feature representation on development games.",
        "limitation": (
            "Predictive, not causal. Correlated families can substitute. Cross-family "
            "combinations may be unrealistic. Shuffle SD is not a confidence interval."
        ),
        "holdout_evaluation": "not_run",
    }


def publish_research_report(root: Path, run: Any) -> None:
    """Materialize current verified aggregates; all computation remains reproducible."""
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.runtime import atomic_bytes, atomic_json, stage

    snapshot = feature_research_snapshot(root)
    destination = root / "artifacts/research/report"
    signature = hashlib.sha256(
        json.dumps(
            {"sources": snapshot["source_signatures"], "report_code": sha256(Path(__file__))},
            sort_keys=True,
        ).encode()
    ).hexdigest()

    def action() -> None:
        importance = permutation_study(root, snapshot)
        atomic_json(destination / "summary.json", snapshot)
        atomic_json(destination / "permutation.json", importance)
        research_static_figure(snapshot, destination / "stability.png")
        catalog = pd.concat(
            [
                research_catalog().assign(stage="research"),
                context_catalog().assign(stage="context"),
                representation_catalog().assign(stage="representation"),
            ],
            ignore_index=True,
        )
        atomic_bytes(
            destination / "catalog.csv",
            catalog[["feature", "family", "stage"]].to_csv(index=False).encode(),
        )

    stage(
        root,
        "research-report",
        signature,
        [
            destination / name
            for name in ("summary.json", "permutation.json", "stability.png", "catalog.csv")
        ],
        action,
        run,
    )


def load_research_report(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Use verified current local results or the clearly identified public aggregate."""
    local = root / "artifacts/research/report"
    if (local / "summary.json").exists():
        summary = json.loads((local / "summary.json").read_text())
        if summary != feature_research_snapshot(root):
            raise ValueError("Research report is stale; rerun nfl research-report.")
        receipt = json.loads((root / ".state/research-report.json").read_text())
        if receipt.get("status") != "completed":
            raise ValueError("Research report has no completed receipt.")
        for path, digest in receipt["outputs"].items():
            if sha256(root / path) != digest:
                raise ValueError("Research report output checksum failed.")
        importance = json.loads((local / "permutation.json").read_text())
        label = "Verified local feature research"
    else:
        summary = json.loads((root / "docs/results/feature_research.json").read_text())
        importance = json.loads((root / "docs/results/feature_permutation.json").read_text())
        label = "Published aggregate from the verified research run"
    if (
        summary.get("status") != "passed"
        or summary.get("holdout_evaluation") != "not_run"
        or summary.get("source_signatures") != importance.get("source_signatures")
        or summary.get("selected_model") != importance.get("selected_model")
    ):
        raise ValueError("Research report and attribution provenance disagree.")
    if sum(r["candidates"] for r in summary["families"]) != summary["candidate_features"]:
        raise ValueError("Research family counts do not cover the catalog.")
    return summary, importance, label


def feature_importance_figure(importance: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    rows = pd.DataFrame(importance["rows"]).sort_values("mean_rmse_increase_yards")
    figure = go.Figure(
        go.Bar(
            x=rows.mean_rmse_increase_yards,
            y=rows.family.str.replace("_", " "),
            orientation="h",
            error_x={"type": "data", "array": rows.shuffle_sd_yards},
            marker_color="#167c80",
            hovertemplate="%{y}<br>RMSE increase: %{x:.4f} yd<extra></extra>",
        )
    )
    figure.update_layout(
        title="Which retained feature families does the predictor use?",
        template="plotly_white",
        height=440,
        xaxis_title="Development RMSE increase after joint trajectory permutation (yards)",
        yaxis_title=None,
        margin={"l": 190, "r": 40, "t": 65, "b": 70},
    )
    return figure


def research_static_figure(snapshot: dict[str, Any], destination: Path) -> None:
    """Readable GitHub fallback for the interactive chronological-fold heatmap."""
    from io import BytesIO

    import matplotlib.pyplot as plt

    from nfl_trajectory.runtime import atomic_bytes

    rows = pd.DataFrame(snapshot["fold_results"])
    rows = rows[rows.model.str.startswith("plus_")].copy()
    rows["label"] = rows.stage + " / " + rows.model.str.removeprefix("plus_")
    pivot = rows.pivot(index="label", columns="fold", values="improvement_vs_parent_percent")
    values = pivot.to_numpy()
    limit = max(float(np.abs(values).max()), 0.1)
    fig, ax = plt.subplots(figsize=(11, 8), layout="constrained")
    chart = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(len(pivot)), pivot.index.str.replace("_", " "), fontsize=9)
    ax.set_xticks(range(len(pivot.columns)), ["Earlier", "Middle", "Later"])
    ax.set_title(
        "Feature gains must survive time\nChronological training-only evaluation folds",
        loc="left",
        fontsize=15,
        pad=18,
    )
    ax.set_xlabel("Positive: lower official coordinate RMSE versus the fixed parent")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"{values[i, j]:+.2f}%",
                ha="center",
                va="center",
                color="white" if abs(values[i, j]) > limit * 0.6 else "#17252c",
                fontsize=9,
            )
    fig.colorbar(chart, ax=ax, shrink=0.6, label="RMSE improvement (%)")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=155)
    plt.close(fig)
    atomic_bytes(destination, buffer.getvalue())


def research_comparison_figure(snapshot: dict[str, Any]) -> Any:
    """Interactive fold stability chart; positive values mean lower official RMSE."""
    import plotly.graph_objects as go

    rows = pd.DataFrame(snapshot["fold_results"])
    rows = rows[rows.model.str.startswith("plus_")]
    rows["label"] = rows.stage + " / " + rows.model.str.removeprefix("plus_")
    pivot = rows.pivot(index="label", columns="fold", values="improvement_vs_parent_percent")
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(pivot.to_numpy(), 2),
            texttemplate="%{text:.2f}%",
            colorbar={"title": "RMSE gain %"},
            hovertemplate="%{y}<br>%{x}: %{z:.2f}% lower RMSE<extra></extra>",
        )
    )
    figure.update_layout(
        title="Which feature families improve across time?",
        template="plotly_white",
        height=570,
        margin={"l": 235, "r": 80, "t": 65, "b": 60},
        xaxis_title="Chronological training-only evaluation folds",
        yaxis_title=None,
    )
    return figure


def validate_summary(summary: dict[str, Any]) -> None:
    """Reject incomplete evidence and arithmetic-inconsistent validation slices."""
    if (
        summary.get("status") != "passed"
        or summary.get("split") != "validation"
        or summary.get("screening_split") != "train"
        or summary.get("holdout_evaluation") != "not_run"
    ):
        raise ValueError("A completed, training-screened validation experiment is required.")
    scores = summary.get("models", [])
    names = [r["model"] for r in scores]
    if not scores or len(names) != len(set(names)):
        raise ValueError("Model names must be nonempty and unique.")
    for row in scores:
        for key in (
            "coordinate_rmse_yards",
            "ade_frame_weighted_yards",
            "fde_trajectory_weighted_yards",
            "p95_displacement_yards",
        ):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Metrics must be finite and nonnegative.")
        for dimension in ("role", "forecast_second"):
            records = [
                r
                for r in summary["slices"]
                if r["model"] == row["model"] and r["dimension"] == dimension
            ]
            rows = sum(r["rows"] for r in records)
            if rows != summary["validation_rows_per_model"] or rows <= 0:
                raise ValueError("Diagnostic slices must cover every scored row.")
            rmse = math.sqrt(
                sum(r["rows"] * r["coordinate_rmse_yards"] ** 2 for r in records) / rows
            )
            if not math.isclose(rmse, row["coordinate_rmse_yards"], rel_tol=1e-10):
                raise ValueError("Diagnostic slices disagree with the pooled RMSE.")
    best = min(scores, key=lambda r: r["coordinate_rmse_yards"])["model"]
    if summary.get("selected_model") != best:
        raise ValueError("Selected model is not the measured RMSE winner.")


def selection_study(bundle: dict[str, Any], summary_hash: str, model_hash: str) -> dict[str, Any]:
    """Inspect selected feature membership and standardized coefficients, not causal importance."""
    landing = bundle["models"]["landing_ridge"]
    interaction = bundle["models"]["interaction_ridge"]
    names = landing["features"]
    weights = np.asarray(landing["coefficients"], dtype=float)
    if weights.shape != (len(names), 2) or not np.isfinite(weights).all():
        raise ValueError("Invalid coefficients in completed model.")
    ranked = sorted(range(len(names)), key=lambda i: -float(np.linalg.norm(weights[i])))[:12]
    return {
        "format": 1,
        "summary_sha256": summary_hash,
        "feature_model_sha256": model_hash,
        "training_rows": landing["training_rows"],
        "landing_selected_count": len(names),
        "landing_ball_ux_count": sum("ball_ux" in name for name in names),
        "overlap_count": len(set(names) & set(interaction["features"])),
        "removed_from_landing": sorted(set(names) - set(interaction["features"])),
        "added_by_interaction": sorted(set(interaction["features"]) - set(names)),
        "coefficient_rows": [
            {
                "feature": names[i],
                "x_coefficient": float(weights[i, 0]),
                "y_coefficient": float(weights[i, 1]),
            }
            for i in ranked
        ],
    }


def load_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Prefer verified local evidence; never silently hide a partial local run."""
    local = root / "artifacts/features"
    published = root / "docs/results"
    path = local / "summary.json" if local.exists() else published / "feature_summary.json"
    summary = json.loads(path.read_text())
    validate_summary(summary)
    if summary.get("source_sha256") != numerical_sources():
        raise ValueError("Feature evidence was generated with different numerical source.")
    if local.exists():
        model_path = local / "model.json"
        checkpoint = json.loads((root / ".state/features-report.json").read_text())
        if checkpoint.get("status") != "completed" or checkpoint.get("signature") != summary.get(
            "numerical_signature"
        ):
            raise ValueError("Feature completion receipt is stale or incomplete.")
        for item in (path, model_path, local / "benchmark.png"):
            if checkpoint.get("outputs", {}).get(item.relative_to(root).as_posix()) != sha256(item):
                raise ValueError("Feature output checksum failed.")
        model = json.loads(model_path.read_text())
        baseline = root / "artifacts/benchmark/model.json"
        if (
            summary.get("baseline_sha256") != sha256(baseline)
            or model.get("baseline_sha256") != sha256(baseline)
            or summary.get("split_sha256") != sha256(root / "artifacts/game_splits.csv")
        ):
            raise ValueError("Feature evidence no longer matches the baseline or frozen split.")
        study = selection_study(model, sha256(path), sha256(model_path))
        return summary, study, "Verified local experiment"
    study = json.loads((published / "feature_selection.json").read_text())
    if study["summary_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Published selection study does not match the published summary.")
    if summary.get("baseline_sha256") != sha256(published / "model.json"):
        raise ValueError("Published evidence does not match its baseline.")
    return summary, study, "Published real-data experiment snapshot"


def error_budget(summary: dict[str, Any], dimension: str) -> pd.DataFrame:
    """Decompose pooled squared coordinate error; never average slice RMSEs."""
    validate_summary(summary)
    if dimension not in ("role", "forecast_second"):
        raise ValueError("Choose role or forecast_second.")
    selected = summary["selected_model"]
    frame = pd.DataFrame(
        [r for r in summary["slices"] if r["model"] == selected and r["dimension"] == dimension]
    )
    squared = frame.rows * frame.coordinate_rmse_yards**2
    frame["row_share_percent"] = 100 * frame.rows / frame.rows.sum()
    frame["squared_error_share_percent"] = 100 * squared / squared.sum()
    return frame[
        [
            "value",
            "rows",
            "coordinate_rmse_yards",
            "row_share_percent",
            "squared_error_share_percent",
        ]
    ]
