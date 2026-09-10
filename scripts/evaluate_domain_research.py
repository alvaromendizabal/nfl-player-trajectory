"""Evaluate every completed domain arm without selecting the minimum observed score."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import bootstrap_scores, error_metrics  # noqa: E402
from nfl_trajectory.domain_features import FAMILIES  # noqa: E402
from nfl_trajectory.features import ROLES  # noqa: E402
from nfl_trajectory.motion import KEYS, require_keys  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256  # noqa: E402
from nfl_trajectory.temporal_research import validate_fold  # noqa: E402

REPEATS = 10000


def comparison(
    first: pd.DataFrame, reference: pd.DataFrame, multiplicity: int = 1
) -> dict[str, Any]:
    reference, first = aligned_errors(reference, first)
    score, baseline = (error_metrics(x)["coordinate_rmse_yards"] for x in (first, reference))
    draws = bootstrap_scores(first, REPEATS) - bootstrap_scores(reference, REPEATS)
    alpha = 0.05 / multiplicity
    return {
        "coordinate_rmse_yards": score,
        "reference_rmse_yards": baseline,
        "rmse_difference": score - baseline,
        "relative_reduction": 1 - score / baseline,
        "paired_game_delta_interval": np.quantile(draws, [alpha / 2, 1 - alpha / 2]).tolist(),
        "interval_level": 1 - alpha,
        "simultaneous_comparisons": multiplicity,
    }


def decision(folds: list[dict[str, Any]], pooled: dict[str, Any]) -> bool:
    return bool(
        len(folds) == 3
        and all(f["primary"]["rmse_difference"] < 0 for f in folds)
        and pooled["relative_reduction"] >= 0.01
        and pooled["paired_game_delta_interval"][1] < 0
    )


def training_concentration(archive: Any) -> dict[str, Any]:
    training = archive["train"]
    frame = pd.DataFrame(archive["keys"][training], columns=KEYS)
    frame["sse"] = (archive["y"][training].astype(float) ** 2).sum(axis=1)
    frame["seconds"] = archive["time"][training]
    plays = frame.groupby(KEYS[:2]).sse.agg(["sum", "count"]).sort_values("sum", ascending=False)
    total = frame.sse.sum()
    return {
        "rows": len(frame),
        "plays": len(plays),
        "frozen_attention_training_rmse": float(np.sqrt(total / (2 * len(frame)))),
        "top_play_error_share": float(plays["sum"].iloc[0] / total),
        "top_one_percent_plays_error_share": float(
            plays["sum"].head(max(1, int(np.ceil(len(plays) * 0.01)))).sum() / total
        ),
        "after_two_seconds_row_share": float(frame.seconds.gt(2).mean()),
        "after_two_seconds_error_share": float(frame.loc[frame.seconds.gt(2), "sse"].sum() / total),
        "top_plays": plays.head(5).reset_index().to_dict("records"),
    }


def plot_report(report: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    chosen = ["tree", "temporal", "control", "all", "all_equal_blend"]
    labels = ["Tree", "Attention", "Matched control", "Domain features", "Domain + tree"]
    colors = ["#64748b", "#9ca3af", "#d97706", "#2563eb", "#059669"]
    x = np.arange(3)
    for i, (arm, label, color) in enumerate(zip(chosen, labels, colors, strict=True)):
        values = [f["scores"][arm]["coordinate_rmse_yards"] for f in report["folds"]]
        axes[0].bar(x + (i - 2) * 0.15, values, 0.15, label=label, color=color)
    axes[0].set_xticks(x, ["Fold 1", "Fold 2", "Fold 3"])
    axes[0].set_ylabel("Coordinate RMSE (yards; lower is better)")
    axes[0].set_title("Same chronological evaluation rows")
    axes[0].set_ylim(
        0,
        max(
            1.0,
            max(f["scores"][a]["coordinate_rmse_yards"] for f in report["folds"] for a in chosen)
            * 1.1,
        ),
    )
    axes[0].legend(frameon=False, fontsize=8)
    for i, row in enumerate(report["family_comparisons"]):
        effect, interval = row["rmse_difference"], row["paired_game_delta_interval"]
        axes[1].plot(
            interval, [i, i], color="#2563eb" if row["comparison"] == "addition" else "#d97706"
        )
        axes[1].plot(
            effect,
            i,
            "o",
            color="#2563eb" if row["comparison"] == "addition" else "#d97706",
            markersize=4,
        )
    axes[1].axvline(0, color="#64748b", linewidth=1)
    axes[1].set_yticks(
        range(8),
        [
            r["family"].replace("_", " ") + " / " + r["comparison"]
            for r in report["family_comparisons"]
        ],
        fontsize=8,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("RMSE change with family (negative favors inclusion)")
    axes[1].set_title("Family effects: simultaneous 95% intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    folder = ROOT / "artifacts/domain_research"
    with Run(ROOT, "domain-research-evaluation") as run:
        plan = json.loads((folder / "plan.json").read_text())
        signature = plan.pop("signature")
        if hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() != signature:
            raise ValueError("Domain protocol fingerprint changed.")
        artifacts = {"artifacts/domain_research/plan.json": sha256(folder / "plan.json")}
        for name, digest in {**plan["sources"], **plan["inputs"]}.items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Domain source or input changed: " + name)
        splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
        models = [*plan["arms"], "tree", "temporal", "all_equal_blend", "control_equal_blend"]
        accumulated: dict[str, list[pd.DataFrame]] = {a: [] for a in models}
        folds, completed, screens, metadata, concentration, catalog = [], [], [], [], [], []
        for fold in plan["folds"]:
            validate_fold(fold, splits)
            name = fold["name"]
            destination = folder / name
            for stage_name, paths in (
                ("features", ["features.npz", "schema.json"]),
                ("screen", ["screen.json"]),
            ):
                receipt = json.loads((ROOT / f".state/{name}-domain-{stage_name}.json").read_text())
                if receipt["status"] != "completed" or receipt["signature"] != signature:
                    raise ValueError("Domain materialization is incomplete or stale.")
                for file in paths:
                    path = destination / file
                    relative = str(path.relative_to(ROOT))
                    if receipt["outputs"].get(relative) != sha256(path):
                        raise ValueError("Domain feature or screening artifact changed.")
                    artifacts[relative] = sha256(path)
            schema = json.loads((destination / "schema.json").read_text())
            fitted = json.loads((destination / "screen.json").read_text())
            reasons = Counter(r["reason"] for r in fitted["rejected"])
            screens.append(
                {
                    "fold": name,
                    "generated": fitted["candidate_count"],
                    "retained": fitted["retained_count"],
                    "rejected": len(fitted["rejected"]),
                    "reasons": dict(reasons),
                    "reference_replay_max_absolute_difference": schema[
                        "reference_replay_max_absolute_difference"
                    ],
                    "candidate_families": dict(
                        Counter(f for f in schema["families"] if f != "control")
                    ),
                    "max_evaluation_clipped_fraction": max(fitted["evaluation_clipped_fraction"]),
                }
            )
            for feature, family, retained in zip(
                schema["names"], schema["families"], fitted["retained"], strict=True
            ):
                if family != "control":
                    catalog.append(
                        {
                            "fold": name,
                            "feature": feature,
                            "family": family,
                            "retained": retained,
                            "reason": next(
                                (
                                    r["reason"]
                                    for r in fitted["rejected"]
                                    if r["feature"] == feature
                                ),
                                "retained",
                            ),
                        }
                    )
            with np.load(destination / "features.npz", allow_pickle=False) as archive:
                mask = ~archive["train"]
                meta = pd.DataFrame(archive["keys"][mask], columns=KEYS)
                meta["role"] = [ROLES[i] for i in archive["role"][mask]]
                meta["forecast_time"] = pd.cut(
                    archive["time"][mask],
                    [0, 0.5, 1, 2, np.inf],
                    labels=["0–0.5 s", "0.5–1 s", "1–2 s", ">2 s"],
                ).astype(str)
                meta["player_history"] = np.where(archive["cold"][mask], "cold", "known")
                metadata.append(meta)
                concentration.append({"fold": name, **training_concentration(archive)})
            values = {
                "tree": pd.read_csv(
                    ROOT / f"artifacts/feature_attribution/{name}/without_metadata.csv"
                ),
                "temporal": pd.read_csv(
                    ROOT / f"artifacts/temporal/research/{name}/full/errors.csv"
                ),
            }
            for arm in plan["arms"]:
                arm_folder = destination / arm
                result = json.loads((arm_folder / "summary.json").read_text())
                expected = hashlib.sha256(
                    json.dumps(
                        {"signature": signature, "fold": name, "arm": arm}, sort_keys=True
                    ).encode()
                ).hexdigest()
                if (
                    result["status"] != "completed"
                    or result["signature"] != expected
                    or result["epochs"] != plan["settings"]["epochs"]
                    or result["settings"] != plan["settings"]
                ):
                    raise ValueError("Incomplete or unmatched domain arm.")
                for file, digest in result["artifacts"].items():
                    path = arm_folder / file
                    if sha256(path) != digest:
                        raise ValueError("Domain arm artifact hash mismatch.")
                    artifacts[str(path.relative_to(ROOT))] = digest
                artifacts[str((arm_folder / "summary.json").relative_to(ROOT))] = sha256(
                    arm_folder / "summary.json"
                )
                values["tree"], values[arm] = aligned_errors(
                    values["tree"], pd.read_csv(arm_folder / "errors.csv")
                )
                completed.append(result)
            if len({r["parameter_count"] for r in completed if r["fold"] == name}) != 1:
                raise ValueError("Feature arms changed model capacity.")
            values["tree"], values["temporal"] = aligned_errors(values["tree"], values["temporal"])
            if set(values["tree"].game_id) != set(fold["evaluation_games"]):
                raise ValueError("Evaluation games differ from the declared fold.")
            for arm in ("all", "control"):
                blend = values["tree"].copy()
                blend[["dx", "dy"]] = (
                    values[arm][["dx", "dy"]].to_numpy() + blend[["dx", "dy"]].to_numpy()
                ) / 2
                values[arm + "_equal_blend"] = blend
            folds.append(
                {
                    "fold": name,
                    "rows": len(values["tree"]),
                    "training_games": len(fold["training_games"]),
                    "evaluation_games": len(fold["evaluation_games"]),
                    "scores": {a: error_metrics(v) for a, v in values.items()},
                    "primary": comparison(values["all"], values["control"]),
                }
            )
            for arm in models:
                accumulated[arm].append(values[arm])
        pooled = {a: pd.concat(v, ignore_index=True) for a, v in accumulated.items()}
        for value in pooled.values():
            require_keys(value)
        comparisons = []
        for family in FAMILIES:
            comparisons.extend(
                [
                    {
                        "family": family,
                        "comparison": "addition",
                        **comparison(pooled[family], pooled["control"], 8),
                    },
                    {
                        "family": family,
                        "comparison": "ablation",
                        **comparison(pooled["all"], pooled["without_" + family], 8),
                    },
                ]
            )
        meta = pd.concat(metadata, ignore_index=True)
        require_keys(meta)
        slices = []
        for arm in ("tree", "temporal", "control", "all", "all_equal_blend"):
            joined = pooled[arm].merge(meta, on=KEYS, validate="one_to_one")
            if len(joined) != len(meta):
                raise ValueError("Domain diagnostic rows were lost.")
            for dimension in ("role", "forecast_time", "player_history"):
                for label, subset in joined.groupby(dimension):
                    slices.append(
                        {
                            "arm": arm,
                            "dimension": dimension,
                            "value": str(label),
                            "rows": len(subset),
                            **error_metrics(subset),
                        }
                    )
        primary = comparison(pooled["all"], pooled["control"])
        report = {
            "status": "completed",
            "signature": signature,
            "run_id": run.run_id,
            "folds": folds,
            "pooled": {a: error_metrics(v) for a, v in pooled.items()},
            "primary": primary,
            "family_comparisons": comparisons,
            "secondary_blend_vs_tree": comparison(pooled["all_equal_blend"], pooled["tree"]),
            "secondary_blend_vs_control_blend": comparison(
                pooled["all_equal_blend"], pooled["control_equal_blend"]
            ),
            "screen_passed": decision(folds, primary),
            "feature_completion_gate": "open",
            "screens": screens,
            "training_error_concentration": concentration,
            "slices": slices,
            "fits": len(completed),
            "epochs_per_fit": plan["settings"]["epochs"],
            "parameter_counts": sorted({r["parameter_count"] for r in completed}),
            "training_seconds": sum(r["training_seconds"] for r in completed),
            "learning_curves": [
                {"fold": r["fold"], "arm": r["arm"], "curve": r["curve"]} for r in completed
            ],
            "rows": len(meta),
            "games": int(meta.game_id.nunique()),
            "bootstrap": {
                "unit": "whole_game",
                "repeats": REPEATS,
                "seed": 2026,
                "family_multiplicity": 8,
            },
            "artifacts": artifacts,
            "sources": plan["sources"],
            "evaluator_sha256": sha256(Path(__file__)),
            "limitations": [
                "One seed, twelve-epoch correction heads on frozen representations.",
                "Reused chronological folds; inference is conditional on prior research.",
                "Corrections use in-sample training residuals, not independent stacking.",
                "No feature family is proven useless by one bounded residual-head screen.",
                "No new Kaggle, original development, or reserved holdout evaluation.",
            ],
        }
        atomic_json(folder / "report.json", report)
        atomic_json(ROOT / "docs/results/domain_research.json", report)
        atomic_bytes(
            ROOT / "docs/results/domain_feature_catalog.csv",
            pd.DataFrame(catalog).to_csv(index=False).encode(),
        )
        plot_report(report, ROOT / "docs/results/domain_research.png")
        run.event(
            "domain_study_evaluated",
            fits=report["fits"],
            rows=report["rows"],
            screen_passed=report["screen_passed"],
            **primary,
        )


if __name__ == "__main__":
    main()
