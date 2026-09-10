"""Compare completed chronological attention fits, matched ablations and fixed blends."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_temporal import aligned_errors  # noqa: E402

from nfl_trajectory.benchmark import bootstrap_scores, error_metrics  # noqa: E402
from nfl_trajectory.feature_research import chronological_folds  # noqa: E402
from nfl_trajectory.features import ROLES  # noqa: E402
from nfl_trajectory.motion import KEYS, require_keys  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256  # noqa: E402
from nfl_trajectory.temporal_research import VARIANTS, channel_audit, validate_fold  # noqa: E402

REFERENCE_HASHES = {
    "inner_1": "b8a45e8d8aa89f8cce14845d4c5b257ac61f3e4affb954abf6a1dd609ea0de97",
    "inner_2": "cb637456b9cf36a3a6fde5083933c1eaeb190cd57ea096a317c7cbc42996059f",
    "inner_3": "b9c4aba61cf6d9b15b1a360446964076976bb1eee798b70f9c4f15d4542c0c4d",
}


def compare(first: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Negative differences favor the first argument; bootstrap clusters are games."""
    reference, first = aligned_errors(reference, first)
    a, b = error_metrics(first), error_metrics(reference)
    differences = bootstrap_scores(first) - bootstrap_scores(reference)
    return {
        "coordinate_rmse_yards": a["coordinate_rmse_yards"],
        "reference_rmse_yards": b["coordinate_rmse_yards"],
        "rmse_difference": a["coordinate_rmse_yards"] - b["coordinate_rmse_yards"],
        "relative_rmse_reduction": 1 - a["coordinate_rmse_yards"] / b["coordinate_rmse_yards"],
        "paired_game_bootstrap_delta_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def gates(rows: list[dict[str, Any]], pooled: dict[str, Any]) -> dict[str, bool]:
    """Apply the dated criteria without selecting a favorable fold or weight."""
    full_better = all(r["full_vs_without_target_statistics"]["rmse_difference"] < 0 for r in rows)
    blend_better = all(r["equal_blend_vs_tree"]["rmse_difference"] < 0 for r in rows)
    return {
        "consistent_target_statistics_benefit": bool(
            len(rows) == 3
            and full_better
            and pooled["full_vs_without_target_statistics"]["paired_game_bootstrap_delta_ci95"][1]
            < 0
        ),
        "equal_blend_ready_for_inference_research": bool(
            len(rows) == 3
            and blend_better
            and pooled["equal_blend_vs_tree"]["relative_rmse_reduction"] >= 0.01
            and pooled["equal_blend_vs_tree"]["paired_game_bootstrap_delta_ci95"][1] < 0
        ),
    }


def main() -> None:
    folder = ROOT / "artifacts/temporal/research"
    with Run(ROOT, "temporal-research-evaluation") as run:
        plan = json.loads((folder / "plan.json").read_text())
        signature = plan.pop("signature")
        if hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest() != signature:
            raise ValueError("Research plan fingerprint changed.")
        for name, digest in {**plan["sources"], **plan["inputs"]}.items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f"Research input or source changed: {name}")
        splits = pd.read_csv(ROOT / "artifacts/game_splits.csv")
        if chronological_folds(splits) != plan["folds"]:
            raise ValueError("Chronological fold definitions changed.")
        protocol = json.loads((folder / "decision_protocol.json").read_text())
        if protocol["protocol_sha256"] != sha256(ROOT / "docs/TEMPORAL_RESEARCH.md"):
            raise ValueError("The declared decision protocol changed.")
        rows, completed, metadata = [], [], []
        all_errors: dict[str, list[pd.DataFrame]] = {
            v: [] for v in (*VARIANTS, "tree", "equal_blend")
        }
        artifacts = {}
        for fold in plan["folds"]:
            validate_fold(fold, splits)
            name = fold["name"]
            tree_path = ROOT / f"artifacts/feature_attribution/{name}/without_metadata.csv"
            if sha256(tree_path) != REFERENCE_HASHES[name]:
                raise ValueError("The preserved inner-fold tree reference changed.")
            errors = {"tree": pd.read_csv(tree_path)}
            if set(errors["tree"].game_id) != set(fold["evaluation_games"]):
                raise ValueError("Reference games do not match this chronological fold.")
            artifacts[str(tree_path.relative_to(ROOT))] = sha256(tree_path)
            tensors = folder / name / "samples.pkl"
            receipt = json.loads((ROOT / f".state/{name}-temporal-samples.json").read_text())
            if receipt["outputs"].get(str(tensors.relative_to(ROOT))) != sha256(tensors):
                raise ValueError("Diagnostic tensors failed their completed-stage checksum.")
            for sample in pickle.loads(tensors.read_bytes()):
                if sample["split"] != "validation":
                    continue
                meta = pd.DataFrame(sample["keys"], columns=KEYS)
                meta["role"] = [ROLES[i] for i in sample["role"][sample["player"]]]
                meta["forecast_time"] = pd.cut(
                    sample["time"],
                    [0, 0.5, 1, 2, np.inf],
                    labels=["0–0.5 s", "0.5–1 s", "1–2 s", ">2 s"],
                ).astype(str)
                meta["player_history"] = np.where(
                    sample["static"][sample["player"], 11] > 0, "cold", "known"
                )
                metadata.append(meta)
            for variant in VARIANTS:
                arm = folder / name / variant
                result = json.loads((arm / "summary.json").read_text())
                if (
                    result["status"] != "completed"
                    or result["fold"] != fold
                    or result["variant"] != variant
                    or result["settings"] != plan["settings"]
                    or result["epochs_completed"] != plan["settings"]["epochs"]
                    or result["signature"]
                    != hashlib.sha256((signature + variant).encode()).hexdigest()
                ):
                    raise ValueError("An ablation arm is incomplete or has different settings.")
                for file, digest in result["artifacts"].items():
                    if sha256(arm / file) != digest:
                        raise ValueError("An arm artifact failed its checksum.")
                    artifacts[str((arm / file).relative_to(ROOT))] = digest
                errors["tree"], errors[variant] = aligned_errors(
                    errors["tree"], pd.read_csv(arm / "errors.csv")
                )
                completed.append(result)
            blend = errors["tree"].copy()
            blend[["dx", "dy"]] = (
                errors["tree"][["dx", "dy"]].to_numpy() + errors["full"][["dx", "dy"]].to_numpy()
            ) / 2
            errors["equal_blend"] = blend
            row = {
                "fold": name,
                "training_games": len(fold["training_games"]),
                "evaluation_games": len(fold["evaluation_games"]),
                "rows": len(blend),
                "models": {v: error_metrics(e) for v, e in errors.items()},
                "full_vs_without_target_statistics": compare(
                    errors["full"], errors["without_target_statistics"]
                ),
                "full_vs_tree": compare(errors["full"], errors["tree"]),
                "equal_blend_vs_tree": compare(blend, errors["tree"]),
            }
            rows.append(row)
            for variant, values in errors.items():
                all_errors[variant].append(values)
        pooled_errors = {v: pd.concat(parts, ignore_index=True) for v, parts in all_errors.items()}
        meta = pd.concat(metadata, ignore_index=True)
        require_keys(meta)
        slices = []
        for variant, errors in pooled_errors.items():
            if (
                not meta[KEYS]
                .sort_values(KEYS)
                .reset_index(drop=True)
                .equals(errors[KEYS].sort_values(KEYS).reset_index(drop=True))
            ):
                raise ValueError("Diagnostic metadata must cover the exact evaluated keys.")
            joined = errors.merge(meta, on=KEYS, validate="one_to_one")
            for dimension in ("role", "forecast_time", "player_history"):
                for value, group in joined.groupby(dimension, sort=True):
                    metrics = error_metrics(group)
                    slices.append(
                        {
                            "model": variant,
                            "dimension": dimension,
                            "value": str(value),
                            "rows": len(group),
                            **{
                                key: metrics[key]
                                for key in (
                                    "coordinate_rmse_yards",
                                    "ade_frame_weighted_yards",
                                    "p95_displacement_yards",
                                )
                            },
                        }
                    )
        pooled = {
            "models": {v: error_metrics(e) for v, e in pooled_errors.items()},
            "full_vs_without_target_statistics": compare(
                pooled_errors["full"], pooled_errors["without_target_statistics"]
            ),
            "full_vs_tree": compare(pooled_errors["full"], pooled_errors["tree"]),
            "equal_blend_vs_tree": compare(pooled_errors["equal_blend"], pooled_errors["tree"]),
        }
        report = {
            "status": "completed",
            "signature": signature,
            "run_id": run.run_id,
            "decision_protocol": protocol,
            "feature_ablation": channel_audit(),
            "folds": rows,
            "pooled": pooled,
            "slices": slices,
            "gates": gates(rows, pooled),
            "fits_completed": len(completed),
            "epochs_per_fit": plan["settings"]["epochs"],
            "training_seconds": sum(r["training_seconds"] for r in completed),
            "learning_curves": [
                {"fold": r["fold"]["name"], "variant": r["variant"], "curve": r["curve"]}
                for r in completed
            ],
            "games": int(pooled_errors["tree"].game_id.nunique()),
            "rows": len(pooled_errors["tree"]),
            "bootstrap": {"resamples": 2000, "seed": 2026, "cluster": "game"},
            "artifacts": artifacts,
            "evaluator_sha256": sha256(Path(__file__)),
            "numerical_sources": {
                **plan["sources"],
                **{
                    name: digest
                    for name, digest in json.loads(
                        (ROOT / "artifacts/temporal/development/plan.json").read_text()
                    )["inputs"].items()
                    if name.startswith(("src/", "scripts/"))
                },
                str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
            },
            "limitations": [
                "One seed and historically reused folds; conditional uncertainty only.",
                "Feature effects include optimization under the fixed 40-epoch budget; "
                "they do not establish each variant's convergence-optimal performance.",
                "No ensemble weights or checkpoints selected from evaluation scores.",
                "No new reserved-set evaluation or Kaggle submission.",
                "The neural feature research gate remains open.",
            ],
        }
        atomic_json(folder / "summary.json", report)
        atomic_json(ROOT / "docs/results/temporal_research.json", report)
        labels = [r["fold"].replace("inner_", "Fold ") for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4.4), layout="constrained")
        for i, (variant, label, color) in enumerate(
            [
                ("tree", "Preserved tree", "#b66929"),
                ("full", "Attention: full", "#087e8b"),
                ("without_target_statistics", "Attention: no target statistics", "#66778a"),
                ("equal_blend", "Fixed equal blend", "#57458b"),
            ]
        ):
            ax.bar(
                np.arange(3) + (i - 1.5) * 0.2,
                [r["models"][variant]["coordinate_rmse_yards"] for r in rows],
                width=0.19,
                label=label,
                color=color,
            )
        ax.set(
            xticks=np.arange(3),
            xticklabels=labels,
            ylabel="Coordinate RMSE (yards)",
            title="Chronological confirmation: all requested rows retained",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        fig.savefig(folder / "comparison.png", dpi=180)
        plt.close(fig)
        atomic_bytes(
            ROOT / "docs/results/temporal_research.png", (folder / "comparison.png").read_bytes()
        )
        run.event("research_evaluated", fits=6, gates=report["gates"], pooled=pooled["models"])


if __name__ == "__main__":
    main()
