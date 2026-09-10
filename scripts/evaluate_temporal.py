"""Compare one completed temporal trial with exact preserved development predictions."""

from __future__ import annotations

import argparse
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

from nfl_trajectory.benchmark import bootstrap_scores, error_metrics  # noqa: E402
from nfl_trajectory.features import ROLES  # noqa: E402
from nfl_trajectory.motion import KEYS, require_keys  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256  # noqa: E402

REFERENCE_SHA256 = "aed587e9de2b3ff1dac2fb36d7d96197710fb6ba249872daa90c078ddbf5cfba"
REFERENCE_RMSE = 0.6880521879583761


def aligned_errors(
    reference: pd.DataFrame, challenger: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require exactly the same rows; sorting must never hide missing predictions."""
    for frame in (reference, challenger):
        require_keys(frame)
        if not np.isfinite(frame[["dx", "dy"]].to_numpy()).all():
            raise ValueError("Comparison errors must be finite.")
    reference = reference.sort_values(KEYS).reset_index(drop=True)
    challenger = challenger.sort_values(KEYS).reset_index(drop=True)
    if not reference[KEYS].equals(challenger[KEYS]):
        raise ValueError("Development comparison requires exactly identical forecast keys.")
    return reference, challenger


def draw(report: dict[str, Any], folder: Path) -> None:
    curve = pd.DataFrame(report["curve"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(
        curve.epoch,
        curve.training_rmse_augmented,
        label="Training, reflected examples",
        color="#587388",
    )
    development = curve.dropna(subset=["development_rmse"])
    axes[0].plot(
        development.epoch,
        development.development_rmse,
        "o-",
        color="#087e8b",
        label="Development, EMA",
    )
    axes[0].axhline(
        REFERENCE_RMSE, color="#b66929", linestyle="--", label="Existing development tree"
    )
    axes[0].set(
        xlabel="Completed epoch",
        ylabel="Coordinate RMSE (yards)",
        title="One predeclared training run",
    )
    axes[0].legend(fontsize=8)
    slices = pd.DataFrame(report["slices"])
    horizon = slices[slices.dimension.eq("forecast_time")]
    labels = ["0–0.5 s", "0.5–1 s", "1–2 s", ">2 s"]
    positions = np.arange(len(labels))
    for i, (name, color) in enumerate(
        (("existing_tree", "#b66929"), ("temporal_attention", "#087e8b"))
    ):
        values = horizon[horizon.model.eq(name)].set_index("value").coordinate_rmse_yards
        axes[1].bar(
            positions + (i - 0.5) * 0.36,
            values.reindex(labels),
            width=0.36,
            label=name.replace("_", " "),
            color=color,
        )
    axes[1].set(
        xticks=positions,
        xticklabels=labels,
        xlabel="Time after throw",
        ylabel="Coordinate RMSE (yards)",
        title="Every development request retained",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(folder / "diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    folder = ROOT / "artifacts/temporal/development"
    with Run(ROOT, "temporal-evaluation") as run:
        trial = json.loads((folder / "summary.json").read_text())
        if trial["status"] != "completed":
            raise ValueError("Complete the predeclared run before the comparison report.")
        for name, digest in trial["artifacts"].items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Temporal trial artifacts changed after evaluation.")
        reference_path = ROOT / "artifacts/feature_attribution/development/without_metadata.csv"
        if sha256(reference_path) != REFERENCE_SHA256:
            raise ValueError("Reference errors differ from the preserved development experiment.")
        reference, challenger = aligned_errors(
            pd.read_csv(reference_path), pd.read_csv(folder / "errors.csv")
        )
        if not np.isclose(
            error_metrics(reference)["coordinate_rmse_yards"], REFERENCE_RMSE, rtol=0, atol=1e-12
        ):
            raise ValueError("Recovered reference does not reproduce the published score.")
        metadata = []
        horizon_max = 0
        for path in sorted((folder / "weeks").glob("*.pkl")):
            # Each cache was hash-verified against the completed trial above.
            for sample in pickle.loads(path.read_bytes()):
                if sample["split"] != "validation":
                    continue
                frame = pd.DataFrame(sample["keys"], columns=KEYS)
                frame["role"] = [ROLES[i] for i in sample["role"][sample["player"]]]
                frame["forecast_time"] = pd.cut(
                    sample["time"],
                    [0, 0.5, 1, 2, np.inf],
                    labels=["0–0.5 s", "0.5–1 s", "1–2 s", ">2 s"],
                ).astype(str)
                frame["player_history"] = np.where(
                    sample["static"][sample["player"], 11] > 0, "cold", "known"
                )
                horizon_max = max(horizon_max, int(frame.frame_id.max()))
                metadata.append(frame)
        meta = pd.concat(metadata, ignore_index=True)
        require_keys(meta)
        if not meta.sort_values(KEYS).reset_index(drop=True)[KEYS].equals(challenger[KEYS]):
            raise ValueError("Diagnostic metadata does not cover the exact development set.")
        slices: list[dict[str, Any]] = []
        models: list[dict[str, Any]] = []
        blend = challenger[KEYS].copy()
        blend[["dx", "dy"]] = (
            challenger[["dx", "dy"]].to_numpy() + reference[["dx", "dy"]].to_numpy()
        ) / 2
        for name, errors in (
            ("existing_tree", reference),
            ("temporal_attention", challenger),
            ("equal_blend_exploratory", blend),
        ):
            models.append({"model": name, **error_metrics(errors)})
            merged = errors[KEYS + ["dx", "dy"]].merge(meta, on=KEYS, validate="one_to_one")
            for dimension in ("role", "forecast_time", "player_history"):
                for value, group in merged.groupby(dimension, sort=True):
                    slices.append(
                        {
                            "model": name,
                            "dimension": dimension,
                            "value": str(value),
                            "rows": len(group),
                            **error_metrics(group),
                        }
                    )
        bootstrap = bootstrap_scores(challenger) - bootstrap_scores(reference)
        score = models[1]["coordinate_rmse_yards"]
        plan = json.loads((folder / "plan.json").read_text())
        report = {
            "status": "passed",
            "run_id": run.run_id,
            "trial_report_sha256": sha256(folder / "summary.json"),
            "trial_source_signature": trial["source_signature"],
            "reference_sha256": REFERENCE_SHA256,
            "models": models,
            "relative_rmse_reduction": 1 - score / REFERENCE_RMSE,
            "paired_game_bootstrap_delta_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "bootstrap": {"resamples": 2000, "seed": 2026, "cluster": "game", "games": 32},
            "ensemble_probe": {
                "status": "exploratory_development_only",
                "weights": {"existing_tree": 0.5, "temporal_attention": 0.5},
                "declaration": (
                    "One equal blend declared during the temporal run, before its final score. "
                    "No blend weight fitting; not part of the primary single-model hypothesis."
                ),
                "paired_game_bootstrap_delta_ci95": np.quantile(
                    bootstrap_scores(blend) - bootstrap_scores(reference), [0.025, 0.975]
                ).tolist(),
                "promotion": "Requires training-side out-of-fold confirmation.",
            },
            "development_rows": len(challenger),
            "maximum_requested_frame": horizon_max,
            "slices": slices,
            "curve": trial["curve"],
            "numerical_sources": {
                k: v for k, v in plan["inputs"].items() if k.startswith(("src/", "scripts/"))
            },
            "caveats": [
                "One predeclared model and seed on an already-inspected development set.",
                "Architecture and representation changed together; feature gains are not isolated.",
                "Final epoch EMA chosen in advance; no best-development checkpoint selection.",
                "Chronological inner folds and organizer gateway validation remain required.",
                "No new holdout evaluation or Kaggle submission; private score remains 0.70090.",
            ],
        }
        atomic_json(folder / "comparison.json", report)
        draw(report, folder)
        if args.publish:
            atomic_json(ROOT / "docs/results/temporal_evaluation.json", report)
            atomic_bytes(
                ROOT / "docs/results/temporal_diagnostics.png",
                (folder / "diagnostics.png").read_bytes(),
            )
        run.event(
            "comparison_complete",
            coordinate_rmse_yards=score,
            paired_ci95=report["paired_game_bootstrap_delta_ci95"],
        )


if __name__ == "__main__":
    main()
