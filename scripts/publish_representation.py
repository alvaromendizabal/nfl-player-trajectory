"""Verify private fitted evidence, render comparisons, and seal public reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_trajectory.representation_evidence import load_representation  # noqa: E402
from nfl_trajectory.runtime import Run, atomic_json, sha256  # noqa: E402


def main() -> None:
    with Run(ROOT, "publish-representation") as run:
        folder = ROOT / "docs/results"
        report = json.loads((folder / "motion_representation.json").read_text())
        for name, digest in {
            **report["sources"],
            **report["inputs"],
            **report["artifacts"],
        }.items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Representation evidence changed: " + name)
        audit = json.loads((folder / "role_contract_audit.json").read_text())
        for name, digest in {**audit["sources"], **audit["input_hashes"]}.items():
            if sha256(ROOT / name) != digest:
                raise ValueError("Role audit evidence changed: " + name)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        labels = ["Fold 1", "Fold 2", "Fold 3", "Pooled"]
        scores = [f["scores"] for f in report["folds"]] + [report["pooled"]]
        for i, (name, label, color) in enumerate(
            [
                ("attention", "Frozen attention", "#a07b38"),
                ("control", "Matched continuation", "#697587"),
                ("smoothed_state", "Motion-state inputs", "#2363c6"),
            ]
        ):
            axes[0].bar(
                np.arange(4) + (i - 1) * 0.25,
                [v[name]["coordinate_rmse_yards"] for v in scores],
                width=0.24,
                label=label,
                color=color,
            )
        axes[0].set(
            xticks=np.arange(4),
            xticklabels=labels,
            ylabel="Coordinate RMSE (yards)",
            title="Same later games; equal continuation budget",
        )
        axes[0].legend(fontsize=8, loc="upper left")
        deltas = [
            s["smoothed_state"]["coordinate_rmse_yards"] - s["control"]["coordinate_rmse_yards"]
            for s in report["horizon_slices"]
        ]
        axes[1].barh(["0–0.5 s", "0.5–1 s", "1–2 s", "Above 2 s"], deltas, color="#2363c6")
        axes[1].axvline(0, color="#202a39", linewidth=1)
        axes[1].set(
            xlabel="Motion minus control RMSE (yards)",
            title="Negative values favor added motion inputs",
        )
        fig.savefig(folder / "motion_representation.png", dpi=160)
        plt.close(fig)
        paths = [
            *report["sources"],
            *audit["sources"],
            "docs/results/motion_representation.json",
            "docs/results/motion_representation.png",
            "docs/results/role_contract_audit.json",
            "docs/REPRESENTATION_RESEARCH.md",
            "src/nfl_trajectory/representation_evidence.py",
            "scripts/publish_representation.py",
        ]
        recovery = folder / "representation_recovery.json"
        if not recovery.exists():
            raise ValueError("The actual completed-fit reuse check is required before publication.")
        recovered = json.loads(recovery.read_text())
        if (
            recovered["status"] != "passed"
            or recovered["fits_reused"] != 6
            or recovered["additional_training_epochs"] != 0
            or recovered["files_verified_unchanged"] != 24
            or not recovered["content_and_modification_time_unchanged"]
        ):
            raise ValueError("The completed-fit recovery check did not pass.")
        for name, state in recovered["artifact_states"].items():
            if sha256(ROOT / name) != state["sha256"]:
                raise ValueError("Recovered fitted artifact changed before publication: " + name)
        paths.append("docs/results/representation_recovery.json")
        atomic_json(
            folder / "representation_manifest.json",
            {
                "status": "passed",
                "run_id": run.run_id,
                "files": {p: sha256(ROOT / p) for p in sorted(set(paths))},
                "private_artifacts_verified": len(report["artifacts"]),
                "feature_completion_gate": "open",
            },
        )
        load_representation(ROOT)
        run.event(
            "representation_published", fitted_models=6, private_artifacts=len(report["artifacts"])
        )


if __name__ == "__main__":
    main()
