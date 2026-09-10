"""Seal public research reports only after their private artifacts pass verification."""

from __future__ import annotations

import json
from pathlib import Path

from nfl_trajectory.domain_evidence import load_domain_research
from nfl_trajectory.runtime import Run, atomic_json, sha256


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with Run(root, "domain-publication") as run:
        names = {
            "docs/DOMAIN_RESEARCH.md",
            "docs/DOMAIN_EXPERIMENT.md",
            "docs/MOTION_EXPERIMENT.md",
            "docs/results/domain_research.json",
            "docs/results/domain_research.png",
            "docs/results/domain_feature_catalog.csv",
            "docs/results/motion_research.json",
            "docs/results/domain_research_recovery.json",
            "docs/results/motion_research_recovery.json",
            "docs/results/domain_reproduction.json",
            "docs/results/motion_training_audit.json",
            "src/nfl_trajectory/domain_evidence.py",
            "scripts/audit_motion.py",
            "scripts/publish_domain_research.py",
        }
        artifacts: dict[str, str] = {}
        for study in ("domain_research", "motion_research"):
            report = json.loads((root / "docs/results" / (study + ".json")).read_text())
            if report["status"] != "completed" or report["fits"] != 30:
                raise ValueError(
                    "All declared feature comparisons must complete before publication."
                )
            for name, digest in {**report["artifacts"], **report["sources"]}.items():
                if sha256(root / name) != digest:
                    raise ValueError("Executed domain artifact or source changed: " + name)
                artifacts[name] = digest
            names.update(report["sources"])
            recovery = json.loads((root / "docs/results" / (study + "_recovery.json")).read_text())
            if (
                recovery["status"] != "passed"
                or recovery["fits_reused"] != 30
                or recovery["new_training_epochs"] != 0
            ):
                raise ValueError("The completed feature fits must pass the restart check.")
        atomic_json(
            root / "docs/results/domain_manifest.json",
            {
                "status": "passed",
                "run_id": run.run_id,
                "files": {p: sha256(root / p) for p in sorted(names)},
                "private_artifact_count": len(artifacts),
                "unique_scientific_fits": 60,
                "reproduction_fits": 30,
                "feature_completion_gate": "open",
            },
        )
        if load_domain_research(root) is None:
            raise ValueError("Published domain research could not be loaded.")
        run.event("domain_publication_verified", unique_fits=60, private_artifacts=len(artifacts))


if __name__ == "__main__":
    main()
