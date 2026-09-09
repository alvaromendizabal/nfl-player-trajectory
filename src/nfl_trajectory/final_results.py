"""Verify final-result lineage for reproducible public notebooks and private reruns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nfl_trajectory.final_protocol import digest
from nfl_trajectory.runtime import atomic_json, sha256

REPORTS = {
    "fit": "final_fit.json",
    "inference": "final_inference.json",
    "seal": "final_seal.json",
    "evaluation": "final_evaluation.json",
}


def validate_lineage(reports: dict[str, Any]) -> None:
    fit, inference, seal, evaluation = (reports[k] for k in REPORTS)
    if (
        fit.get("status") != "passed"
        or fit.get("final_fit") is not True
        or inference.get("status") != "passed"
        or evaluation.get("status") != "passed"
        or evaluation.get("holdout_evaluation") != "completed"
        or evaluation.get("primary_scenario") != "complete"
        or evaluation.get("seal") != seal.get("source_signature")
        or evaluation.get("fit_source") != fit.get("source_signature")
        or inference.get("fit_source") != fit.get("source_signature")
        or evaluation.get("protocol_source") != inference.get("protocol_source")
        or evaluation.get("protocol_source") != fit["provenance"]["protocol_source"]
        or inference["provenance"]["bundle_sha256"] != seal["provenance"]["bundle_sha256"]
        or evaluation.get("rows") != seal.get("rows")
        or evaluation.get("games") != len(seal["games"])
        or evaluation.get("outcome_hashes") != seal["provenance"]["outcome_hashes"]
    ):
        raise ValueError("Final notebook results mix incomplete or incompatible model lineages.")
    if seal["source_signature"] != digest(
        {k: v for k, v in seal.items() if k != "source_signature"}
    ):
        raise ValueError("The published prediction seal is inconsistent.")


def publish_final_results(root: Path) -> dict[str, Any]:
    """Publish only after verifying private sealed predictions and the scored-stage receipts."""
    from nfl_trajectory.final_evaluation import load_seal
    from nfl_trajectory.research_evidence import verified_checkpoint

    folder = root / "artifacts/final"
    seal = load_seal(root)
    for name in ("summary.json", "errors.csv"):
        verified_checkpoint(
            root, "final-holdout-evaluation", seal["source_signature"], folder / "evaluation" / name
        )
    reports = {
        "fit": json.loads((folder / "models/summary.json").read_text()),
        "inference": json.loads((folder / "inference/summary.json").read_text()),
        "seal": seal,
        "evaluation": json.loads((folder / "evaluation/summary.json").read_text()),
    }
    validate_lineage(reports)
    public = root / "docs/results"
    for kind, name in REPORTS.items():
        atomic_json(public / name, reports[kind])
    inputs = {"docs/results/" + name: sha256(public / name) for name in REPORTS.values()}
    for name in ("final_results", "final_evaluation", "final_inference"):
        relative = "src/nfl_trajectory/" + name + ".py"
        inputs[relative] = sha256(root / relative)
    payload = {
        "status": "passed",
        "files": inputs,
        "seal": seal["source_signature"],
        "holdout_evaluation": "completed",
    }
    atomic_json(
        public / "final_results_manifest.json", {**payload, "source_signature": digest(payload)}
    )
    return reports


def load_final_results(root: Path) -> dict[str, Any] | None:
    """A clean Git checkout verifies the exact public reports; local artifacts cannot mask drift."""
    public = root / "docs/results"
    path = public / "final_results_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest["source_signature"] != digest(
        {k: v for k, v in manifest.items() if k != "source_signature"}
    ):
        raise ValueError("Final publication manifest has inconsistent provenance.")
    expected = {"docs/results/" + name for name in REPORTS.values()} | {
        "src/nfl_trajectory/" + name + ".py"
        for name in ("final_results", "final_evaluation", "final_inference")
    }
    if set(manifest["files"]) != expected:
        raise ValueError("Final publication manifest is incomplete.")
    for relative, value in manifest["files"].items():
        if sha256(root / relative) != value:
            raise ValueError("Final published source or report changed: " + relative)
    reports = {kind: json.loads((public / name).read_text()) for kind, name in REPORTS.items()}
    validate_lineage(reports)
    if reports["seal"]["source_signature"] != manifest["seal"]:
        raise ValueError("Final publication references a different prediction seal.")
    local = root / "artifacts/final/evaluation/summary.json"
    if local.exists() and json.loads(local.read_text()) != reports["evaluation"]:
        raise ValueError("Local final evaluation differs from the published result.")
    return reports
