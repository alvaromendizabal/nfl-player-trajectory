"""Check the published end-to-end feature evidence before notebook presentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nfl_trajectory.domain_evidence import verify_pooled
from nfl_trajectory.runtime import sha256


def load_representation(root: Path) -> dict[str, Any] | None:
    folder = root / "docs/results"
    manifest = folder / "representation_manifest.json"
    if not manifest.exists():
        if (folder / "motion_representation.json").exists():
            raise ValueError("Representation publication is missing its verified manifest.")
        return None
    receipt = json.loads(manifest.read_text())
    if receipt["status"] != "passed":
        raise ValueError("Representation publication has not passed validation.")
    for name, digest in receipt["files"].items():
        if sha256(root / name) != digest:
            raise ValueError("Published representation evidence changed: " + name)
    report = json.loads((folder / "motion_representation.json").read_text())
    if report["status"] != "completed" or report["fits"] != 6:
        raise ValueError("The matched representation study is incomplete.")
    if report["feature_completion_gate"] != "open":
        raise ValueError("This bounded experiment cannot close feature research.")
    for name, digest in report["sources"].items():
        if sha256(root / name) != digest:
            raise ValueError("Representation numerical source changed: " + name)
    for fold in report["folds"]:
        arms = fold["fits"]
        if arms["control"]["parameters"] != arms["smoothed_state"]["parameters"]:
            raise ValueError("The feature and control arms have different parameter counts.")
        if any(arm["epochs"] != report["settings"]["epochs"] for arm in arms.values()):
            raise ValueError("A feature arm did not complete the matched training budget.")
    counts = {f["fold"]: f["rows"] for f in report["folds"]}
    if sum(counts.values()) != report["rows"]:
        raise ValueError("Published representation row counts disagree.")
    verify_pooled(report, counts)
    return report
