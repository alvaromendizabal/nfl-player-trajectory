"""Verify published domain research before a notebook presents its conclusions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from nfl_trajectory.runtime import sha256


def verify_pooled(report: dict[str, Any], row_counts: dict[str, int]) -> None:
    if len(report["folds"]) != 3 or set(row_counts) != {f["fold"] for f in report["folds"]}:
        raise ValueError("Domain research requires the same three complete folds.")
    for arm, metrics in report["pooled"].items():
        expected = math.sqrt(
            sum(
                row_counts[f["fold"]] * f["scores"][arm]["coordinate_rmse_yards"] ** 2
                for f in report["folds"]
            )
            / sum(row_counts.values())
        )
        if not math.isclose(expected, metrics["coordinate_rmse_yards"], abs_tol=1e-7):
            raise ValueError("Published pooled RMSE does not equal row-weighted squared error.")


def load_domain_research(root: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    folder = root / "docs/results"
    manifest = folder / "domain_manifest.json"
    if not manifest.is_file():
        if any(
            (folder / name).exists() for name in ("domain_research.json", "motion_research.json")
        ):
            raise ValueError("Domain research publication is incomplete: missing manifest.")
        return None
    receipt = json.loads(manifest.read_text())
    for name, digest in receipt["files"].items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("Published domain evidence changed: " + name)
    reports = [
        json.loads((folder / name).read_text())
        for name in ("domain_research.json", "motion_research.json")
    ]
    domain, motion = reports
    for report in reports:
        if (
            report["status"] != "completed"
            or report["fits"] != 30
            or report["feature_completion_gate"] != "open"
        ):
            raise ValueError("Domain research completion state is inconsistent.")
        for name, digest in report["sources"].items():
            if sha256(root / name) != digest:
                raise ValueError("Domain research source differs from the executed study: " + name)
    if motion["parent_signature"] != domain["signature"]:
        raise ValueError("Motion follow-up uses a different parent feature study.")
    counts = {f["fold"]: f["rows"] for f in domain["folds"]}
    if sum(counts.values()) != domain["rows"] or domain["rows"] != motion["rows"]:
        raise ValueError("Domain studies have different evaluation row counts.")
    for report in reports:
        verify_pooled(report, counts)
    return domain, motion
