"""Verify extended experiment lineage before publishing compact aggregate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.runtime import atomic_json, sha256

FOLDS = ("inner_1", "inner_2", "inner_3", "development")
EXTRA_REPORTS = {
    "feature_probe.json": "nonlinear_probe/summary.json",
    "feature_ablation.json": "feature_ablation/summary.json",
    "feature_budget.json": "feature_budget/summary.json",
    "feature_inference.json": "research/inference/summary.json",
    "feature_input_failure.json": "research/inference_before_fallback.json",
    "feature_joint.json": "joint_linear/summary.json",
}


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def verified_checkpoint(root: Path, name: str, signature: str, output: Path) -> None:
    receipt = read(root / ".state" / (name + ".json"))
    if (
        receipt.get("status") != "completed"
        or receipt.get("signature") != signature
        or receipt.get("outputs", {}).get(str(output.relative_to(root))) != sha256(output)
    ):
        raise ValueError(f"Unverified research checkpoint: {name}")


def verified_plan(root: Path, folder: Path, script: str) -> dict[str, Any]:
    plan = read(folder / "plan.json")
    values = {k: v for k, v in plan.items() if k != "source_signature"}
    signature = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    if (
        signature != plan["source_signature"]
        or plan["script"] != sha256(root / "scripts" / script)
        or plan["lock"] != sha256(root / "scripts" / (script + ".lock"))
    ):
        raise ValueError(f"Stale research plan: {folder.name}")
    return plan


def verify_error_metric(path: Path, row: dict[str, Any]) -> None:
    error = pd.read_csv(path, usecols=["dx", "dy"]).to_numpy(float)
    if not np.isfinite(error).all() or not np.isclose(
        np.sqrt(np.mean(error**2)), row["coordinate_rmse_yards"], rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Reported metric differs from its verified frame errors.")


def pooled_scores(results: list[dict[str, Any]], count_key: str) -> dict[str, float]:
    names = [r["model"] for r in results[0]["models"]]
    return {
        name: float(
            np.sqrt(
                sum(
                    next(m["coordinate_rmse_yards"] for m in r["models"] if m["model"] == name) ** 2
                    * r[count_key]
                    for r in results
                )
                / sum(r[count_key] for r in results)
            )
        )
        for name in names
    }


def joint_evidence(root: Path, sources: dict[str, str]) -> dict[str, Any] | None:
    """All joint variants share one estimator; selection uses inner folds exclusively."""
    folder = root / "artifacts/joint_linear"
    if not folder.exists():
        return None
    parent = verified_plan(root, root / "artifacts/nonlinear_probe", "nonlinear_probe.py")
    results = []
    signatures = {}
    for fold in FOLDS:
        plan = verified_plan(root, folder / fold, "joint_feature_fit.py")
        if (
            plan["representation_sources"] != sources
            or plan["parent"] != parent["source_signature"]
            or plan["helper"] != sha256(root / "scripts/ablate_features.py")
            or plan["contracts"] != sha256(root / "src/nfl_trajectory/feature_contracts.py")
        ):
            raise ValueError("Joint linear feature lineage is stale.")
        summary = read(folder / fold / "summary.json")
        if summary["source_signature"] != plan["source_signature"]:
            raise ValueError("Joint model summary is stale.")
        for row in summary["models"]:
            name = row["model"]
            for phase, extension in (("fit", ".json"), ("evaluate", ".csv")):
                verified_checkpoint(
                    root,
                    f"joint-{fold}-{name}-{phase}",
                    plan["source_signature"],
                    folder / fold / (name + extension),
                )
            verify_error_metric(folder / fold / (name + ".csv"), row)
        results.append(summary)
        signatures[fold] = plan["source_signature"]
    scores = pooled_scores(results[:3], "rows")
    return {
        **results[-1],
        "inner_folds": results[:3],
        "inner_scores": scores,
        "selected_model": min(scores, key=lambda n: (scores[n], n)),
        "selection": "pooled chronological inner folds; no development selection",
        "source_signatures": sources,
        "joint_sources": signatures,
        "feature_gate": "open",
        "final_model": False,
    }


def extended_evidence(root: Path) -> dict[str, str]:
    """Recompute metrics from verified errors; do not publish unverified JSON summaries."""
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.research_inference import research_bundle

    if not (root / "artifacts/nonlinear_probe/plan.json").exists():
        return {}
    sources = feature_research_snapshot(root)["source_signatures"]
    parent = root / "artifacts/nonlinear_probe"
    plan = verified_plan(root, parent, "nonlinear_probe.py")
    if plan["representation_sources"] != sources:
        raise ValueError("Nonlinear probe uses stale feature artifacts.")
    probe = read(parent / "summary.json")
    inner = []
    for fold in FOLDS:
        summary = read(parent / fold / "summary.json")
        if summary["source_signature"] != plan["source_signature"]:
            raise ValueError("Nonlinear fold belongs to another experiment.")
        for row in summary["models"]:
            name = row["model"]
            if name == "constant_velocity":
                continue
            for phase, extension in (("fit", ".pkl"), ("evaluate", ".csv")):
                verified_checkpoint(
                    root,
                    f"nonlinear-{fold}-{name}-{phase}",
                    plan["source_signature"],
                    parent / fold / (name + extension),
                )
            verify_error_metric(parent / fold / (name + ".csv"), row)
        if fold != "development":
            inner.append(summary)
        elif summary["models"] != probe["models"]:
            raise ValueError("Public nonlinear summary differs from its development fold.")
    scores = pooled_scores(inner, "validation_rows_per_model")
    scores = {k: v for k, v in scores.items() if k not in ("constant_velocity", "core_plus_noise")}
    if scores != probe["selection"]["inner_scores"] or probe["selection"]["selected_model"] != min(
        scores, key=lambda n: (scores[n], n)
    ):
        raise ValueError("Nonlinear feature choice is not the pooled inner-fold winner.")
    for label, script, prefix in (
        ("feature_ablation", "ablate_features.py", "ablation"),
        ("feature_budget", "feature_budget.py", "budget"),
    ):
        folder = root / "artifacts" / label
        results = []
        for fold in FOLDS:
            current = verified_plan(root, folder / fold, script)
            representation_key = (
                "representations" if prefix == "ablation" else "representation_sources"
            )
            if (
                current["parent"] != plan["source_signature"]
                or current[representation_key] != sources
            ):
                raise ValueError("Extended experiment has stale parent features.")
            if prefix == "ablation" and current["contracts"] != sha256(
                root / "src/nfl_trajectory/feature_contracts.py"
            ):
                raise ValueError("Optional-input contract changed after the fallback fit.")
            if prefix == "budget" and current["helper"] != sha256(
                root / "scripts/ablate_features.py"
            ):
                raise ValueError("Budget helper changed after experiment execution.")
            summary = read(folder / fold / "summary.json")
            if summary["source_signature"] != current["source_signature"]:
                raise ValueError("Extended summary belongs to another plan.")
            for row in summary["models"]:
                name = row["model"]
                if name == "all_engineered":
                    continue
                stage_name = "robust-linear" if name == "robust_linear" else name
                for phase, extension in (
                    ("fit", ".json" if name == "robust_linear" else ".pkl"),
                    ("evaluate", ".csv"),
                ):
                    verified_checkpoint(
                        root,
                        f"{prefix}-{fold}-{stage_name}-{phase}",
                        current["source_signature"],
                        folder / fold / (name + extension),
                    )
                verify_error_metric(folder / fold / (name + ".csv"), row)
            results.append(summary)
        scores = pooled_scores(results[:3], "rows")
        atomic_json(
            folder / "summary.json",
            {
                **results[-1],
                "inner_folds": results[:3],
                "inner_scores": scores,
                "source_signatures": sources,
                "feature_gate": "open",
                "selection": "diagnostic only; no automatic final-model promotion",
            },
        )
    joint = joint_evidence(root, sources)
    if joint is None:
        raise ValueError("The joint linear availability experiment is incomplete.")
    atomic_json(root / "artifacts/joint_linear/summary.json", joint)
    inference = read(root / "artifacts/research/inference/summary.json")
    bundle = research_bundle(root)
    bundle_hash = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    if (
        inference["validator_sha256"] != sha256(root / "scripts/validate_research.py")
        or inference["bundle_sha256"] != bundle_hash
        or inference["source_signatures"] != sources
        or inference["inference_sources"] != bundle["inference_sources"]
    ):
        raise ValueError("Inference evidence does not cover the current model and source.")
    for path in inference["weekly_receipts"]:
        output = root / path
        verified_checkpoint(
            root, "research-inference-" + output.stem, inference["validation_signature"], output
        )
    return {
        published: sha256(root / "artifacts" / local) for published, local in EXTRA_REPORTS.items()
        if (root / "artifacts" / local).is_file()
    }
