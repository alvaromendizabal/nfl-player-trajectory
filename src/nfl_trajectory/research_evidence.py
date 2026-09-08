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
ARCHIVED_INPUT_FAILURE_SHA256 = "68b503bfb1727e0ac00687feed1a1992bb02683895b510516c7eb9dda7fea1c1"
EXTRA_REPORTS = {
    "feature_probe.json": "nonlinear_probe/summary.json",
    "feature_ablation.json": "feature_ablation/summary.json",
    "feature_budget.json": "feature_budget/summary.json",
    "feature_inference.json": "research/inference/summary.json",
    "feature_input_failure.json": "research/inference_before_fallback.json",
    "feature_joint.json": "joint_linear/summary.json",
    "feature_attribution.json": "feature_attribution/summary.json",
    "feature_attribution.png": "feature_attribution/figure.png",
    "feature_gateway.json": "research/gateway/summary.json",
    "feature_tree.json": "research/tree/summary.json",
    "feature_wide_ablation.json": "wide_ablation/summary.json",
    "feature_gate.json": "research/gate/summary.json",
    "feature_freeze.json": "research/gate/selection_manifest.json",
    "feature_diagnostics.json": "research/gate/diagnostics.json",
    "feature_provenance.csv": "research/gate/catalog.csv",
}
GATE_REPORTS = {name for name, path in EXTRA_REPORTS.items() if path.startswith("research/gate/")}


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


def wide_group_evidence(
    root: Path, sources: dict[str, str], bundle_hash: str
) -> dict[str, Any] | None:
    """Verify every wide refit and recompute its reported error before publication."""
    from nfl_trajectory.runtime import Run, stage
    from nfl_trajectory.wide_ablation import GROUPS

    folder = root / "artifacts/wide_ablation"
    if not folder.exists():
        return None
    results: list[dict[str, Any]] = []
    paths = [Path(__file__), root / "src/nfl_trajectory/wide_ablation.py"]
    for fold in FOLDS:
        path = folder / fold / "summary.json"
        report = read(path)
        provenance = report["provenance"]
        source = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        if (
            source != report["source_signature"]
            or report["source_signatures"] != sources
            or provenance["bundle_sha256"] != bundle_hash
            or report["fold"] != fold
            or report["holdout_evaluation"] != "not_run"
            or provenance["groups"] != {name: sorted(group) for name, group in GROUPS.items()}
        ):
            raise ValueError("Wide group refits do not cover the current representation.")
        for relative, expected in provenance["inputs"].items():
            if sha256(root / relative) != expected:
                raise ValueError("Wide group refit source or inputs changed.")
        verified_checkpoint(root, "wide-group-" + fold, source, path)
        reference = root / "artifacts/feature_attribution" / fold / (report["profile"] + ".csv")
        verify_error_metric(reference, report["reference"])
        if {row["group"] for row in report["models"]} != set(GROUPS):
            raise ValueError("A predeclared wide ablation group is missing.")
        for row in report["models"]:
            if row["refitted"]:
                for phase, extension in (("fit", ".pkl"), ("evaluate", ".csv")):
                    output = folder / fold / (row["model"] + extension)
                    verified_checkpoint(
                        root, f"wide-group-{fold}-{row['group']}-{phase}", source, output
                    )
                verify_error_metric(folder / fold / (row["model"] + ".csv"), row)
            elif (
                row["removed_features"] != 0
                or row["coordinate_rmse_yards"] != report["reference"]["coordinate_rmse_yards"]
            ):
                raise ValueError("An absent-group control cannot claim a refitted improvement.")
        paths.append(path)
        results.append(report)
    weights = [r["rows"] for r in results[:3]]
    reference_rmse = float(
        np.sqrt(
            np.average(
                [r["reference"]["coordinate_rmse_yards"] ** 2 for r in results[:3]], weights=weights
            )
        )
    )
    scores = pooled_scores(results[:3], "rows")
    comparison = []
    for group in GROUPS:
        name = "without_" + group
        changes = [
            next(row["coordinate_rmse_yards"] for row in report["models"] if row["model"] == name)
            / report["reference"]["coordinate_rmse_yards"]
            - 1
            for report in results[:3]
        ]
        comparison.append(
            {
                "group": group,
                "pooled_coordinate_rmse_yards": scores[name],
                "pooled_rmse_change": scores[name] - reference_rmse,
                "pooled_relative_gain": 1 - scores[name] / reference_rmse,
                "inner_relative_costs": changes,
            }
        )
    provenance = {
        "inputs": {str(path.relative_to(root)): sha256(path) for path in paths},
        "source_signatures": sources,
        "bundle_sha256": bundle_hash,
        "groups": {name: sorted(group) for name, group in GROUPS.items()},
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    summary = {
        "status": "passed",
        "source_signature": signature,
        "source_signatures": sources,
        "provenance": provenance,
        "selected_model": results[-1]["selected_model"],
        "profile": results[-1]["profile"],
        "inner_folds": results[:3],
        "development": results[-1],
        "pooled_reference_rmse_yards": reference_rmse,
        "pooled_comparisons": comparison,
        "covered_families": sorted(set().union(*GROUPS.values())),
        "holdout_evaluation": "not_run",
        "final_model": False,
    }
    output = folder / "summary.json"
    with Run(root, "wide-group-report") as run:
        stage(
            root,
            "wide-group-report",
            signature,
            [output],
            lambda: atomic_json(output, summary),
            run,
        )
    return read(output)


def extended_evidence(root: Path, *, include_gate: bool = True) -> dict[str, str]:
    """Recompute metrics from verified errors; do not publish unverified JSON summaries."""
    from nfl_trajectory.research import feature_research_snapshot
    from nfl_trajectory.research_inference import research_bundle

    if not (root / "artifacts/nonlinear_probe/plan.json").exists():
        return {}
    archived = root / "artifacts/research/inference_before_fallback.json"
    if archived.exists() and sha256(archived) != ARCHIVED_INPUT_FAILURE_SHA256:
        raise ValueError("The archived input failure differs from the original recorded result.")
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
    wide_group_evidence(root, sources, bundle_hash)
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
    optional_reports = [
        ("feature_attribution/summary.json", "feature-attribution-report"),
        ("research/gateway/summary.json", "official-gateway"),
        ("research/tree/summary.json", "research-tree-bundle"),
        ("wide_ablation/summary.json", "wide-group-report"),
    ]
    if include_gate:
        optional_reports.append(("research/gate/summary.json", "feature-gate-review"))
    for relative, stage_name in optional_reports:
        output = root / "artifacts" / relative
        if not output.exists():
            continue
        report = read(output)
        provenance = report["provenance"]
        signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        if signature != report["source_signature"] or report["source_signatures"] != sources:
            raise ValueError("Extended attribution or gateway provenance is inconsistent.")
        for path, expected in provenance["inputs"].items():
            if sha256(root / path) != expected:
                raise ValueError("Extended evidence has stale inputs or source code.")
        if "bundle_sha256" in provenance and provenance["bundle_sha256"] != bundle_hash:
            raise ValueError("Gateway evidence uses another inference bundle.")
        verified_checkpoint(root, stage_name, signature, output)
        if stage_name == "feature-attribution-report":
            verified_checkpoint(
                root, stage_name, signature, root / "artifacts/feature_attribution/figure.png"
            )
        if stage_name == "feature-gate-review":
            for name in GATE_REPORTS:
                verified_checkpoint(
                    root, stage_name, signature, root / "artifacts" / EXTRA_REPORTS[name]
                )
    return {
        published: sha256(root / "artifacts" / local)
        for published, local in EXTRA_REPORTS.items()
        if (root / "artifacts" / local).is_file()
        and (include_gate or published not in GATE_REPORTS)
    }
