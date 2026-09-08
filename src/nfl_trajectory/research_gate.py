"""An auditable stopping decision from verified feature experiments, never test labels."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage

TOLERANCES = {
    "minimum_feature_gain_each_fold": 0.05,
    "maximum_tail_gain_pooled": 0.005,
    "maximum_tail_gain_any_fold": 0.01,
    "maximum_metadata_omission_cost": 0.01,
    "maximum_positional_fallback_cost": 0.05,
}
POLICY_COMMIT = "87f7d5479c85203d0f9634190637e9e0e989ef37"


def seasons_from_audit(audit: dict[str, Any], games: list[int]) -> set[int]:
    """Use the organizer's season filename; January's calendar year is misleading."""
    mapping = {}
    for pair in audit["pairs"]:
        match = re.fullmatch(r"input_(\d{4})_w\d{2}\.csv", pair["file"])
        if match is None:
            raise ValueError("The audited filename does not identify an NFL season.")
        for game in pair["games"]:
            mapping[game] = int(match[1])
    return {mapping[game] for game in games}


def closure_checks(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the tolerances recorded before the full-pool result was reviewed."""
    measurements = [
        *evidence["feature_gains"],
        *evidence["tail_gains"],
        evidence["pooled_tail_gain"],
        evidence["metadata_cost"],
        evidence["positional_cost"],
    ]
    if not np.isfinite(measurements).all() or len(evidence["feature_gains"]) != 3:
        raise ValueError("The stopping decision requires finite evidence from three inner folds.")
    checks = {
        "every_catalog_candidate_screened": evidence["all_screened"],
        "complete_eligible_pool_explored": evidence["complete_pool"],
        "available_labelled_seasons_represented": evidence["scope_verified"],
        "fixed_estimator_gain_in_every_inner_fold": min(evidence["feature_gains"])
        >= TOLERANCES["minimum_feature_gain_each_fold"],
        "small_final_width_gain": evidence["pooled_tail_gain"]
        < TOLERANCES["maximum_tail_gain_pooled"]
        and max(evidence["tail_gains"]) < TOLERANCES["maximum_tail_gain_any_fold"],
        "metadata_independence": evidence["metadata_cost"]
        <= TOLERANCES["maximum_metadata_omission_cost"],
        "positional_fallback_remains_strong": evidence["positional_cost"]
        <= TOLERANCES["maximum_positional_fallback_cost"],
        "family_removals_and_permutations": evidence["family_evidence"],
        "latest_tree_raw_replay_and_input_stress": evidence["raw_verified"],
        "latest_tree_organizer_gateway": evidence["gateway_verified"],
        "reserved_holdout_unscored": evidence["holdout_unscored"],
    }
    return [{"criterion": name, "passed": bool(value)} for name, value in checks.items()]


def review(root: Path, run: Run) -> dict[str, Any]:
    from nfl_trajectory.benchmark import bootstrap_scores, error_metrics
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.research_evidence import EXTRA_REPORTS, FOLDS, extended_evidence
    from nfl_trajectory.research_inference import research_bundle

    # Recompute reported metrics and verify all underlying fit/evaluation receipts.
    hashes = extended_evidence(root, include_gate=False)
    reports = {
        name: json.loads((root / "artifacts" / original).read_text())
        for name, original in EXTRA_REPORTS.items()
        if name in hashes and name.endswith(".json")
    }
    attribution = reports["feature_attribution.json"]
    tree, inference, gateway = (
        reports[name]
        for name in ("feature_tree.json", "feature_inference.json", "feature_gateway.json")
    )
    bundle = research_bundle(root)
    inventory_path = root / "artifacts/data_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    audit_path = root / "artifacts/audit_summary.json"
    research_seasons = seasons_from_audit(
        json.loads(audit_path.read_text()), bundle["training_games"] + bundle["evaluation_games"]
    )
    labelled_seasons = sorted(
        {
            int(match[1])
            for item in inventory
            if (match := re.fullmatch(r"train/output_(\d{4})_w\d{2}\.csv", item["name"]))
        }
    )
    catalog = pd.concat(
        [
            research_catalog().assign(stage="research"),
            context_catalog().assign(
                stage="context",
                provenance="Verified observed tracking and supplied player metadata; no outputs.",
            ),
            representation_catalog().assign(stage="representation"),
        ],
        ignore_index=True,
    )
    for column in ("feature", "family", "rationale", "availability", "provenance"):
        if catalog[column].isna().any():
            raise ValueError("Every candidate requires documented rationale and provenance.")
    if catalog.feature.duplicated().any():
        raise ValueError("Feature catalog contains duplicate names.")
    catalog["leakage_guard"] = np.where(
        catalog.family.eq("player_history"),
        "Exclude the complete current date; freeze training lookup during evaluation.",
        np.where(
            catalog.family.eq("route_representation"),
            "Fit components/prototypes on fold training inputs only; no outcome labels.",
            "Use observed input frames and supplied context; future player coordinates excluded.",
        ),
    )
    folds = [*attribution["inner_folds"], attribution["development"]]
    coverage, gains, tails, weights = [], [], [], []
    previous_scores, last_scores = [], []
    for name, result in zip(FOLDS, folds, strict=True):
        if result["fold"] != name:
            raise ValueError("Attribution fold ordering differs from the frozen protocol.")
        folder = root / "artifacts/feature_budget" / name
        plan = json.loads((folder / "plan.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        screens = pd.concat(
            [
                pd.read_csv(
                    root
                    / "artifacts"
                    / bank
                    / name
                    / ("conditional.csv" if bank == "research" else "screening.csv")
                )
                for bank in ("research", "context", "representation")
            ],
            ignore_index=True,
        )
        eligible = set(screens.loc[screens.screen_status.eq("eligible"), "feature"])
        coverage.append(
            {
                "fold": name,
                "catalog_candidates": len(catalog),
                "screened_candidates": len(screens),
                "eligible_candidates": len(eligible),
                "pool_candidates": len(plan["pool"]),
                "retained_after_redundancy": summary["retained_after_redundancy"],
                "all_screened": len(screens) == len(catalog)
                and set(screens.feature) == set(catalog.feature),
                "complete_pool": eligible <= set(plan["pool"])
                and max(plan["budgets"]) >= summary["retained_after_redundancy"],
            }
        )
        if name == "development":
            continue
        probe = json.loads((root / "artifacts/nonlinear_probe" / name / "summary.json").read_text())
        baseline = next(r for r in probe["models"] if r["model"] == "landing_features")
        gains.append(1 - result["coordinate_rmse_yards"] / baseline["coordinate_rmse_yards"])
        last = summary["models"][-1]["coordinate_rmse_yards"]
        previous = summary["models"][-2]["coordinate_rmse_yards"]
        tails.append(max(0.0, 1 - last / previous))
        previous_scores.append(previous)
        last_scores.append(last)
        weights.append(result["rows"])

    def pooled(values: list[float]) -> float:
        return float(np.sqrt(np.average(np.square(values), weights=weights)))

    reference = pooled([r["coordinate_rmse_yards"] for r in folds[:3]])
    metadata = pooled(
        [
            next(
                r["coordinate_rmse_yards"]
                for r in f["omissions"]
                if r["model"] == "without_metadata"
            )
            for f in folds[:3]
        ]
    )
    positional = pooled(
        [
            next(
                r["coordinate_rmse_yards"]
                for r in f["omissions"]
                if r["model"] == "without_optional_inputs"
            )
            for f in folds[:3]
        ]
    )
    expected_scenarios = {
        "complete_inputs",
        "missing_metadata",
        "missing_telemetry",
        "cold_player_history",
    }
    raw_scores = {r["scenario"]: r["coordinate_rmse_yards"] for r in inference["scenarios"]}
    raw_robust = set(raw_scores) == expected_scenarios and bool(
        np.isfinite(list(raw_scores.values())).all()
    )
    if raw_robust:
        reference_raw = raw_scores["complete_inputs"]
        raw_robust = all(
            raw_scores[name] <= reference_raw * (1 + limit)
            for name, limit in (
                ("missing_metadata", TOLERANCES["maximum_metadata_omission_cost"]),
                ("cold_player_history", TOLERANCES["maximum_metadata_omission_cost"]),
                ("missing_telemetry", TOLERANCES["maximum_positional_fallback_cost"]),
            )
        )
    evidence = {
        "all_screened": all(r["all_screened"] for r in coverage),
        "complete_pool": all(r["complete_pool"] for r in coverage),
        "scope_verified": bool(labelled_seasons) and set(labelled_seasons) <= research_seasons,
        "feature_gains": gains,
        "tail_gains": tails,
        "pooled_tail_gain": max(0.0, 1 - pooled(last_scores) / pooled(previous_scores)),
        "metadata_cost": metadata / reference - 1,
        "positional_cost": positional / reference - 1,
        "family_evidence": all(len(f["permutation"]) >= 15 for f in folds)
        and all(len(f["models"]) >= 8 for f in reports["feature_ablation.json"]["inner_folds"]),
        "raw_verified": inference["status"] == "passed"
        and raw_robust
        and inference["selected_stage"] == "fixed_tree"
        and inference["selected_model"] == tree["selected_model"]
        and {r["scenario"] for r in inference["scenarios"]} == expected_scenarios
        and all(r["rows"] == folds[-1]["rows"] for r in inference["scenarios"]),
        "gateway_verified": gateway["official_gateway_status"] == "passed"
        and gateway["selected_model"] == tree["selected_model"]
        and gateway["sample_rows"] > 0
        and gateway["plays"] > 0,
        "holdout_unscored": all(
            r.get("holdout_evaluation") == "not_run"
            for r in (attribution, tree, inference, gateway)
        ),
    }
    checks = closure_checks(evidence)
    status = "closed" if all(c["passed"] for c in checks) else "open"
    inputs = {"artifacts/" + EXTRA_REPORTS[name]: value for name, value in hashes.items()}
    inputs["src/nfl_trajectory/research_gate.py"] = sha256(Path(__file__))
    inputs["artifacts/data_inventory.json"] = sha256(inventory_path)
    inputs["artifacts/audit_summary.json"] = sha256(audit_path)
    variant = tree["provenance"]["selected_variant"]
    used_sets = {
        item["fold"]: set(item["used_features"])
        for item in tree["feature_use"]
        if item["variant"] == variant and item["fold"] != "development"
    }
    if set(used_sets) != set(FOLDS[:3]):
        raise ValueError("Actual tree feature use is required for every inner fold.")
    common_used = sorted(set.intersection(*used_sets.values()))
    used_stability = {
        "used_in_all_inner_fits": common_used,
        "count_used_in_all_inner_fits": len(common_used),
        "pairwise_jaccard": [
            {
                "first": first,
                "second": second,
                "jaccard": len(used_sets[first] & used_sets[second])
                / len(used_sets[first] | used_sets[second]),
            }
            for i, first in enumerate(FOLDS[:3])
            for second in FOLDS[i + 1 : 3]
        ],
        "caveat": "Shared definitions; learned route component axes can rotate between fits.",
    }
    error_path = root / "artifacts/feature_attribution/development" / (variant + ".csv")
    baseline_path = root / "artifacts/nonlinear_probe/development/landing_features.csv"
    inputs[str(error_path.relative_to(root))] = sha256(error_path)
    inputs[str(baseline_path.relative_to(root))] = sha256(baseline_path)
    provenance = {
        "inputs": inputs,
        "source_signatures": bundle["source_signatures"],
        "bundle_sha256": hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest(),
        "policy_commit": POLICY_COMMIT,
        "tolerances": TOLERANCES,
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    destination = root / "artifacts/research/gate"

    def write_reports() -> None:
        errors = pd.read_csv(error_path)
        baseline = pd.read_csv(baseline_path)
        keys = ["game_id", "play_id", "nfl_id", "frame_id"]
        pd.testing.assert_frame_equal(errors[keys], baseline[keys])
        baseline_rmse = error_metrics(baseline)["coordinate_rmse_yards"]
        current_rmse = error_metrics(errors)["coordinate_rmse_yards"]
        paired_delta = bootstrap_scores(errors) - bootstrap_scores(baseline)
        comparison = {
            "estimator_settings": attribution["provenance"]["settings"],
            "baseline_model": "landing_features",
            "baseline_features": 64,
            "baseline_coordinate_rmse_yards": baseline_rmse,
            "engineered_model": tree["selected_model"],
            "engineered_coordinate_rmse_yards": current_rmse,
            "feature_gain_percent": 100 * (1 - current_rmse / baseline_rmse),
            "paired_delta_ci95_yards": np.quantile(paired_delta, [0.025, 0.975]).tolist(),
            "interpretation": "Fixed estimators and baseline; paired game bootstrap.",
        }
        slices = []
        groups = {
            "role": errors.player_role,
            "requested_horizon": pd.cut(
                errors.num_frames_output,
                [0, 10, 20, 30, np.inf],
                labels=["up to 1 s", "1–2 s", "2–3 s", "over 3 s"],
            ),
        }
        total_sse = float(np.sum(errors[["dx", "dy"]].to_numpy() ** 2))
        for dimension, grouping in groups.items():
            for label, group in errors.groupby(grouping, observed=True, sort=True):
                slices.append(
                    {
                        "dimension": dimension,
                        "value": str(label),
                        "rows": len(group),
                        "games": int(group.game_id.nunique()),
                        **error_metrics(group),
                        "squared_error_share_percent": 100
                        * float(np.sum(group[["dx", "dy"]].to_numpy() ** 2))
                        / total_sse,
                    }
                )
        atomic_json(
            destination / "diagnostics.json",
            {
                "status": "passed",
                "holdout_evaluation": "not_run",
                "source_signature": signature,
                "selected_model": tree["selected_model"],
                "rows": len(errors),
                "metrics": error_metrics(errors),
                "controlled_comparison": comparison,
                "slices": slices,
                "scope": "Inspected development games; descriptive rather than selection evidence.",
            },
        )
        atomic_json(
            destination / "selection_manifest.json",
            {
                "source_signature": signature,
                "source_signatures": bundle["source_signatures"],
                "bundle_sha256": provenance["bundle_sha256"],
                "selected_model": tree["selected_model"],
                "features": bundle["tree"]["features"],
                "screened_feature_count": bundle["tree"]["screened_feature_count"],
                "used_feature_count": bundle["retained_features"],
                "historical_priors_used": [
                    n for n in bundle["tree"]["features"] if n.startswith("prior__")
                ],
                "freeze_status": "frozen_for_final_phase"
                if status == "closed"
                else "research_only",
                "training_games": bundle["training_games"],
                "development_games": bundle["evaluation_games"],
                "holdout_evaluation": "not_run",
            },
        )
        atomic_bytes(destination / "catalog.csv", catalog.to_csv(index=False).encode())
        atomic_json(
            destination / "summary.json",
            {
                "status": "passed",
                "feature_gate": status,
                "final_training_ready": status == "closed",
                "source_signature": signature,
                "source_signatures": bundle["source_signatures"],
                "provenance": provenance,
                "checks": checks,
                "measurements": evidence,
                "coverage": coverage,
                "controlled_comparison": comparison,
                "data_scope": {
                    "competition": "nfl-big-data-bowl-2026-prediction",
                    "verified_inventory_files": len(inventory),
                    "inventory_sha256": sha256(inventory_path),
                    "labelled_seasons": labelled_seasons,
                    "research_seasons": sorted(research_seasons),
                    "sample_calendar_years": gateway["sample_calendar_years"],
                    "across_season_accuracy": "not_established",
                },
                "families": int(catalog.family.nunique()),
                "selected_model": tree["selected_model"],
                "retained_features": bundle["retained_features"],
                "stable_screened_columns": len(attribution["selected_in_all_inner_folds"]),
                "used_feature_stability": used_stability,
                "holdout_evaluation": "not_run",
                "final_model": False,
                "scope": "Documented, data-supported avenues; no global optimum claim.",
                "remaining": [
                    "External ratings/coaching need verified timing and frame-level value.",
                    "Across-season accuracy needs another labelled season.",
                    "Final refitting and reserved holdout evaluation follow this decision.",
                ],
            },
        )

    outputs = [
        destination / name
        for name in ("summary.json", "selection_manifest.json", "diagnostics.json", "catalog.csv")
    ]
    stage(root, "feature-gate-review", signature, outputs, write_reports, run)
    return dict(json.loads((destination / "summary.json").read_text()))
