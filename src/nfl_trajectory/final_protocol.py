"""Freeze final refit inputs and reporting rules without opening holdout outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock

from nfl_trajectory.feature_contracts import metadata_dependent, telemetry_dependent
from nfl_trajectory.motion import ENTITY, KEYS, require_keys
from nfl_trajectory.research_gate import closure_checks
from nfl_trajectory.runtime import Run, atomic_json, sha256, stage

PROFILES = ("without_metadata", "without_optional_inputs")
EVALUATION = {
    "primary_metric": "coordinate_rmse_yards",
    "formula": "sqrt(sum(dx**2 + dy**2) / (2 * forecast_frames))",
    "aggregation": "Pool coordinate squared errors; never average game or week RMSEs.",
    "confidence_interval": {
        "method": "game_cluster_percentile_bootstrap",
        "resamples": 2000,
        "seed": 2026,
        "quantiles": [0.025, 0.975],
        "scope": "Game sampling uncertainty within this single labelled season.",
    },
    "secondary_metrics": [
        "ade_frame_weighted_yards",
        "ade_trajectory_weighted_yards",
        "fde_trajectory_weighted_yards",
        "p95_displacement_yards",
        "coordinate_mae_yards",
    ],
    "descriptive_slices": ["player_role", "forecast_second", "week"],
    "reference_models": ["constant_velocity", "role_ridge"],
    "availability_checks": ["complete", "without_metadata", "without_telemetry", "cold_history"],
    "decision_rule": "Report the frozen model regardless of score; no holdout model selection.",
    "release_rule": (
        "Seal the protocol, fitted model, inference sources and all keyed predictions before "
        "opening holdout outcomes. Resume only the same sealed evaluation. A failed run does "
        "not authorize a different model; an error correction must retain its audit trail."
    ),
    "gateway": "Verify final package/standalone parity using the unchanged organizer gateway.",
    "submission": "Owner-controlled; never submit automatically.",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def refit_contract(
    gate: dict[str, Any], freeze: dict[str, Any], splits: pd.DataFrame, catalog: pd.DataFrame
) -> dict[str, Any]:
    """A closed gate is necessary; validate its complete schema and original game assignments."""
    checks = closure_checks(gate["measurements"])
    if (
        gate.get("status") != "passed"
        or gate.get("feature_gate") != "closed"
        or gate.get("final_training_ready") is not True
        or gate.get("checks") != checks
        or not all(row["passed"] for row in checks)
        or gate.get("holdout_evaluation") != "not_run"
        or freeze.get("holdout_evaluation") != "not_run"
        or freeze.get("freeze_status") != "frozen_for_final_phase"
    ):
        raise ValueError("Final preparation requires the complete closed, unscored feature gate.")
    if (
        digest(gate["provenance"]) != gate["source_signature"]
        or freeze["source_signature"] != gate["source_signature"]
        or freeze["source_signatures"] != gate["source_signatures"]
        or freeze["bundle_sha256"] != gate["provenance"]["bundle_sha256"]
        or freeze["selected_model"] != gate["selected_model"]
    ):
        raise ValueError("The feature freeze belongs to a different research lineage.")
    if (
        splits.empty
        or splits[["game_id", "game_date", "split"]].isna().any().any()
        or splits.game_id.duplicated().any()
        or set(splits.split) != {"train", "validation", "holdout"}
    ):
        raise ValueError("Final partitions require each audited game exactly once.")
    dates = pd.to_datetime(splits.game_date, errors="raise")
    expected = pd.to_datetime(splits.game_id.astype(str).str[:8], format="%Y%m%d")
    if not dates.equals(expected):
        raise ValueError("Final partition dates disagree with game identifiers.")
    games = {name: sorted(rows.game_id.astype(int)) for name, rows in splits.groupby("split")}
    if (
        freeze["training_games"] != games["train"]
        or freeze["development_games"] != games["validation"]
        or max(games["train"]) // 100 >= min(games["validation"]) // 100
        or max(games["validation"]) // 100 >= min(games["holdout"]) // 100
    ):
        raise ValueError("Refitting must preserve the original strictly chronological partitions.")
    profiles = freeze["availability_profile_features"]
    if set(profiles) != set(PROFILES) or catalog.feature.duplicated().any():
        raise ValueError("Both frozen availability profiles and a unique catalog are required.")
    known = set(catalog.feature)
    for profile in PROFILES:
        names = profiles[profile]
        if not names or len(names) != len(set(names)) or not set(names).issubset(known):
            raise ValueError("Frozen refit columns must be unique, nonempty catalog features.")
        if any(metadata_dependent(name) for name in names) or (
            profile == "without_optional_inputs" and any(telemetry_dependent(n) for n in names)
        ):
            raise ValueError("Frozen columns violate their input availability contract.")
    if profiles[PROFILES[1]] != [
        name for name in profiles[PROFILES[0]] if not telemetry_dependent(name)
    ]:
        raise ValueError(
            "The positional profile must preserve the frozen omission and column order."
        )
    if (
        freeze["screened_features"] != profiles[PROFILES[0]]
        or freeze["screened_feature_count"] != len(profiles[PROFILES[0]])
        or freeze["used_feature_count"] != len(freeze["features"])
        or len(set(freeze["features"])) != len(freeze["features"])
        or not set(freeze["features"]).issubset(profiles[PROFILES[0]])
    ):
        raise ValueError("Refit columns and currently used tree inputs must remain distinct.")
    return {
        "selected_model": freeze["selected_model"],
        "training_games": sorted(games["train"] + games["validation"]),
        "holdout_games": games["holdout"],
        "original_partitions": games,
        "availability_profile_features": profiles,
        "research_active_inputs": freeze["used_feature_count"],
        "estimator": {
            "class": "sklearn.ensemble.HistGradientBoostingRegressor",
            "targets": "Independent x/y residuals about the refitted role-conditioned baseline.",
            "settings": gate["controlled_comparison"]["estimator_settings"],
            "optimization": "Use the validated fixed capacity; no additional parameter search.",
        },
        "preprocessing": {
            "baseline": "Refit on all training_games; never reuse research fitted coefficients.",
            "history": (
                "Recompute raw constant-velocity residual encodings, smoothing 20, from strictly "
                "earlier training dates; exclude the whole current date. Freeze evaluation lookup."
            ),
            "routes": "Refit 16 route components and eight prototypes on training_games only.",
            "columns": "Preserve each frozen ordered schema; no screening or reselection.",
            "export": "Prune after fitting, with full-row original/portable prediction parity.",
        },
        "evaluation": EVALUATION,
    }


def training_inventory(
    root: Path, caches: list[Path], splits: pd.DataFrame, audit: dict[str, Any], run: Run
) -> dict[str, Any]:
    """Read verified cache identifiers, never holdout tracking or outcome coordinates."""
    allowed = splits.loc[~splits.split.eq("holdout")].set_index("game_id").split.to_dict()
    audited = {}
    for pair in audit["pairs"]:
        if set(pair["games"]) & set(allowed):
            if not set(pair["games"]).issubset(allowed):
                raise ValueError(
                    "Final input weeks must not straddle the reserved holdout boundary."
                )
            audited[Path(pair["file"]).stem] = pair
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    seen_games: set[int] = set()
    for cache in caches:
        baseline = root / "artifacts/benchmark/weeks" / cache.parent.name / "design.npz"
        with (
            np.load(cache, allow_pickle=False) as saved,
            np.load(baseline, allow_pickle=False) as design,
        ):
            keys, labels, bank_keys = saved["keys"], saved["labels"], saved["bank_keys"]
            if not np.array_equal(keys, design["keys"]) or not np.array_equal(
                labels, design["split"]
            ):
                raise ValueError("Final input caches disagree on row order or partitions.")
        frame = pd.DataFrame(keys, columns=KEYS)
        require_keys(frame)
        assigned = frame.game_id.map(allowed)
        if assigned.isna().any() or not np.array_equal(labels, assigned.to_numpy(str)):
            raise ValueError("Holdout, unassigned, or mislabelled rows cannot enter a final fit.")
        bank = pd.DataFrame(bank_keys, columns=ENTITY)
        if bank.isna().any().any() or bank.duplicated(ENTITY).any():
            raise ValueError("Final input player entities must be nonnull and unique.")
        expected_entities = pd.MultiIndex.from_frame(frame[ENTITY].drop_duplicates())
        actual_entities = pd.MultiIndex.from_frame(bank)
        if len(expected_entities) != len(actual_entities) or len(
            expected_entities.difference(actual_entities)
        ):
            raise ValueError("Feature-bank entities do not exactly match the authorized targets.")
        games = set(frame.game_id.astype(int))
        pair = audited.get(cache.parent.name)
        if pair is None or games != set(pair["games"]) or len(frame) != pair["output_rows"]:
            raise ValueError(
                "Final cache games and frame counts must exactly match the data audit."
            )
        if seen_games & games:
            raise ValueError("A game appears in more than one weekly input cache.")
        seen_games.update(games)
        frames.append(frame)
        row = {
            "week": cache.parent.name,
            "rows": len(frame),
            "trajectories": len(expected_entities),
            "games": sorted(games),
            "original_partition_rows": {str(k): int(v) for k, v in assigned.value_counts().items()},
            "ordered_keys_sha256": hashlib.sha256(
                np.ascontiguousarray(keys, dtype="<i8").tobytes()
            ).hexdigest(),
        }
        rows.append(row)
        run.event("final_input_week_verified", week=row["week"], rows=row["rows"], games=len(games))
    if seen_games != set(allowed):
        raise ValueError("Final input caches do not cover every authorized training game.")
    combined = pd.concat(frames, ignore_index=True)
    require_keys(combined)
    return {
        "status": "passed",
        "weeks": rows,
        "training_games": len(seen_games),
        "training_rows": len(combined),
        "training_trajectories": len(combined[ENTITY].drop_duplicates()),
        "original_partition_rows": {
            partition: sum(row["original_partition_rows"].get(partition, 0) for row in rows)
            for partition in ("train", "validation")
        },
        "holdout_outcomes_opened": False,
    }


def load_protocol(root: Path) -> dict[str, Any]:
    """A saved plan must still match its exact source, frozen evidence, and input bytes."""
    value: dict[str, Any] = json.loads((root / "artifacts/final/protocol.json").read_text())
    payload = {k: v for k, v in value.items() if k != "source_signature"}
    if value.get("status") != "prepared" or value.get("source_signature") != digest(payload):
        raise ValueError("The final protocol is incomplete or its signature changed.")
    for relative, expected in value["inputs"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Final protocol input changed: {relative}")
    return value


def freeze_protocol(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Never let an ordinary rerun overwrite the plan for a final evaluation."""
    destination = root / "artifacts/final/protocol.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {**payload, "source_signature": digest(payload)}
    with FileLock(str(destination) + ".lock", timeout=1):
        if destination.exists():
            if load_protocol(root) != value:
                raise ValueError(
                    "The final protocol is frozen; changed plans require explicit review."
                )
        else:
            # Validate bindings before the first durable write as well as on subsequent loads.
            for relative, expected in payload["inputs"].items():
                if sha256(root / relative) != expected:
                    raise ValueError("A final protocol input changed during preparation.")
            atomic_json(destination, value)
    return value


def prepare(root: Path, run: Run) -> dict[str, Any]:
    """Review actual research evidence, validate all final-fit rows, then freeze the plan."""
    from nfl_trajectory.benchmark import protocol
    from nfl_trajectory.context_features import context_catalog
    from nfl_trajectory.feature_candidates import research_catalog
    from nfl_trajectory.feature_research import verify_inputs
    from nfl_trajectory.representation_features import representation_catalog
    from nfl_trajectory.research_evidence import verified_checkpoint, verified_plan
    from nfl_trajectory.research_gate import review

    folder = root / "artifacts/final"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / "pipeline.lock"), timeout=1):
        gate = review(root, run)
        freeze_path = root / "artifacts/research/gate/selection_manifest.json"
        verified_checkpoint(root, "feature-gate-review", gate["source_signature"], freeze_path)
        freeze = json.loads(freeze_path.read_text())
        splits, audit = protocol(root)
        catalog = pd.concat([research_catalog(), context_catalog(), representation_catalog()])
        contract = refit_contract(gate, freeze, splits, catalog)
        estimator = verified_plan(root, root / "artifacts/nonlinear_probe", "nonlinear_probe.py")
        if estimator["settings"] != contract["estimator"]["settings"]:
            raise ValueError(
                "Final estimator settings differ from the verified controlled experiment."
            )
        contract["estimator"]["sklearn_version"] = estimator["sklearn_version"]
        caches, inputs = verify_inputs(root)
        manifest_path = root / "artifacts/research/input_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected = {row["path"]: row["sha256"] for row in manifest["files"]}
        # Raw pre-throw observations are needed to rebuild contextual features on the final fit.
        # No output CSV or reserved input week is opened here.
        for cache in caches:
            relative = "data/raw/train/" + cache.parent.name + ".csv"
            if expected.get(relative) != sha256(root / relative):
                raise ValueError(
                    "Final raw observations differ from the verified research restore."
                )
            inputs[relative] = expected[relative]
        numerical_modules = (
            "benchmark",
            "models",
            "motion",
            "features",
            "feature_experiment",
            "feature_candidates",
            "feature_contracts",
            "feature_research",
            "context_features",
            "context_experiment",
            "representation_features",
            "representation_experiment",
            "tree_inference",
        )
        for name in numerical_modules:
            relative = f"src/nfl_trajectory/{name}.py"
            inputs[relative] = sha256(root / relative)
        for path in (
            freeze_path,
            manifest_path,
            root / "artifacts/research/gate/summary.json",
            root / "artifacts/benchmark/protocol.json",
            root / "artifacts/nonlinear_probe/plan.json",
            root / "src/nfl_trajectory/final_protocol.py",
            root / "src/nfl_trajectory/runtime.py",
            root / "scripts/prepare_final.py",
            root / "uv.lock",
        ):
            inputs[str(path.relative_to(root))] = sha256(path)
        inventory_path = folder / "input_review.json"
        stage(
            root,
            "final-input-review",
            digest({"inputs": inputs, "contract": contract}),
            [inventory_path],
            lambda: atomic_json(
                inventory_path, training_inventory(root, caches, splits, audit, run)
            ),
            run,
        )
        inventory = json.loads(inventory_path.read_text())
        inputs[str(inventory_path.relative_to(root))] = sha256(inventory_path)
        value = freeze_protocol(
            root,
            {
                "format": 1,
                "status": "prepared",
                "feature_gate_source": gate["source_signature"],
                "inputs": inputs,
                "contract": contract,
                "training_inventory": inventory,
                "final_model": False,
                "holdout_evaluation": "not_run",
            },
        )
        summary = {
            "status": "prepared",
            "source_signature": value["source_signature"],
            "feature_gate_source": gate["source_signature"],
            "selected_model": contract["selected_model"],
            "partitions": {
                name: {"games": len(games), "first_game": min(games), "last_game": max(games)}
                for name, games in (
                    ("final_training", contract["training_games"]),
                    ("reserved_holdout", contract["holdout_games"]),
                )
            },
            "training_rows": inventory["training_rows"],
            "training_trajectories": inventory["training_trajectories"],
            "original_partition_rows": inventory["original_partition_rows"],
            "input_files_verified": len(inputs),
            "profile_columns": {
                name: {"count": len(names), "ordered_schema_sha256": digest(names)}
                for name, names in contract["availability_profile_features"].items()
            },
            "research_active_inputs": contract["research_active_inputs"],
            "estimator": contract["estimator"],
            "preprocessing": contract["preprocessing"],
            "evaluation": contract["evaluation"],
            "final_model": False,
            "holdout_evaluation": "not_run",
            "remaining": [
                "Execute and checkpoint the final preprocessing and two availability-profile fits.",
                "Seal final model and predictions; implement and run the reserved evaluation.",
                "Verify final raw/standalone/gateway parity and publish final notebook evidence.",
            ],
        }
        atomic_json(folder / "summary.json", summary)
        run.event(
            "final_protocol_prepared",
            source_signature=value["source_signature"],
            training_games=inventory["training_games"],
            training_rows=inventory["training_rows"],
            holdout_games=len(contract["holdout_games"]),
            holdout_evaluation="not_run",
        )
        return summary
