"""The final phase must preserve the scientific boundary and verified input lineage."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.feature_contracts import telemetry_dependent
from nfl_trajectory.final_protocol import (
    freeze_protocol,
    load_protocol,
    refit_contract,
    training_inventory,
)
from nfl_trajectory.runtime import Run, sha256


@pytest.fixture
def frozen_evidence():
    public = Path(__file__).resolve().parents[1] / "docs/results"
    gate = json.loads((public / "feature_gate.json").read_text())
    freeze = json.loads((public / "feature_freeze.json").read_text())
    # Exercise the real closure receipt with small explicit schemas and game assignments.
    names = freeze["screened_features"][:8]
    freeze.update(
        screened_features=names,
        screened_feature_count=len(names),
        features=names[:1],
        used_feature_count=1,
        availability_profile_features={
            "without_metadata": names,
            "without_optional_inputs": [n for n in names if not telemetry_dependent(n)],
        },
        training_games=[2023090700],
        development_games=[2023120300],
    )
    splits = pd.DataFrame(
        {
            "game_id": [2023090700, 2023120300, 2023122400],
            "game_date": ["2023-09-07", "2023-12-03", "2023-12-24"],
            "split": ["train", "validation", "holdout"],
        }
    )
    return gate, freeze, splits, pd.DataFrame({"feature": names})


def test_final_refit_uses_screened_columns_and_train_plus_development(frozen_evidence):
    gate, freeze, splits, catalog = frozen_evidence
    value = refit_contract(gate, freeze, splits, catalog)
    assert value["training_games"] == [2023090700, 2023120300]
    assert value["holdout_games"] == [2023122400]
    assert value["availability_profile_features"]["without_metadata"] == freeze["screened_features"]
    assert len(value["availability_profile_features"]["without_metadata"]) > 1
    assert value["research_active_inputs"] == 1
    assert value["estimator"]["settings"]["early_stopping"] is False


@pytest.mark.parametrize("failure", ["empty_checks", "open", "scored", "stale_lineage"])
def test_unverified_or_open_gate_cannot_authorize_final_training(frozen_evidence, failure):
    gate, freeze, splits, catalog = frozen_evidence
    if failure == "empty_checks":
        gate["checks"] = []
    elif failure == "open":
        gate["measurements"]["pooled_tail_gain"] = 0.02
    elif failure == "scored":
        freeze["holdout_evaluation"] = "completed"
    else:
        freeze["source_signature"] = "0" * 64
    with pytest.raises(ValueError, match="gate|lineage"):
        refit_contract(gate, freeze, splits, catalog)


@pytest.mark.parametrize("failure", ["active_only", "duplicate", "unknown", "count", "reordered"])
def test_final_schema_cannot_be_silently_changed(frozen_evidence, failure):
    gate, freeze, splits, catalog = frozen_evidence
    if failure == "active_only":
        freeze["screened_features"] = freeze["features"]
    elif failure == "duplicate":
        freeze["availability_profile_features"]["without_metadata"].append(
            freeze["screened_features"][0]
        )
    elif failure == "unknown":
        freeze["availability_profile_features"]["without_metadata"].append("unvalidated_feature")
    elif failure == "count":
        freeze["screened_feature_count"] -= 1
    else:
        freeze["availability_profile_features"]["without_optional_inputs"].reverse()
    with pytest.raises(ValueError, match="columns|profile"):
        refit_contract(gate, freeze, splits, catalog)


@pytest.mark.parametrize("feature", ["context__metadata__weight", "motion__telemetry__speed"])
def test_final_profiles_cannot_acquire_unavailable_dependencies(frozen_evidence, feature):
    gate, freeze, splits, catalog = frozen_evidence
    catalog.loc[len(catalog)] = [feature]
    for names in freeze["availability_profile_features"].values():
        names.append(feature)
    with pytest.raises(ValueError, match="availability contract"):
        refit_contract(gate, freeze, splits, catalog)


def test_same_date_cannot_cross_final_boundary(frozen_evidence):
    gate, freeze, splits, catalog = frozen_evidence
    splits.loc[2, ["game_id", "game_date"]] = [2023120301, "2023-12-03"]
    with pytest.raises(ValueError, match="chronological"):
        refit_contract(gate, freeze, splits, catalog)


def test_edited_training_assignment_cannot_reuse_feature_freeze(frozen_evidence):
    gate, freeze, splits, catalog = frozen_evidence
    freeze["development_games"] = [2023122400]
    with pytest.raises(ValueError, match="chronological"):
        refit_contract(gate, freeze, splits, catalog)


def write_cache(root, name, game, label, frame_ids=(1, 2)):
    folder = root / "artifacts/features/weeks" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "features.npz"
    keys = np.array([[game, 1, 101, frame] for frame in frame_ids], dtype=np.int64)
    labels = np.array([label] * len(keys))
    # No outcome array is supplied: preparation should only request identifiers and partitions.
    np.savez(path, keys=keys, labels=labels, bank_keys=keys[:1, :3])
    baseline = root / "artifacts/benchmark/weeks" / name
    baseline.mkdir(parents=True, exist_ok=True)
    np.savez(baseline / "design.npz", keys=keys, split=labels)
    return path


@pytest.fixture
def cached_inputs(tmp_path, frozen_evidence):
    splits = frozen_evidence[2]
    caches = [
        write_cache(tmp_path, "input_2023_w01", 2023090700, "train"),
        write_cache(tmp_path, "input_2023_w13", 2023120300, "validation"),
    ]
    audit = {
        "pairs": [
            {"file": "input_2023_w01.csv", "games": [2023090700], "output_rows": 2},
            {"file": "input_2023_w13.csv", "games": [2023120300], "output_rows": 2},
            {"file": "input_2023_w16.csv", "games": [2023122400], "output_rows": 999},
        ]
    }
    return caches, splits, audit


def test_final_input_review_needs_no_holdout_or_outcome_arrays(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    with Run(tmp_path, "test-final") as run:
        result = training_inventory(tmp_path, caches, splits, audit, run)
    assert result["training_rows"] == 4
    assert result["original_partition_rows"] == {"train": 2, "validation": 2}
    assert result["training_games"] == 2
    assert result["training_trajectories"] == 2
    assert result["holdout_outcomes_opened"] is False


@pytest.mark.parametrize("game,label", [(2023122400, "train"), (2023090700, "validation")])
def test_matching_caches_cannot_hide_holdout_or_mislabelled_rows(
    tmp_path, cached_inputs, game, label
):
    caches, splits, audit = cached_inputs
    write_cache(tmp_path, "input_2023_w01", game, label)
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="cannot enter"):
        training_inventory(tmp_path, caches, splits, audit, run)


def test_incomplete_frame_coverage_is_rejected_even_when_both_caches_agree(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    write_cache(tmp_path, "input_2023_w01", 2023090700, "train", frame_ids=(1,))
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="frame counts"):
        training_inventory(tmp_path, caches, splits, audit, run)


def test_reordered_baseline_rows_cannot_change_feature_alignment(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    path = tmp_path / "artifacts/benchmark/weeks/input_2023_w01/design.npz"
    with np.load(path, allow_pickle=False) as saved:
        keys, labels = saved["keys"], saved["split"]
    np.savez(path, keys=keys[::-1], split=labels)
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="row order"):
        training_inventory(tmp_path, caches, splits, audit, run)


def test_extra_unscored_player_in_feature_bank_is_rejected(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    path = caches[0]
    with np.load(path, allow_pickle=False) as saved:
        keys, labels, bank = saved["keys"], saved["labels"], saved["bank_keys"]
    np.savez(path, keys=keys, labels=labels, bank_keys=np.vstack([bank, [2023122400, 1, 101]]))
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="entities"):
        training_inventory(tmp_path, caches, splits, audit, run)


def test_missing_week_cannot_yield_partial_final_training(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="every authorized"):
        training_inventory(tmp_path, caches[:1], splits, audit, run)


def test_duplicate_week_cannot_double_weight_a_game(tmp_path, cached_inputs):
    caches, splits, audit = cached_inputs
    with Run(tmp_path, "test-final") as run, pytest.raises(ValueError, match="more than one"):
        training_inventory(tmp_path, caches + caches[:1], splits, audit, run)


def test_immutable_protocol_reuses_same_plan_without_rewriting(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"verified": true}')
    payload = {"status": "prepared", "inputs": {"evidence.json": sha256(evidence)}, "setting": 100}
    expected = freeze_protocol(tmp_path, payload)
    path = tmp_path / "artifacts/final/protocol.json"
    timestamp = path.stat().st_mtime_ns
    assert freeze_protocol(tmp_path, payload) == expected == load_protocol(tmp_path)
    assert path.stat().st_mtime_ns == timestamp
    different = copy.deepcopy(payload)
    different["setting"] = 200
    before = path.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        freeze_protocol(tmp_path, different)
    assert path.read_bytes() == before


@pytest.mark.parametrize("changed", ["input", "plan", "missing_input"])
def test_mutated_protocol_or_source_cannot_authorize_a_fit(tmp_path, changed):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("verified input bytes")
    freeze_protocol(tmp_path, {"status": "prepared", "inputs": {"evidence.json": sha256(evidence)}})
    if changed == "input":
        evidence.write_text("different feature definition or raw input")
    elif changed == "missing_input":
        evidence.unlink()
    else:
        path = tmp_path / "artifacts/final/protocol.json"
        payload = json.loads(path.read_text())
        payload["extra_training_game"] = 2023122400
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed"):
        load_protocol(tmp_path)
