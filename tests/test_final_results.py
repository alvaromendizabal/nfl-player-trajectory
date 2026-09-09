"""Public notebooks must reject mixed final-model lineages and stale report bytes."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from nfl_trajectory.final_protocol import digest
from nfl_trajectory.final_results import REPORTS, load_final_results, validate_lineage
from nfl_trajectory.runtime import atomic_json, sha256


def reports_fixture():
    seal = {
        "provenance": {"bundle_sha256": "bundle", "outcome_hashes": {"week": "truth"}},
        "rows": 12,
        "games": [2025010100],
    }
    seal["source_signature"] = digest(seal)
    return {
        "fit": {
            "status": "passed",
            "final_fit": True,
            "source_signature": "fit",
            "provenance": {"protocol_source": "protocol"},
        },
        "inference": {
            "status": "passed",
            "fit_source": "fit",
            "protocol_source": "protocol",
            "provenance": {"bundle_sha256": "bundle"},
        },
        "seal": seal,
        "evaluation": {
            "status": "passed",
            "primary_scenario": "complete",
            "holdout_evaluation": "completed",
            "seal": seal["source_signature"],
            "fit_source": "fit",
            "protocol_source": "protocol",
            "rows": 12,
            "games": 1,
            "outcome_hashes": {"week": "truth"},
        },
    }


def test_final_publication_rejects_mixed_lineage():
    reports = reports_fixture()
    validate_lineage(reports)
    for kind, key, replacement in (
        ("evaluation", "fit_source", "other"),
        ("evaluation", "seal", "other"),
        ("evaluation", "rows", 11),
        ("inference", "protocol_source", "other"),
        ("fit", "status", "failed"),
    ):
        changed = copy.deepcopy(reports)
        changed[kind][key] = replacement
        with pytest.raises(ValueError, match="lineages"):
            validate_lineage(changed)


def test_clean_checkout_verifies_reports_and_rejects_local_or_public_drift(tmp_path):
    assert load_final_results(tmp_path) is None
    reports = reports_fixture()
    inputs = {}
    for kind, name in REPORTS.items():
        path = tmp_path / "docs/results" / name
        atomic_json(path, reports[kind])
        inputs[str(path.relative_to(tmp_path))] = sha256(path)
    source = Path(__file__).resolve().parents[1]
    for name in ("final_results", "final_evaluation", "final_inference"):
        relative = "src/nfl_trajectory/" + name + ".py"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, path)
        inputs[relative] = sha256(path)
    manifest = {
        "status": "passed",
        "files": inputs,
        "seal": reports["seal"]["source_signature"],
        "holdout_evaluation": "completed",
    }
    atomic_json(
        tmp_path / "docs/results/final_results_manifest.json",
        {**manifest, "source_signature": digest(manifest)},
    )
    assert load_final_results(tmp_path) == reports
    local = tmp_path / "artifacts/final/evaluation/summary.json"
    atomic_json(local, {**reports["evaluation"], "fit_source": "stale"})
    with pytest.raises(ValueError, match="Local final evaluation"):
        load_final_results(tmp_path)
    local.unlink()
    public = tmp_path / "docs/results/final_evaluation.json"
    public.write_text(json.dumps({**reports["evaluation"], "rows": 11}))
    with pytest.raises(ValueError, match="report changed"):
        load_final_results(tmp_path)
