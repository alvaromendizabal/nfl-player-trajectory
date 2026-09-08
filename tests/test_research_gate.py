"""A working pipeline cannot close research while feature or inference evidence is weak."""

import pytest

from nfl_trajectory.research_gate import closure_checks, seasons_from_audit


@pytest.fixture
def evidence():
    return {
        "all_screened": True,
        "complete_pool": True,
        "scope_verified": True,
        "feature_gains": [0.12, 0.11, 0.10],
        "tail_gains": [0.002, 0.003, 0.0],
        "pooled_tail_gain": 0.002,
        "metadata_cost": 0.001,
        "positional_cost": 0.025,
        "family_evidence": True,
        "raw_verified": True,
        "gateway_verified": True,
        "holdout_unscored": True,
    }


def test_successful_interface_cannot_hide_remaining_feature_gain(evidence):
    evidence["pooled_tail_gain"] = 0.012
    assert not all(c["passed"] for c in closure_checks(evidence))


@pytest.mark.parametrize(
    "missing", ["complete_pool", "scope_verified", "raw_verified", "gateway_verified"]
)
def test_missing_research_or_latest_artifact_evidence_keeps_gate_open(evidence, missing):
    evidence[missing] = False
    assert not all(c["passed"] for c in closure_checks(evidence))


def test_average_gain_cannot_hide_one_weak_fold(evidence):
    evidence["feature_gains"] = [0.20, 0.20, 0.01]
    assert not all(c["passed"] for c in closure_checks(evidence))


def test_verified_stable_feature_set_can_close_gate(evidence):
    assert all(c["passed"] for c in closure_checks(evidence))


def test_nonfinite_evidence_is_not_a_success(evidence):
    evidence["positional_cost"] = float("nan")
    with pytest.raises(ValueError, match="finite evidence"):
        closure_checks(evidence)


def test_january_calendar_year_does_not_invent_another_labelled_season():
    audit = {"pairs": [{"file": "input_2023_w18.csv", "games": [2024010600]}]}
    assert seasons_from_audit(audit, [2024010600]) == {2023}
