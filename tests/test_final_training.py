"""Exercise actual preprocessing fits, interrupted recovery, and final partition boundaries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_feature_research import research_project as research_project

from nfl_trajectory.feature_candidates import HISTORY_NAMES
from nfl_trajectory.feature_research import verify_inputs
from nfl_trajectory.final_protocol import freeze_protocol
from nfl_trajectory.final_training import fit_route_encoder, preprocess, validate_preprocessing
from nfl_trajectory.runtime import Run, sha256


@pytest.fixture
def final_project(research_project):
    root = research_project
    _, inputs = verify_inputs(root)
    splits = pd.read_csv(root / "artifacts/game_splits.csv")
    games = splits.loc[~splits.split.eq("holdout"), "game_id"].tolist()
    (root / "scripts").mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/refit_final.py"
    (root / "scripts/refit_final.py").write_bytes(script.read_bytes())
    freeze_protocol(
        root,
        {
            "status": "prepared",
            "inputs": inputs,
            "contract": {
                "training_games": games,
                "holdout_games": splits.loc[splits.split.eq("holdout"), "game_id"].tolist(),
            },
            "training_inventory": {"training_rows": len(games) * 30, "training_trajectories": 21},
        },
    )
    return root, games


def test_preprocessing_refits_combined_partition_and_recovers_after_interruption(
    final_project, monkeypatch
):
    root, games = final_project
    original = sha256(root / "artifacts/benchmark/model.json")
    folder = root / "artifacts/final/preprocessing"

    def interrupted(*args):
        raise RuntimeError("Controlled route-stage interruption")

    monkeypatch.setattr("nfl_trajectory.final_training.fit_route_encoder", interrupted)
    with Run(root, "final-test") as run, pytest.raises(RuntimeError, match="Controlled"):
        preprocess(root, run)
    completed = [folder / n for n in ("baseline.json", "history.csv", "history_model.json")]
    before = {p.name: (sha256(p), p.stat().st_mtime_ns) for p in completed}
    monkeypatch.setattr("nfl_trajectory.final_training.fit_route_encoder", fit_route_encoder)
    with Run(root, "final-test") as resumed:
        result = preprocess(root, resumed)
    assert {p.name: (sha256(p), p.stat().st_mtime_ns) for p in completed} == before
    assert result["baseline_coordinate_count"] == 420
    assert result["route_training_entities"] == result["history_entities"] == 21
    assert result["training_games"] == len(games) == 7
    assert result["route_components"] == 16 and result["route_prototypes"] == 8
    assert result["holdout_evaluation"] == "not_run" and result["final_model"] is False
    assert sha256(root / "artifacts/benchmark/model.json") == original
    events = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
    assert {e["stage"] for e in events if e["event"] == "stage_reused"} == {
        "final-baseline",
        "final-history",
    }
    with Run(root, "final-test") as repeat:
        assert preprocess(root, repeat) == result
    assert '"stage_started"' not in repeat.log_path.read_text()


def test_route_fit_is_order_invariant_and_excludes_unselected_games(final_project):
    root, games = final_project
    caches, _ = verify_inputs(root)
    first = fit_route_encoder(caches, games[:-1])
    assert first == fit_route_encoder(list(reversed(caches)), games[:-1])
    assert first["training_entities"] == 18
    assert first["training_games"] == games[:-1]
    with pytest.raises(ValueError, match="exactly once"):
        fit_route_encoder(caches + caches[:1], games)


def test_saved_preprocessing_rejects_invalid_coverage_and_first_date_leakage(final_project):
    root, games = final_project
    with Run(root, "final-test") as run:
        preprocess(root, run)
    folder = root / "artifacts/final/preprocessing"
    path = folder / "history.csv"
    original = path.read_bytes()
    history = pd.read_csv(path)
    first = history.game_id == games[0]
    history.loc[first, HISTORY_NAMES[2]] = 100
    history.to_csv(path, index=False)
    with pytest.raises(ValueError, match="first training date"):
        validate_preprocessing(folder, games, 210, 21)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="forecast coordinate"):
        validate_preprocessing(folder, games, 209, 21)
    with pytest.raises(ValueError, match="route encoder"):
        validate_preprocessing(folder, games, 210, 20)
    history.loc[first, HISTORY_NAMES[2]] = np.nan
    history.to_csv(path, index=False)
    with pytest.raises(ValueError, match="historical encodings"):
        validate_preprocessing(folder, games, 210, 21)


def test_sealed_model_prevents_preprocessing_refit(final_project):
    root, _ = final_project
    (root / "artifacts/final/model_seal.json").write_text("{}")
    with Run(root, "final-test") as run, pytest.raises(ValueError, match="sealed"):
        preprocess(root, run)
