"""End-to-end checks for sealed holdout, metric pooling, and resumable computation."""

import base64
import json
import re

import numpy as np
import pandas as pd
import pytest

import nfl_trajectory.benchmark as module
from nfl_trajectory.benchmark import benchmark, bootstrap_scores, error_metrics, protocol
from nfl_trajectory.motion import KEYS, trajectory_metrics
from nfl_trajectory.runtime import Run, atomic_json, sha256
from nfl_trajectory.visualization import example_figure


@pytest.fixture
def project(tmp_path):
    (tmp_path / "data/raw/train").mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    pairs, split = [], []
    for week, (game, label) in enumerate(
        [
            (2023090700, "train"),
            (2023091400, "train"),
            (2023120400, "validation"),
            (2023122100, "holdout"),
        ],
        1,
    ):
        observed, truth = [], []
        for player, role in [(1, "Targeted Receiver"), (2, "Defensive Coverage")]:
            for frame in range(1, 6):
                observed.append(
                    {
                        "game_id": game,
                        "play_id": 1,
                        "nfl_id": player,
                        "frame_id": frame,
                        "x": 30 + 0.5 * frame,
                        "y": 20 + player + 0.2 * frame,
                        "ball_land_x": 40,
                        "ball_land_y": 28,
                        "num_frames_output": 12,
                        "player_role": role,
                    }
                )
            for frame in range(1, 13):
                truth.append(
                    {
                        "game_id": game,
                        "play_id": 1,
                        "nfl_id": player,
                        "frame_id": frame,
                        "x": 32.5 + frame * 0.5 - frame**2 * 0.01,
                        "y": 21 + player + frame * 0.2 + frame**2 * 0.01,
                    }
                )
        name = f"input_2023_w{week:02d}.csv"
        pd.DataFrame(observed).to_csv(tmp_path / "data/raw/train" / name, index=False)
        pd.DataFrame(truth).to_csv(
            tmp_path / "data/raw/train" / name.replace("input_", "output_"), index=False
        )
        pairs.append({"file": name, "games": [game], "status": "passed"})
        date = pd.to_datetime(str(game)[:8], format="%Y%m%d").strftime("%Y-%m-%d")
        split.append({"game_id": game, "game_date": date, "split": label})
    pd.DataFrame(split).to_csv(tmp_path / "artifacts/game_splits.csv", index=False)
    atomic_json(
        tmp_path / "artifacts/audit_summary.json",
        {"status": "passed", "competition": "nfl-big-data-bowl-2026-prediction", "pairs": pairs},
    )
    return tmp_path


def test_error_metrics_match_reference_with_unequal_trajectory_lengths():
    truth = pd.DataFrame(
        {
            "game_id": [1] * 4,
            "play_id": [1] * 4,
            "nfl_id": [1, 2, 2, 2],
            "frame_id": [1, 1, 2, 3],
            "x": [0.0] * 4,
            "y": [0.0] * 4,
        }
    )
    prediction = truth.copy()
    prediction["x"] = [4.0, 0.0, 0.0, 2.0]
    errors = prediction[KEYS].copy()
    errors["dx"] = prediction.x
    errors["dy"] = prediction.y
    assert error_metrics(errors) == trajectory_metrics(truth, prediction)
    assert error_metrics(errors)["coordinate_rmse_yards"] == pytest.approx(np.sqrt(20 / 8))
    assert error_metrics(errors)["fde_trajectory_weighted_yards"] == 3


def test_bootstrap_resamples_games_and_pools_coordinate_errors():
    errors = pd.DataFrame({"game_id": [1] + [2] * 9, "dx": [2.0] + [0.0] * 9, "dy": [0.0] * 10})
    values = bootstrap_scores(errors)
    np.testing.assert_allclose(np.unique(values), [0, np.sqrt(0.2), np.sqrt(2)])
    np.testing.assert_array_equal(values, bootstrap_scores(errors))


def test_protocol_rejects_overlapping_and_changed_partitions(project):
    protocol(project)
    path = project / "artifacts/game_splits.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "split"] = "validation"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Frozen"):
        protocol(project)
    frame.loc[0, "split"] = "holdout"
    frame.loc[3, "split"] = "train"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        protocol(project)


def test_pipeline_seals_holdout_reuses_weeks_and_recovers_corruption(project, monkeypatch):
    original = pd.read_csv
    reads = []

    def tracked(path, *args, **kwargs):
        if "w04" in str(path):
            raise AssertionError("Holdout CSV opened during development")
        reads.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracked)
    with Run(project, "benchmark") as run:
        benchmark(project, run)
    summary = json.loads((project / "artifacts/benchmark/summary.json").read_text())
    assert summary["validation_games"] == 1
    assert summary["holdout_evaluation"] == "not_run"
    assert len(summary["models"]) == 6
    fitted_path = project / "artifacts/benchmark/model.json"
    fitted_hash = sha256(fitted_path)
    fitted = json.loads(fitted_path.read_text())
    assert fitted["training_games"] == [2023090700, 2023091400]
    animation = example_figure(project, summary, fitted)
    assert len(animation.frames) == 13
    assert len(animation.layout.sliders[0].steps) == 13
    for frame in animation.frames:
        assert list(frame.traces) == [1, 2, 4, 5]
        assert all(len(trace.x) == int(frame.name) + 1 for trace in frame.data)
    report = (project / "artifacts/benchmark/report.html").read_text()
    sources = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', report)
    assert len(sources) == 3
    for source in sources:
        assert source.startswith("data:image/png;base64,")
        decoded = base64.b64decode(source.split(",", 1)[1], validate=True)
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
    assert "Plotly.addFrames" in report
    caches = sorted((project / "artifacts/benchmark/weeks").glob("*/design.npz"))
    timestamps = {path: path.stat().st_mtime_ns for path in caches}
    reads.clear()
    with Run(project, "benchmark") as run:
        benchmark(project, run)
    assert {path: path.stat().st_mtime_ns for path in caches} == timestamps
    assert not any("input_2023_w01.csv" in path for path in reads)
    assert sha256(fitted_path) == fitted_hash
    caches[0].write_bytes(b"interrupted file")
    with Run(project, "benchmark") as run:
        benchmark(project, run)
    assert caches[0].stat().st_mtime_ns != timestamps[caches[0]]
    assert caches[1].stat().st_mtime_ns == timestamps[caches[1]]
    assert sha256(fitted_path) == fitted_hash
    assert (project / "artifacts/benchmark/report.html").stat().st_size > 1000


def test_interruption_keeps_successful_week_checkpoint(project, monkeypatch):
    original = module.prepare_week

    def interrupted(root, pair, splits, destination):
        if pair["file"].endswith("w02.csv"):
            raise KeyboardInterrupt
        return original(root, pair, splits, destination)

    monkeypatch.setattr(module, "prepare_week", interrupted)
    with Run(project, "benchmark") as run, pytest.raises(KeyboardInterrupt):
        benchmark(project, run)
    first = project / "artifacts/benchmark/weeks/input_2023_w01/design.npz"
    before = first.stat().st_mtime_ns
    monkeypatch.setattr(module, "prepare_week", original)
    with Run(project, "benchmark") as run:
        benchmark(project, run)
    assert first.stat().st_mtime_ns == before
