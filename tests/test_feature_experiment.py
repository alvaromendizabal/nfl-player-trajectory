"""Feature ablation tests: training isolation, complete scoring, and restart parity."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nfl_trajectory.feature_experiment as module
from nfl_trajectory.feature_experiment import feature_experiment, load_week, screen_week, screening
from nfl_trajectory.models import design, fit_statistics, sufficient_statistics
from nfl_trajectory.motion import KEYS
from nfl_trajectory.runtime import Run, atomic_json, sha256


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "data/raw/train"
    folder.mkdir(parents=True)
    (tmp_path / "artifacts/benchmark").mkdir(parents=True)
    splits, pairs, statistics = [], [], []
    for week, (game, split) in enumerate(
        [
            (2023090700, "train"),
            (2023091400, "train"),
            (2023120400, "validation"),
            (2023122100, "holdout"),
        ],
        1,
    ):
        observed, truth = [], []
        for player in (1, 2, 3):
            for frame in range(1, 9):
                observed.append(
                    {
                        "game_id": game,
                        "play_id": 1,
                        "nfl_id": player,
                        "frame_id": frame,
                        "x": 30 + player + frame * (0.2 + 0.03 * week),
                        "y": 20 + player + frame * 0.1,
                        "player_role": "Targeted Receiver" if player == 1 else "Defensive Coverage",
                        "player_side": "Offense" if player == 1 else "Defense",
                        "play_direction": "right",
                        "ball_land_x": 42.0,
                        "ball_land_y": 24.0,
                        "num_frames_output": 12,
                    }
                )
            for frame in range(1, 13):
                truth.append(
                    {
                        "game_id": game,
                        "play_id": 1,
                        "nfl_id": player,
                        "frame_id": frame,
                        "x": 30 + player + (8 + frame) * (0.2 + 0.03 * week) - 0.005 * frame**2,
                        "y": 20 + player + (8 + frame) * 0.1 + 0.01 * frame**2,
                    }
                )
        x, y = pd.DataFrame(observed), pd.DataFrame(truth)
        name = f"input_2023_w{week:02d}.csv"
        x.to_csv(folder / name, index=False)
        y.to_csv(folder / name.replace("input_", "output_"), index=False)
        pairs.append({"file": name, "games": [game], "status": "passed"})
        date = pd.to_datetime(str(game)[:8], format="%Y%m%d").strftime("%Y-%m-%d")
        splits.append({"game_id": game, "game_date": date, "split": split})
        if split == "train":
            state, features = design(x, y[KEYS])
            statistics.append(sufficient_statistics(state, features, y))
    split_path = tmp_path / "artifacts/game_splits.csv"
    pd.DataFrame(splits).to_csv(split_path, index=False)
    fitted = fit_statistics(statistics)
    fitted.update({"training_games": [2023090700, 2023091400], "split_sha256": sha256(split_path)})
    atomic_json(tmp_path / "artifacts/benchmark/model.json", fitted)
    atomic_json(
        tmp_path / "artifacts/audit_summary.json",
        {
            "status": "passed",
            "competition": "nfl-big-data-bowl-2026-prediction",
            "pairs": pairs,
        },
    )
    return tmp_path


def run_project(root):
    with Run(root, "features", heartbeat_seconds=0.05) as run:
        feature_experiment(root, run)
    return run


def test_complete_experiment_and_verified_resume(project):
    # A holdout-only file cannot even be read by the experiment.
    (project / "data/raw/train/input_2023_w04.csv").unlink()
    (project / "data/raw/train/output_2023_w04.csv").unlink()
    baseline = sha256(project / "artifacts/benchmark/model.json")
    run_project(project)
    summary = json.loads((project / "artifacts/features/summary.json").read_text())
    assert summary["status"] == "passed"
    assert summary["candidate_features"] == 2843
    assert summary["validation_rows_per_model"] == 36
    assert summary["validation_games"] == 1
    assert summary["holdout_evaluation"] == "not_run"
    assert len(summary["models"]) == 5
    assert sha256(project / "artifacts/benchmark/model.json") == baseline
    before = {
        p: sha256(p)
        for p in (project / "artifacts/features").rglob("*")
        if p.is_file() and p.suffix != ".lock"
    }
    resumed = run_project(project)
    after = {p: sha256(p) for p in before}
    assert before == after
    events = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
    assert not any(e["event"] == "stage_started" for e in events)
    assert any(e["event"] == "stage_reused" for e in events)
    assert all("timestamp" in e and "elapsed_seconds" in e for e in events)


def test_validation_targets_cannot_change_screening_or_selected_features(project):
    run_project(project)
    folder = project / "artifacts/features/weeks"
    paths = sorted(folder.glob("*/screening.json"))
    _, before = screening(paths)
    cache = folder / "input_2023_w03/features.npz"
    with np.load(cache, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["truth"] += 10000
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    cache.write_bytes(buffer.getvalue())
    screen_week(cache, cache.with_name("screening.json"))
    _, after = screening(paths)
    assert before == after


def test_ablation_signal_groups_are_actually_separated(project):
    run_project(project)
    _, selected = screening(sorted((project / "artifacts/features/weeks").glob("*/screening.json")))
    catalog = module.feature_catalog().set_index("feature")
    assert set(catalog.loc[selected["motion_ridge"], "signal"]) == {"motion"}
    assert "interaction" not in set(catalog.loc[selected["landing_ridge"], "signal"])


def test_failure_retains_completed_week_and_rerun_finishes(project, monkeypatch):
    original = module.save_week
    calls = []

    def fail_second(root, pair, splits, baseline, path):
        calls.append(pair["file"])
        if len(calls) == 2:
            raise RuntimeError("Injected interruption")
        return original(root, pair, splits, baseline, path)

    monkeypatch.setattr(module, "save_week", fail_second)
    with pytest.raises(RuntimeError):
        run_project(project)
    first = project / "artifacts/features/weeks/input_2023_w01/features.npz"
    digest = sha256(first)
    monkeypatch.setattr(module, "save_week", original)
    resumed = run_project(project)
    assert sha256(first) == digest
    assert '"stage_reused"' in resumed.log_path.read_text()


def test_baseline_training_provenance_must_match(project):
    path = project / "artifacts/benchmark/model.json"
    model = json.loads(path.read_text())
    model["training_games"].append(2023120400)
    atomic_json(path, model)
    with pytest.raises(ValueError, match="Baseline training games"):
        run_project(project)


def test_feature_cache_roundtrip_has_no_pickled_objects(project):
    run_project(project)
    cache = project / "artifacts/features/weeks/input_2023_w01/features.npz"
    bank, targets, arrays = load_week(cache)
    assert np.isfinite(bank.matrix(targets)).all()
    assert set(arrays["labels"]) == {"train"}
    assert bank.values.dtype == np.float32


def test_tampered_week_recomputes_instead_of_reusing(project):
    run_project(project)
    path = project / "artifacts/features/weeks/input_2023_w01/features.npz"
    path.write_bytes(b"truncated")
    resumed = run_project(project)
    assert load_week(path)[0].values.size > 0
    events = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
    assert any(
        e["event"] == "stage_started" and e.get("stage") == "features-prepare-input_2023_w01"
        for e in events
    )


def notebook_tools():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts/notebooks.py"
    spec = importlib.util.spec_from_file_location("notebook_tools", path)
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    return tools


def test_feature_publication_requires_matching_hashes(project):
    run_project(project)
    tools = notebook_tools()
    assert len(tools.feature_results(project)) == 3
    path = project / "artifacts/features/benchmark.png"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash"):
        tools.feature_results(project)


def test_incomplete_feature_run_cannot_be_published(project):
    (project / "artifacts/features").mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        notebook_tools().feature_results(project)


def test_remote_checkpoint_hook_runs_between_completed_phases(project):
    calls = []
    with Run(project, "features") as run:
        feature_experiment(project, run, lambda: calls.append("checkpoint"))
    # Three development weeks, three fitted challengers, one completed report.
    assert len(calls) == 7


def export_tools(root):
    import importlib.util
    import shutil

    source = Path(__file__).resolve().parents[1]
    for folder in ("src", "kaggle"):
        shutil.copytree(source / folder, root / folder, dirs_exist_ok=True)
    spec = importlib.util.spec_from_file_location("export_tools", root / "kaggle/export.py")
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    return tools


@pytest.mark.parametrize("model_name", ["motion_ridge", "landing_ridge", "interaction_ridge"])
def test_residual_standalone_export_matches_package(project, model_name):
    import ast

    from nfl_trajectory.feature_experiment import predict_residual
    from nfl_trajectory.features import build_player_features
    from nfl_trajectory.models import predict

    run_project(project)
    tools = export_tools(project)
    notebook, _ = tools.build_notebook(
        project, model_name, project / "artifacts/benchmark/model.json"
    )
    cells = [c.source for c in notebook.cells if c.cell_type == "code"]
    compile("\n".join(cells), "inference.py", "exec")
    namespace = {"__name__": "__main__"}
    exec(compile(cells[0], "model.py", "exec"), namespace)
    interface = next(
        n
        for n in ast.parse(cells[-1]).body
        if isinstance(n, ast.FunctionDef) and n.name == "predict"
    )
    exec(compile(ast.Module(body=[interface], type_ignores=[]), "callback.py", "exec"), namespace)
    observed = pd.read_csv(project / "data/raw/train/input_2023_w03.csv")
    truth = pd.read_csv(project / "data/raw/train/output_2023_w03.csv")
    targets = truth.sample(frac=1, random_state=7).reset_index(drop=True)
    bundle = json.loads((project / "artifacts/features/model.json").read_text())
    baseline = json.loads((project / "artifacts/benchmark/model.json").read_text())
    expected = predict(observed, targets[KEYS], "role_ridge", baseline)[["x", "y"]]
    bank = build_player_features(observed, targets[KEYS[:3]])
    expected += predict_residual(bank, targets[KEYS], bundle["models"][model_name])
    targets[["x", "y"]] = np.nan
    actual = namespace["predict"](targets, observed)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    batches = pd.concat(
        [
            namespace["predict"](targets.iloc[:9], observed),
            namespace["predict"](targets.iloc[9:], observed),
        ],
        ignore_index=True,
    )
    np.testing.assert_allclose(batches, actual, rtol=1e-12, atol=1e-12)
    assert notebook.metadata.nfl_export.model == model_name
    assert notebook.metadata.nfl_export.automatically_submitted is False
    assert notebook.metadata.nfl_export.official_gateway_status == "not_run"
    assert not (project / "artifacts/kaggle/submission.ipynb").exists()


def test_residual_export_rejects_tampered_weights(project):
    run_project(project)
    tools = export_tools(project)
    path = project / "artifacts/features/model.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        tools.build_notebook(project, "landing_ridge", project / "artifacts/benchmark/model.json")


def test_local_selection_analysis_matches_completed_model(project):
    from nfl_trajectory.research import load_evidence

    run_project(project)
    summary, study, label = load_evidence(project)
    assert label == "Verified local experiment"
    assert study["summary_sha256"] == sha256(project / "artifacts/features/summary.json")
    assert study["training_rows"] == 72
    assert summary["validation_rows_per_model"] == 36
