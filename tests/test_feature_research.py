"""Research contracts: chronology, conditional ablations, leakage and recovery."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_trajectory.benchmark import save_design
from nfl_trajectory.feature_candidates import (
    HISTORY_NAMES,
    candidate_matrix,
    historical_encodings,
    research_catalog,
)
from nfl_trajectory.feature_experiment import numerical_sources, save_week
from nfl_trajectory.feature_research import (
    baseline_prediction,
    chronological_folds,
    correction_candidates,
    feature_research,
    fit_baseline,
    model_correction,
    solve_ridge,
    verify_inputs,
)
from nfl_trajectory.features import build_player_features, feature_catalog
from nfl_trajectory.models import design, fit_statistics, sufficient_statistics
from nfl_trajectory.runtime import Run, atomic_json, sha256


def example_tracking(game=2023090700):
    observations, targets = [], []
    for player in range(1, 4):
        for frame in range(1, 9):
            observations.append(
                {
                    "game_id": game,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 30 + player + frame * (0.2 + player * 0.01),
                    "y": 20 + player + frame * 0.12,
                    "player_role": "Targeted Receiver" if player == 1 else "Defensive Coverage",
                    "player_side": "Offense" if player == 1 else "Defense",
                    "play_direction": "right",
                    "ball_land_x": 45.0,
                    "ball_land_y": 25.0,
                    "num_frames_output": 10,
                }
            )
        for frame in range(1, 11):
            targets.append(
                {
                    "game_id": game,
                    "play_id": 1,
                    "nfl_id": player,
                    "frame_id": frame,
                    "x": 30 + player + (8 + frame) * (0.2 + player * 0.01) - 0.007 * frame**2,
                    "y": 20 + player + (8 + frame) * 0.12 + player * 0.002 * frame**2,
                }
            )
    return pd.DataFrame(observations), pd.DataFrame(targets)


def test_catalog_provenance_and_target_poisoning():
    catalog = research_catalog()
    assert len(catalog) == 5387
    assert catalog.feature.is_unique
    assert catalog[["rationale", "availability", "provenance"]].notna().all().all()
    inputs, targets = example_tracking()
    bank = build_player_features(inputs)
    history = np.zeros((len(targets), len(HISTORY_NAMES)))
    reference = candidate_matrix(bank, targets, history)
    targets[["x", "y"]] = np.nan
    np.testing.assert_array_equal(candidate_matrix(bank, targets, history), reference)
    np.testing.assert_array_equal(reference[:, :2843], bank.matrix(targets))


def test_candidate_subset_shuffle_and_batch_parity():
    inputs, targets = example_tracking()
    bank = build_player_features(inputs.sample(frac=1, random_state=3))
    targets = targets.sample(frac=1, random_state=4)
    history = np.arange(len(targets) * len(HISTORY_NAMES)).reshape(len(targets), -1)
    names = research_catalog().feature.tolist()
    columns = [names[i] for i in (10, 2850, 4500, 5200, 5386)]
    expected = candidate_matrix(bank, targets, history)[:, [names.index(c) for c in columns]]
    actual = np.concatenate(
        [
            candidate_matrix(bank, targets.iloc[:11], history[:11], columns),
            candidate_matrix(bank, targets.iloc[11:], history[11:], columns),
        ]
    )
    np.testing.assert_array_equal(actual, expected)


def test_missing_neighbour_projection_is_zero():
    inputs, targets = example_tracking()
    bank = build_player_features(inputs[inputs.nfl_id.eq(1)])
    targets = targets[targets.nfl_id.eq(1)]
    columns = research_catalog().query("family == 'projected_geometry'").feature
    columns = [c for c in columns if "opponent" in c]
    values = candidate_matrix(bank, targets, np.zeros((len(targets), 10)), columns)
    np.testing.assert_array_equal(values, 0)


def history_example():
    return pd.DataFrame(
        {
            "game_id": [2023090700, 2023090701, 2023091400, 2023092100, 2023092200],
            "play_id": [1] * 5,
            "nfl_id": [1] * 5,
            "player_role": ["Defensive Coverage"] * 5,
            "error_x": [1.0, 3.0, 5.0, 1000.0, 2000.0],
            "error_y": [2.0] * 5,
        }
    )


def test_history_excludes_same_date_and_freezes_for_evaluation():
    observations = history_example()
    training = set(observations.game_id[:3])
    reference, fitted = historical_encodings(observations, training, smoothing=2)
    np.testing.assert_array_equal(reference[:2, [0, 2, 3]], 0)
    assert reference[2, 2] == 1.0
    np.testing.assert_array_equal(reference[3], reference[4])
    observations.loc[3:, ["error_x", "error_y"]] = np.nan
    actual, actual_fitted = historical_encodings(observations, training, smoothing=2)
    np.testing.assert_array_equal(reference, actual)
    assert fitted == actual_fitted
    observations.loc[2, "error_x"] = 999
    changed, _ = historical_encodings(observations, training, smoothing=2)
    np.testing.assert_array_equal(changed[:3], reference[:3])


def test_history_rejects_nonchronological_and_unknown_players_are_cold():
    observations = history_example()
    with pytest.raises(ValueError, match="strictly follow"):
        historical_encodings(observations, {2023091400})
    observations.loc[3:, "nfl_id"] = 99
    values, _ = historical_encodings(observations, set(observations.game_id[:3]))
    assert values[3, 1] == 1
    assert values[3, 0] == values[3, 2] == values[3, 3] == 0


def test_conditional_candidates_never_displace_landing_features():
    catalog = research_catalog()
    catalog["training_association"] = np.linspace(1, 0, len(catalog))
    catalog["screen_status"] = "eligible"
    protected = feature_catalog().feature[:64].tolist()
    candidates = correction_candidates(catalog, {"features": protected})
    assert all(not set(names) & set(protected) for names, _ in candidates.values())
    for name, (columns, _) in candidates.items():
        if name.startswith("balanced_without_"):
            group = name.removeprefix("balanced_without_")
            assert not set(columns) & set(candidates[f"plus_{group}"][0])


def test_ridge_rejects_duplicates_and_preserves_budget_reasons():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 3))
    x = np.column_stack([x[:, 0], x[:, 0], x[:, 1], x[:, 2], np.ones(100)])
    y = rng.normal(size=(100, 2))
    model = solve_ridge(x.T @ x, x.T @ y, x.sum(0), y.sum(0), len(x), list("abcde"), 2)
    assert model["features"] == ["a", "c"]
    assert model["rejected"] == {
        "b": "training_redundancy",
        "d": "feature_budget",
        "e": "constant_or_near_constant",
    }
    assert np.isfinite(model_correction(x, dict(zip("abcde", range(5), strict=True)), model)).all()


@pytest.fixture
def research_project(tmp_path):
    games = [
        2023090700,
        2023091400,
        2023092100,
        2023092800,
        2023100500,
        2023101200,
        2023120400,
        2023122100,
    ]
    assignments = ["train"] * 6 + ["validation", "holdout"]
    splits = pd.DataFrame(
        {
            "game_id": games,
            "game_date": [pd.to_datetime(str(g)[:8]).strftime("%Y-%m-%d") for g in games],
            "split": assignments,
        }
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    splits.to_csv(artifacts / "game_splits.csv", index=False)
    raw = tmp_path / "data/raw/train"
    raw.mkdir(parents=True)
    statistics, pairs = [], []
    for week, (game, partition) in enumerate(zip(games, assignments, strict=True), 1):
        inputs, targets = example_tracking(game)
        pair = {"file": f"input_2023_w{week:02d}.csv", "games": [game], "status": "passed"}
        pairs.append(pair)
        if partition == "holdout":
            continue
        inputs.to_csv(raw / pair["file"], index=False)
        targets.to_csv(raw / pair["file"].replace("input_", "output_"), index=False)
        inputs = pd.read_csv(raw / pair["file"])
        targets = pd.read_csv(raw / pair["file"].replace("input_", "output_"))
        state, basis = design(inputs, targets)
        destination = artifacts / "benchmark/weeks" / Path(pair["file"]).stem / "design.npz"
        save_design(destination, state, basis, targets, splits)
        if partition == "train":
            statistics.append(sufficient_statistics(state, basis, targets))
    baseline = fit_statistics(statistics)
    baseline.update(training_games=games[:6], split_sha256=sha256(artifacts / "game_splits.csv"))
    atomic_json(artifacts / "benchmark/model.json", baseline)
    atomic_json(
        artifacts / "audit_summary.json",
        {"status": "passed", "competition": "nfl-big-data-bowl-2026-prediction", "pairs": pairs},
    )
    for pair in pairs[:-1]:
        path = artifacts / "features/weeks" / Path(pair["file"]).stem / "features.npz"
        save_week(tmp_path, pair, splits, baseline, path)
    atomic_json(
        artifacts / "features/model.json",
        {
            "source_sha256": numerical_sources(),
            "baseline_sha256": sha256(artifacts / "benchmark/model.json"),
            "split_sha256": baseline["split_sha256"],
            "training_games": games[:6],
        },
    )
    entries = [
        {"path": str(p.relative_to(tmp_path)), "sha256": sha256(p)}
        for p in [*artifacts.rglob("*.npz"), *raw.glob("input_*.csv")]
    ]
    atomic_json(
        artifacts / "research/input_manifest.json",
        {"format": 1, "files": entries, "scope": "synthetic test only"},
    )
    return tmp_path


def test_fold_local_baseline_and_true_without_landing(research_project):
    root = research_project
    caches, _ = verify_inputs(root)
    splits = pd.read_csv(root / "artifacts/game_splits.csv")
    folds = chronological_folds(splits)
    train = set(folds[0]["training_games"])
    model = fit_baseline(root, caches, train, no_landing=True)
    assert model["training_games"] == sorted(train)
    for params in [model["global"], *model["roles"].values()]:
        np.testing.assert_array_equal(params["coefficients"][3:], 0)
    inputs, targets = example_tracking()
    state, basis = design(inputs, targets)
    first = baseline_prediction(state, basis, model)
    basis[:, :, 3:] = np.nan
    np.testing.assert_array_equal(baseline_prediction(state, basis, model), first)


def test_input_corruption_is_rejected(research_project):
    caches, _ = verify_inputs(research_project)
    caches[0].write_bytes(caches[0].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="unverified"):
        verify_inputs(research_project)


def test_research_end_to_end_resume_and_original_preservation(research_project):
    root = research_project
    before = sha256(root / "artifacts/benchmark/model.json")
    with Run(root, "feature-research", heartbeat_seconds=0.1) as run:
        feature_research(root, run)
    summary = json.loads((root / "artifacts/research/summary.json").read_text())
    assert summary["candidate_features"] == 5387
    assert len(summary["inner_folds"]) == 3
    assert summary["feature_gate"] == "open"
    assert summary["holdout_evaluation"] == "not_run"
    assert summary["validation_rows_per_model"] == 30
    assert sha256(root / "artifacts/benchmark/model.json") == before
    assert not json.loads((root / "artifacts/research/selection.json").read_text())[
        "outer_validation_used_for_selection"
    ]
    models = root / "artifacts/research/development/models.json"
    digest = sha256(models)
    with Run(root, "feature-research", heartbeat_seconds=0.1) as resumed:
        feature_research(root, resumed)
    events = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
    assert not any(e["event"] == "stage_started" for e in events)
    assert sha256(models) == digest
    from nfl_trajectory.context_experiment import context_research

    with Run(root, "context-research", heartbeat_seconds=0.1) as context_run:
        context_research(root, context_run)
    context_summary = json.loads((root / "artifacts/context/summary.json").read_text())
    assert context_summary["candidate_features"] == 1468
    assert context_summary["validation_rows_per_model"] == 30
    assert context_summary["holdout_evaluation"] == "not_run"
    assert len(context_summary["inner_folds"]) == 3
    with Run(root, "context-research", heartbeat_seconds=0.1) as context_resume:
        context_research(root, context_resume)
    assert '"stage_started"' not in context_resume.log_path.read_text()
    from nfl_trajectory.representation_experiment import representation_research

    with Run(root, "representation-research", heartbeat_seconds=0.1) as rep_run:
        representation_research(root, rep_run)
    with Run(root, "representation-research", heartbeat_seconds=0.1) as rep_resume:
        representation_research(root, rep_resume)
    assert '"stage_started"' not in rep_resume.log_path.read_text()
    routes = json.loads((root / "artifacts/representation/inner_1/routes.json").read_text())
    expected_training = summary["inner_folds"][0]["fold"]["training_games"]
    assert routes["training_games"] == expected_training
    assert routes["training_entities"] == 3 * len(expected_training)
    from nfl_trajectory.research import feature_research_snapshot, permutation_study

    snapshot = feature_research_snapshot(root)
    assert snapshot["candidate_features"] == 7999
    assert len(snapshot["families"]) == 20
    assert not snapshot["final_training_ready"]
    importance = permutation_study(root, snapshot)
    assert importance["status"] == "passed"
    assert importance["unpermuted_coordinate_rmse_yards"] == pytest.approx(
        snapshot["selected_metrics"]["coordinate_rmse_yards"]
    )
    from nfl_trajectory.research_inference import predict_research, research_bundle

    bundle = research_bundle(root)
    observed = pd.read_csv(root / "data/raw/train/input_2023_w07.csv")
    targets = pd.read_csv(root / "data/raw/train/output_2023_w07.csv")
    actual = predict_research(observed, targets, bundle)
    error = actual[["x", "y"]].to_numpy() - targets[["x", "y"]].to_numpy()
    assert np.sqrt(np.mean(error**2)) == pytest.approx(
        snapshot["selected_metrics"]["coordinate_rmse_yards"], abs=1e-8
    )
    poisoned = targets.copy()
    poisoned[["x", "y"]] = np.nan
    pd.testing.assert_frame_equal(predict_research(observed, poisoned, bundle), actual)
    with pytest.raises(ValueError, match="strictly follow"):
        predict_research(observed, targets.assign(game_id=bundle["training_games"][0]), bundle)
    import importlib.util
    import shutil

    from nfl_trajectory.research_inference import INFERENCE_MODULES

    source_root = Path(__file__).resolve().parents[1]
    for name in (*INFERENCE_MODULES, "runtime"):
        destination = root / "src/nfl_trajectory" / (name + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "src/nfl_trajectory" / (name + ".py"), destination)
    spec = importlib.util.spec_from_file_location(
        "research_export_test", source_root / "kaggle/export.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    source, _ = exporter.model_source(root, "research", root / "artifacts/benchmark/model.json")
    namespace = {}
    exec(compile(source, "standalone.py", "exec"), namespace)
    exported = namespace["research_predict"](observed, poisoned, namespace["RESEARCH_MODEL"])
    pd.testing.assert_frame_equal(exported, actual)
    probe_spec = importlib.util.spec_from_file_location(
        "nonlinear_probe_test", source_root / "scripts/nonlinear_probe.py"
    )
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    core = json.loads((root / "artifacts/research/inner_1/models.json").read_text())
    ctx = json.loads((root / "artifacts/context/inner_1/models.json").read_text())
    rep = json.loads((root / "artifacts/representation/inner_1/models.json").read_text())
    variants = probe.feature_sets(core, ctx, rep)
    names = sorted(set(variants["all_engineered"] + variants["core_plus_noise"]))
    caches, _ = verify_inputs(root)
    matrix, residual, _, _, _, states = probe.materialize(root, caches, core["fold"], names, True)
    assert set(states.game_id) == set(core["fold"]["training_games"])
    assert matrix.shape == (len(states), len(names))
    assert residual.shape == (len(states), 2)
    assert np.isfinite(matrix).all()
    context_model = root / "artifacts/context/development/models.json"
    context_model.write_text(context_model.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        feature_research_snapshot(root)


def test_trajectory_permutation_preserves_roles_horizons_and_frame_alignment():
    from nfl_trajectory.research import trajectory_permutation

    _, targets = example_tracking()
    targets["player_role"] = "Defensive Coverage"
    targets["num_frames_output"] = 10
    targets = targets.sample(frac=1, random_state=3).reset_index(drop=True)
    order = trajectory_permutation(targets, seed=3)
    assert len(set(order)) == len(targets)
    assert (order != np.arange(len(targets))).any()
    np.testing.assert_array_equal(targets.frame_id, targets.iloc[order].frame_id)
    for indices in targets.groupby(["game_id", "play_id", "nfl_id"]).indices.values():
        assert targets.iloc[order[indices]].nfl_id.nunique() == 1
    broken = targets.copy()
    broken.loc[0, "num_frames_output"] = 99
    with pytest.raises(ValueError, match="constant"):
        trajectory_permutation(broken, seed=3)
