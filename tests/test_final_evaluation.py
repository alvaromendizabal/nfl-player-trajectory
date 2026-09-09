"""Exercise the prediction-before-outcomes boundary, keyed scoring and interrupted recovery."""

import copy
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_feature_research import example_tracking
from test_feature_research import research_project as research_project
from test_final_fit import fitting_project as fitting_project
from test_final_inference import inference_project as inference_project

from nfl_trajectory import final_evaluation as evaluation
from nfl_trajectory.final_inference import final_bundle
from nfl_trajectory.final_protocol import EVALUATION, digest, load_protocol
from nfl_trajectory.motion import KEYS
from nfl_trajectory.runtime import Run, atomic_bytes, atomic_json, sha256, stage


def test_requests_come_only_from_observed_prediction_players_and_horizons():
    observed, truth = example_tracking()
    observed["player_to_predict"] = observed.nfl_id.ne(2)
    observed.loc[observed.nfl_id.eq(3), "num_frames_output"] = 4
    keys, roles = evaluation.prediction_request(observed.sample(frac=1, random_state=7))
    assert len(keys) == len(roles) == 14
    assert keys.groupby("nfl_id").frame_id.max().to_dict() == {1: 10, 3: 4}
    assert "x" not in keys and "y" not in keys
    for horizon in (0, -1, 1.5, np.inf, 1001):
        malformed = observed.copy()
        malformed["num_frames_output"] = float(horizon)
        with pytest.raises(ValueError, match="horizons"):
            evaluation.prediction_request(malformed)


def test_sealed_scoring_joins_exact_keys_and_rejects_lost_rows():
    _, truth = example_tracking()
    keys = truth[KEYS].sample(frac=1, random_state=2026).reset_index(drop=True)
    prediction = keys.merge(truth, on=KEYS)[["x", "y"]].to_numpy() + [3, 4]
    errors = evaluation.score_predictions(truth, keys, prediction)
    pd.testing.assert_frame_equal(errors[KEYS], keys)
    np.testing.assert_array_equal(errors[["dx", "dy"]], np.broadcast_to([3, 4], (len(keys), 2)))
    assert evaluation.error_metrics(errors)["coordinate_rmse_yards"] == np.sqrt(12.5)
    with pytest.raises(ValueError, match="exactly one"):
        evaluation.score_predictions(truth.iloc[1:], keys, prediction)
    with pytest.raises(ValueError, match="unique"):
        evaluation.score_predictions(pd.concat([truth, truth.iloc[:1]]), keys, prediction)
    with pytest.raises(ValueError, match="frozen holdout"):
        evaluation.validate_forecast_partition(keys, [2025010100])


def test_seal_recovery_outcome_boundary_and_same_evaluation_resume(inference_project, monkeypatch):
    root = inference_project
    bundle = final_bundle(root)
    protocol = copy.deepcopy(load_protocol(root))
    protocol["contract"]["evaluation"] = EVALUATION
    games = [2025010100, 2025010200, 2025010300]
    protocol["contract"]["holdout_games"] = games
    monkeypatch.setattr(evaluation, "load_protocol", lambda root: protocol)
    monkeypatch.setattr(evaluation, "final_bundle", lambda root: bundle)
    source = Path(__file__).resolve().parents[1]
    for relative in ("src/nfl_trajectory/final_evaluation.py", "scripts/evaluate_final.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, path)
    directory = root / "artifacts/final/inference"
    names = ("summary.json", "bundle.json", "standalone.py", "submission.parquet")

    def gateway():
        for name in names:
            atomic_json(
                directory / name,
                {
                    "status": "passed",
                    "source_signature": "fixture",
                    "provenance": {"bundle_sha256": digest(bundle)},
                },
            )

    with Run(root, "gateway-fixture") as run:
        stage(root, "final-gateway", "fixture", [directory / n for n in names], gateway, run)
    entries, labels = [], {}
    for week, game in zip((16, 17, 18), games, strict=True):
        observed, truth = example_tracking(game)
        observed["player_to_predict"] = True
        for kind, frame in (("input", observed), ("output", truth)):
            name = f"data/raw/train/{kind}_2023_w{week}.csv"
            payload = frame.to_csv(index=False).encode()
            entries.append(
                {"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
            )
            if kind == "input":
                atomic_bytes(root / name, payload)
            else:
                labels[name] = payload
    manifest = root / "artifacts/final/source_manifest.json"
    atomic_json(manifest, {"format": 1, "files": entries})
    monkeypatch.setattr(evaluation, "SOURCE_SNAPSHOT", sha256(manifest))
    actual_predict = evaluation.predict_final
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("Controlled next-week interruption")
        return actual_predict(*args, **kwargs)

    monkeypatch.setattr(evaluation, "predict_final", interrupted)
    with Run(root, "seal-test") as run, pytest.raises(RuntimeError, match="Controlled"):
        evaluation.seal_predictions(root, run)
    first = root / "artifacts/final/predictions/week_16.npz"
    before = sha256(first), first.stat().st_mtime_ns
    assert not (root / "artifacts/final/model_seal.json").exists()
    with Run(root, "seal-test") as run:
        seal = evaluation.seal_predictions(root, run)
    assert (sha256(first), first.stat().st_mtime_ns) == before
    assert seal["rows"] == 90 and seal["games"] == games
    assert all(not (root / name).exists() for name in labels)
    assert not (root / "artifacts/final/outcome_access.json").exists()
    for name, payload in labels.items():
        atomic_bytes(root / name, payload)
    read = pd.read_csv
    seen = []

    def guarded_read(path, *args, **kwargs):
        if Path(path).name.startswith("output_"):
            assert evaluation.load_seal(root)["source_signature"] == seal["source_signature"]
            seen.append(str(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    with Run(root, "evaluation-test") as run:
        report = evaluation.evaluate_sealed(root, run)
    assert len(seen) == 3
    assert report["primary_scenario"] == "complete" and report["rows"] == 90
    assert len(report["metrics"]) == 6
    assert (
        report["metrics"][0]["coordinate_rmse_yards"]
        == report["metrics"][1]["coordinate_rmse_yards"]
    )
    with Run(root, "evaluation-test") as run:
        assert evaluation.evaluate_sealed(root, run) == report
        assert evaluation.seal_predictions(root, run) == seal
    assert len(seen) == 3
    original = first.read_bytes()
    first.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="sealed input"):
        evaluation.load_seal(root)
    first.write_bytes(original)
    implementation = root / "scripts/evaluate_final.py"
    implementation.write_text(implementation.read_text() + "\n")
    with Run(root, "evaluation-test") as run, pytest.raises(ValueError, match="implementation"):
        evaluation.evaluate_sealed(root, run)


def test_evaluation_refuses_to_open_outcomes_without_a_seal(tmp_path, monkeypatch):
    (tmp_path / "artifacts/final").mkdir(parents=True)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("Outcomes opened before seal"))
    with Run(tmp_path, "unsealed-test") as run, pytest.raises(FileNotFoundError):
        evaluation.evaluate_sealed(tmp_path, run)
