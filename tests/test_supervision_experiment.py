"""Scientific-runner contracts: matched keys, durable epochs, bootstrap and resume."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_motion_supervision import assert_tree_equal
from test_supervision_batches import samples

from nfl_trajectory.motion import KEYS
from nfl_trajectory.supervision_evidence import load_generation
from nfl_trajectory.supervision_experiment import (
    TrainingSettings,
    evaluate_ema,
    experiment_signature,
    paired_game_bootstrap,
    summarize_pair,
    train_arm,
)
from nfl_trajectory.supervision_plan import training_plan


def dataset(train_count=5):
    training = samples(train_count)
    validation = copy.deepcopy(training[:2])
    for index, sample in enumerate(validation):
        sample["split"] = np.array("validation")
        sample["keys"][:, 0] = 9000 + index
        sample["keys"][:, 1] = index + 1
    return training + validation


def settings(epochs=3):
    return TrainingSettings(epochs=epochs, batch_plays=2, warmup_steps=2, width=32)


def signature(arm="coordinate"):
    return experiment_signature("a" * 64, "b" * 64, "c" * 64, arm)


def publisher_receipts():
    receipts = []

    def publish(folder: Path, receipt):
        assert (folder / "checkpoint.json").is_file()
        assert (folder / (receipt["sha256"] + ".pt")).is_file()
        receipts.append(copy.deepcopy(receipt))

    return receipts, publish


def denominators(values, batch=2):
    plan = training_plan([s for s in values if str(s["split"]) == "train"], batch)
    return (
        plan["training_only_velocity_rms"],
        plan["coordinate_loss_denominator"],
        plan["velocity_loss_denominator"],
    )


def test_signature_binds_named_arm_and_full_hashes():
    assert signature("coordinate") != signature("velocity")
    with pytest.raises(ValueError):
        experiment_signature("a", "b" * 64, "c" * 64, "coordinate")
    with pytest.raises(ValueError):
        experiment_signature("a" * 64, "b" * 64, "c" * 64, "joint")


def test_epoch_checkpoint_publication_and_exact_resume(tmp_path):
    values = dataset()
    scale, cdenom, vdenom = denominators(values)
    config = settings(3)
    receipts, publish = publisher_receipts()
    folder = tmp_path / "coordinate"
    interrupted, _ = train_arm(
        values,
        "coordinate",
        scale,
        cdenom,
        vdenom,
        config,
        folder,
        signature(),
        publish,
        stop_after_steps=4,
    )
    assert interrupted.steps == 4
    assert [r["step"] for r in receipts] == [3, 4]
    durable = load_generation(folder, signature())
    assert durable["steps"] == 4
    resumed, _ = train_arm(
        list(reversed(values)),
        "coordinate",
        scale,
        cdenom,
        vdenom,
        config,
        folder,
        signature(),
        publish,
    )
    clean_receipts, clean_publish = publisher_receipts()
    clean, _ = train_arm(
        values,
        "coordinate",
        scale,
        cdenom,
        vdenom,
        config,
        tmp_path / "clean",
        signature(),
        clean_publish,
    )
    assert resumed.steps == clean.steps == 9
    assert_tree_equal(resumed.payload(), clean.payload())


def test_failed_remote_publication_stops_after_durable_local_epoch(tmp_path):
    values = dataset()
    scale, cdenom, vdenom = denominators(values)

    def fail(_folder, _receipt):
        raise RuntimeError("remote readback failed")

    folder = tmp_path / "velocity"
    with pytest.raises(RuntimeError, match="remote readback"):
        train_arm(
            values,
            "velocity",
            scale,
            cdenom,
            vdenom,
            settings(2),
            folder,
            signature("velocity"),
            fail,
        )
    assert load_generation(folder, signature("velocity"))["steps"] == 3


def test_evaluation_retains_exact_keys_and_all_rows(tmp_path):
    values = dataset()
    scale, cdenom, vdenom = denominators(values)
    _, publish = publisher_receipts()
    state, _ = train_arm(
        values,
        "coordinate",
        scale,
        cdenom,
        vdenom,
        settings(1),
        tmp_path / "coordinate",
        signature(),
        publish,
    )
    errors = evaluate_ema(state, values, batch_plays=1)
    validation = [s for s in values if str(s["split"]) == "validation"]
    expected = np.concatenate([s["keys"] for s in validation])
    observed = errors[KEYS].to_numpy(np.int64)
    np.testing.assert_array_equal(
        observed,
        pd.DataFrame(expected, columns=KEYS).sort_values(KEYS).to_numpy(np.int64),
    )
    assert len(errors) == sum(len(s["keys"]) for s in validation)
    assert errors[["dx", "dy"]].notna().all().all()


def error_frame(offset=0.0):
    rows = []
    for game in (10, 20, 30):
        for frame in (1, 2, 3):
            rows.append(
                {
                    "game_id": game,
                    "play_id": 1,
                    "nfl_id": game + 100,
                    "frame_id": frame,
                    "dx": 1.0 + offset,
                    "dy": 0.5 + offset,
                    "role_id": 1,
                    "forecast_second": 1,
                }
            )
    return pd.DataFrame(rows)


def test_paired_bootstrap_deterministic_and_gate_direction():
    control = error_frame(0.0)
    better = error_frame(-0.2)
    first = paired_game_bootstrap(control, better, repeats=1000, seed=2026)
    second = paired_game_bootstrap(control, better, repeats=1000, seed=2026)
    assert first == second
    assert first["delta_ci95_high"] < 0
    summary = summarize_pair(control, better, minimum_relative_gain=0.01, repeats=1000)
    assert summary["continuation_gate_passed"]
    assert summary["relative_rmse_gain"] > 0.01


def test_mismatched_keys_and_invalid_exposure_rejected(tmp_path):
    control = error_frame()
    treatment = error_frame(-0.1).iloc[:-1].copy()
    with pytest.raises(ValueError, match="identical forecast keys"):
        paired_game_bootstrap(control, treatment, repeats=1000)
    values = dataset()
    scale, cdenom, vdenom = denominators(values)
    _, publish = publisher_receipts()
    with pytest.raises(ValueError, match="Warmup"):
        train_arm(
            values,
            "coordinate",
            scale,
            cdenom,
            vdenom,
            TrainingSettings(epochs=1, batch_plays=2, warmup_steps=2, width=32),
            tmp_path / "bad",
            signature(),
            publish,
        )
