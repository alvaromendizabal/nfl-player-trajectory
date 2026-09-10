"""Synthetic matched-arm, information-boundary and independent recovery checks."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_temporal_data import sample

# The repository's default environment intentionally excludes optional PyTorch.
torch = pytest.importorskip("torch")

from nfl_trajectory.motion_supervision import (  # noqa: E402
    FEATURE_KEYS,
    GroupedMotionModel,
    MatchedState,
    fit_velocity_scale,
    observed_batch,
    supervised_batch,
    supervised_loss,
)
from nfl_trajectory.motion_targets import motion_targets  # noqa: E402
from nfl_trajectory.supervision_evidence import (  # noqa: E402
    load_generation,
    save_generation,
)
from nfl_trajectory.temporal_data import STATIC_NAMES, reflect  # noqa: E402

SIGNATURE = hashlib.sha256(b"synthetic-only;fixed-6-step-order;seed2026").hexdigest()


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_deterministic)


def fixture(players=3, frames=(1, 2, 3, 51, 52, 94), split="train"):
    s = sample(players)
    frames = np.asarray(frames, dtype=np.int64)
    n = len(frames)
    s["horizon"][:] = max(94, int(frames.max()))
    s["static"][:, 6] = s["horizon"] / 10
    s["keys"] = np.column_stack(
        [np.full(n, 2023090700), np.ones(n, dtype=int), np.ones(n, dtype=int), frames]
    )
    s["player"] = np.zeros(n, dtype=np.int64)
    s["time"] = frames.astype(np.float32) / 10
    displacement = s["time"][:, None] * np.array([3.0, -2.0], dtype=np.float32)
    s["baseline"] = (0.7 * displacement).astype(np.float32)
    s["truth"] = (0.3 * displacement).astype(np.float32)
    s["split"] = np.array(split)
    return s


def test_same_initialization_and_parameter_count_across_arms():
    a = MatchedState(32, 2026, 0.0, 3.0)
    b = MatchedState(32, 2026, 0.1, 3.0)
    assert sum(p.numel() for p in a.model.parameters()) == sum(
        p.numel() for p in b.model.parameters()
    )
    for name, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(a.rng, b.rng, rtol=0, atol=0)


def test_grouped_stem_preserves_signal_filters_and_parent_width():
    model = GroupedMotionModel(32)
    for i in (0, 2, 4):
        assert model.temporal[i].groups == 31
        assert model.temporal[i].out_channels == 124
    assert model.temporal[6].out_channels == 48


def test_future_truth_does_not_change_any_forward_tensor_or_prediction():
    a = fixture()
    b = copy.deepcopy(a)
    b["truth"][:] = 123456
    first, second = observed_batch([a]), observed_batch([b])
    assert set(first) == FEATURE_KEYS
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    model = GroupedMotionModel(32).eval()
    with torch.no_grad():
        for key in model(first):
            torch.testing.assert_close(model(first)[key], model(second)[key], rtol=0, atol=0)


@pytest.mark.parametrize("forbidden", ["truth", "future_x", "velocity_target", "split"])
def test_forward_rejects_labels_and_unreviewed_fields(forbidden):
    batch = observed_batch([fixture()])
    batch[forbidden] = torch.zeros(1)
    with pytest.raises(ValueError, match="labels are forbidden"):
        GroupedMotionModel(32)(batch)


def test_inference_does_not_require_truth_or_training_motion_labels():
    s = fixture()
    del s["truth"]
    out = GroupedMotionModel(32).eval()(observed_batch([s]))
    assert out["coordinate"].shape == (1, 6, 2)
    assert torch.isfinite(out["velocity"]).all()


def test_more_than_48_frames_and_variable_request_padding_are_preserved():
    first = fixture(frames=(1, 49, 94))
    second = fixture(5, frames=(1, 2, 3, 4, 5, 6, 94))
    batch, labels = supervised_batch([first, second])
    assert int(batch["scored"].sum()) == 10
    assert float(batch["time"].max()) == pytest.approx(9.4)
    model = GroupedMotionModel(32).eval()
    torch.nn.init.normal_(model.decode[-1].weight, std=0.02)
    out = model(batch)
    assert torch.isfinite(out["coordinate"]).all()
    assert torch.count_nonzero(out["coordinate"][0, 3:]) == 0
    assert labels["coordinate"].shape == (2, 7, 2)


def test_player_permutation_and_padding_invariance():
    original = fixture(3)
    changed = copy.deepcopy(original)
    order = np.array([2, 0, 1])
    for name in (
        "history",
        "observed",
        "static",
        "role",
        "side",
        "xy",
        "velocity",
        "ids",
        "horizon",
    ):
        changed[name] = changed[name][order]
    changed["player"] = np.argsort(order)[changed["player"]]
    model = GroupedMotionModel(32).eval()
    torch.nn.init.normal_(model.decode[-1].weight, std=0.02)
    with torch.no_grad():
        expected = model(observed_batch([original]))
        permuted = model(observed_batch([changed]))
        padded = model(observed_batch([original, fixture(5)]))
    for key in expected:
        torch.testing.assert_close(expected[key], permuted[key], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(expected[key], padded[key][:1], atol=2e-6, rtol=2e-5)


def test_masked_observations_cannot_change_predictions():
    s = fixture()
    s["observed"][:, :5] = False
    first = observed_batch([s])
    s["history"][:, :5] = np.nan
    second = observed_batch([s])
    model = GroupedMotionModel(32).eval()
    with torch.no_grad():
        for key in model(first):
            torch.testing.assert_close(model(first)[key], model(second)[key], rtol=0, atol=0)


def test_velocity_labels_match_analytical_motion_and_gap_masks():
    s = fixture(frames=(1, 2, 4, 5, 94))
    _, labels = supervised_batch([s])
    np.testing.assert_array_equal(labels["velocity_mask"][0], [True, True, False, True, False])
    actual = labels["velocity"][0][labels["velocity_mask"][0]].numpy()
    np.testing.assert_allclose(actual, [[3, -2]] * 3, atol=1e-5)


def test_stale_endpoint_not_used_as_a_velocity_predecessor():
    s = fixture(frames=(1, 2, 4))
    s["static"][0, STATIC_NAMES.index("last_observation_age")] = 0.1
    _, labels = supervised_batch([s])
    np.testing.assert_array_equal(labels["velocity_mask"][0], [False, True, False])


def test_reflection_transforms_both_coordinate_and_velocity_labels():
    s = fixture(frames=(1, 2, 3))
    first_features, first_labels = supervised_batch([s])
    reflected_features, reflected_labels = supervised_batch([reflect(s)])
    for key in ("coordinate", "velocity"):
        expected = first_labels[key] * torch.tensor([1, -1])
        torch.testing.assert_close(expected, reflected_labels[key], atol=1e-5, rtol=1e-5)
    restored = observed_batch([reflect(reflect(s))])
    for key in first_features:
        torch.testing.assert_close(first_features[key], restored[key], atol=1e-5, rtol=1e-5)
    assert reflected_features["time"].shape == first_features["time"].shape


def test_shared_scale_uses_only_train_and_supported_coordinates():
    s = fixture(frames=(1, 2, 4, 5))
    scale = fit_velocity_scale([s])
    assert scale["rms"] == pytest.approx(np.sqrt(6.5), abs=1e-5)
    assert scale["coordinates"] == 6
    s["split"] = np.array("validation")
    with pytest.raises(ValueError, match="training samples only"):
        fit_velocity_scale([s])


def test_scale_matches_maintained_target_contract():
    s = fixture(frames=(1, 2, 3, 4))
    values = motion_targets(s)["velocity"]
    assert fit_velocity_scale([s])["rms"] == pytest.approx(np.sqrt(np.mean(values**2)))


def test_no_supported_scale_rejected():
    with pytest.raises(ValueError, match="supported training"):
        fit_velocity_scale([fixture(frames=(94,))])


def test_coordinate_loss_is_official_coordinate_mse_and_not_euclidean_mean():
    features, labels = supervised_batch([fixture()])
    output = {key: torch.zeros_like(labels["coordinate"]) for key in ("coordinate", "velocity")}
    loss, stats = supervised_loss(output, labels, features["scored"], 0, 3)
    assert float(loss) == pytest.approx(float(labels["coordinate"].square().mean()))
    assert stats["coordinates"] == 12


def test_fixed_denominators_give_uniform_coordinates_across_unequal_batches():
    x = [fixture(frames=(1,)), fixture(frames=(1, 2, 3, 4))]
    all_features, all_labels = supervised_batch(x)
    zero = {key: torch.zeros_like(all_labels["coordinate"]) for key in ("coordinate", "velocity")}
    total, _ = supervised_loss(zero, all_labels, all_features["scored"], 0, 3)
    pieces = []
    for s in x:
        features, labels = supervised_batch([s])
        output = {key: torch.zeros_like(labels["coordinate"]) for key in ("coordinate", "velocity")}
        loss, _ = supervised_loss(output, labels, features["scored"], 0, 3, 10 / 2)
        pieces.append(float(loss))
    assert np.mean(pieces) == pytest.approx(float(total))


def test_padding_cannot_change_either_loss_even_with_nan_labels():
    features, labels = supervised_batch([fixture(frames=(1,)), fixture()])
    output = GroupedMotionModel(32).eval()(features)
    before, _ = supervised_loss(output, labels, features["scored"], 0.1, 3)
    for key in ("coordinate", "velocity"):
        labels[key][~features["scored"]] = float("nan")
    after, _ = supervised_loss(output, labels, features["scored"], 0.1, 3)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_control_never_reads_auxiliary_labels_or_updates_auxiliary_head():
    features, labels = supervised_batch([fixture()])
    labels.pop("velocity")
    labels.pop("velocity_mask")
    state = MatchedState(32, 2026, 0, 3)
    before = copy.deepcopy(state.model.velocity_head.state_dict())
    state.step(features, labels)
    assert state.model.velocity_head.weight.grad is None
    for key in before:
        torch.testing.assert_close(
            before[key], state.model.velocity_head.state_dict()[key], rtol=0, atol=0
        )


def test_treatment_has_nonzero_velocity_and_encoder_gradients():
    features, labels = supervised_batch([fixture()])
    state = MatchedState(32, 2026, 0.1, 3)
    state.step(features, labels)
    assert float(state.model.velocity_head.weight.grad.abs().sum()) > 0
    assert float(state.model.temporal[0].weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("weight,scale", [(0.2, 3), (0.1, 0), (0.1, float("inf"))])
def test_invalid_loss_settings_rejected(weight, scale):
    features, labels = supervised_batch([fixture()])
    with pytest.raises(ValueError):
        supervised_loss(GroupedMotionModel(32)(features), labels, features["scored"], weight, scale)


def test_zero_velocity_support_is_zero_auxiliary_loss_not_a_nan():
    features, labels = supervised_batch([fixture(frames=(94,))])
    output = GroupedMotionModel(32).eval()(features)
    coordinate, _ = supervised_loss(output, labels, features["scored"], 0, 3)
    auxiliary, stats = supervised_loss(output, labels, features["scored"], 0.1, 3)
    torch.testing.assert_close(coordinate, auxiliary, rtol=0, atol=0)
    assert stats["velocity_coordinates"] == 0


def batch_for_cursor(cursor):
    # Stateless order/reflection fixture: the same cursor produces the same batch.
    originals = [fixture(3), fixture(4, frames=(1, 2, 3, 4))]
    ordered = originals if cursor % 2 == 0 else list(reversed(originals))
    if cursor % 3 == 1:
        ordered = [reflect(s) for s in ordered]
    return supervised_batch(ordered)


def resume_worker(source, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    state = MatchedState.restore(load_generation(Path(source), SIGNATURE))
    for cursor in range(state.steps, 6):
        state.step(*batch_for_cursor(cursor))
    save_generation(Path(output), state.payload(), SIGNATURE)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_tree_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_fresh_process_resume_matches_uninterrupted_model_ema_optimizer_rng_and_cursor(
    tmp_path, weight
):
    state = MatchedState(32, 2026, weight, 3)
    for cursor in range(2):
        state.step(*batch_for_cursor(cursor))
    save_generation(tmp_path / "original", state.payload(), SIGNATURE)
    shutil.copytree(tmp_path / "original", tmp_path / "independent_copy")
    shutil.rmtree(tmp_path / "original")
    for cursor in range(2, 6):
        state.step(*batch_for_cursor(cursor))
    expected = copy.deepcopy(state.payload())
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "tests"), str(root / "src")])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from test_motion_supervision import resume_worker; "
            "import sys; resume_worker(sys.argv[1], sys.argv[2])",
            str(tmp_path / "independent_copy"),
            str(tmp_path / "resumed"),
        ],
        capture_output=True,
        text=True,
        timeout=40,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    actual = load_generation(tmp_path / "resumed", SIGNATURE)
    assert_tree_equal(expected, actual)


def test_checkpoint_rejects_changed_signature_and_corrupted_bytes(tmp_path):
    state = MatchedState(32, 2026, 0, 3)
    receipt = save_generation(tmp_path, state.payload(), SIGNATURE)
    with pytest.raises(ValueError, match="signature"):
        load_generation(tmp_path, "b" * 64)
    path = tmp_path / (receipt["sha256"] + ".pt")
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash/size"):
        load_generation(tmp_path, SIGNATURE)


def test_checkpoint_publish_failure_keeps_previous_committed_generation(tmp_path, monkeypatch):
    import nfl_trajectory.supervision_evidence as evidence

    state = MatchedState(32, 2026, 0, 3)
    save_generation(tmp_path, state.payload(), SIGNATURE)
    state.step(*batch_for_cursor(0))
    original = evidence._atomic

    def fail_pointer(path, data):
        if path.name == "checkpoint.json":
            raise OSError("simulated publication interruption")
        original(path, data)

    monkeypatch.setattr(evidence, "_atomic", fail_pointer)
    with pytest.raises(OSError, match="publication interruption"):
        save_generation(tmp_path, state.payload(), SIGNATURE)
    assert load_generation(tmp_path, SIGNATURE)["steps"] == 0


def test_checkpoint_refuses_different_experiment_and_regressed_cursor(tmp_path):
    state = MatchedState(32, 2026, 0, 3)
    zero = copy.deepcopy(state.payload())
    state.step(*batch_for_cursor(0))
    save_generation(tmp_path, state.payload(), SIGNATURE)
    with pytest.raises(ValueError, match="different experiment"):
        save_generation(tmp_path, state.payload(), "c" * 64)
    with pytest.raises(ValueError, match="newer committed"):
        save_generation(tmp_path, zero, SIGNATURE)


def test_checkpoint_rejects_changed_runtime(tmp_path):
    state = MatchedState(32, 2026, 0, 3)
    save_generation(tmp_path, state.payload(), SIGNATURE)
    path = tmp_path / "checkpoint.json"
    receipt = json.loads(path.read_text())
    receipt["runtime"]["torch"] = "other"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="runtime changed"):
        load_generation(tmp_path, SIGNATURE)


def test_checkpoint_never_claims_remote_durability(tmp_path):
    state = MatchedState(32, 2026, 0, 3)
    receipt = save_generation(tmp_path, state.payload(), SIGNATURE)
    assert receipt["remote_verified"] is False


def test_short_synthetic_learning_is_not_only_a_forward_smoke():
    features, labels = supervised_batch([fixture(frames=(1, 2, 3, 4))])
    state = MatchedState(32, 7, 0.1, 3)
    first = state.step(features, labels)["coordinate_sse"]
    for _ in range(23):
        last = state.step(features, labels)["coordinate_sse"]
    assert last < first * 0.5


def test_full_width_model_forward_and_step():
    state = MatchedState(96, 2026, 0.1, 3)
    stats = state.step(*supervised_batch([fixture(3, frames=(1, 2, 94))]))
    assert stats["steps"] == 1
    assert np.isfinite(stats["loss"])
