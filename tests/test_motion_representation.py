"""Motion normalization, initial parity and actual interrupted-training recovery."""

import numpy as np
import pytest
from test_temporal_data import sample

torch = pytest.importorskip("torch")

from nfl_trajectory.motion_representation import (  # noqa: E402
    SETTINGS,
    MotionModel,
    attach_motion,
    collate_motion,
    fit_motion_screen,
    motion_state,
    reflect_motion,
    train,
)
from nfl_trajectory.runtime import Run, sha256  # noqa: E402
from nfl_trajectory.temporal_data import reflect  # noqa: E402
from nfl_trajectory.temporal_model import TemporalModel, collate  # noqa: E402


def examples():
    values = []
    for i in range(4):
        s = sample(i + 3)
        s["split"] = np.array("train")
        s["keys"] = s["keys"].copy()
        s["keys"][:, 1] = i + 1
        values.append(attach_motion(s))
    names = motion_state(values[0])[1]
    return values, names, fit_motion_screen(values, names)


def test_motion_reflection_and_training_only_screen():
    values, names, fitted = examples()
    original = values[0]
    odd = np.array([n.endswith("_y") for n in names])
    np.testing.assert_allclose(
        reflect_motion(original, odd)["motion"], motion_state(reflect(original))[0], atol=2e-6
    )
    changed = {k: v.copy() for k, v in original.items()}
    changed["truth"] *= 1e6
    changed["keys"][:] = 99999
    np.testing.assert_array_equal(motion_state(changed)[0], original["motion"])
    changed["split"] = np.array("validation")
    changed["motion"][:] = 1e9
    assert fit_motion_screen([*values, changed], names) == fitted


def test_zero_initialized_feature_branch_preserves_pretrained_predictions():
    torch.set_num_threads(2)
    values, _, fitted = examples()
    base = TemporalModel(32).eval()
    torch.nn.init.normal_(base.decode[-1].weight, std=0.03)
    control = MotionModel(base.state_dict(), fitted, False, 32).eval()
    feature = MotionModel(base.state_dict(), fitted, True, 32).eval()
    batch = collate_motion(values)
    with torch.no_grad():
        reference = base(collate(values))
        torch.testing.assert_close(control(batch), reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(feature(batch), reference, rtol=1e-5, atol=1e-6)
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in feature.parameters()
    )
    with torch.no_grad():
        feature.temporal[0].weight[:, 30:60].fill_(0.1)
        batch["motion"][:, :, 0] += 2
        assert not torch.allclose(feature(batch), reference)
        torch.testing.assert_close(control(batch), reference, rtol=1e-5, atol=1e-6)


def test_exact_interrupted_and_completed_training_recovery(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    values, names, fitted = examples()
    initial = TemporalModel(32).state_dict()
    config = {**SETTINGS, "epochs": 3, "batch_plays": 2, "threads": 2}

    def model():
        return MotionModel(initial, fitted, True, 32)

    with Run(tmp_path, "continuous") as run:
        expected, state = train(
            model(), values, names, tmp_path / "continuous", "test", run, config
        )
    with Run(tmp_path, "interrupted") as run, pytest.raises(KeyboardInterrupt):
        train(
            model(),
            values,
            names,
            tmp_path / "resumed",
            "test",
            run,
            config,
            interrupt_after_batches=3,
        )
    with Run(tmp_path, "resumed") as run:
        actual, resumed = train(model(), values, names, tmp_path / "resumed", "test", run, config)
    assert state["curve"] == resumed["curve"]
    for name, expected_value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected_value, rtol=0, atol=0)
    path = tmp_path / "resumed/checkpoint.pt"
    before = (sha256(path), path.stat().st_mtime_ns)
    with Run(tmp_path, "completed_reuse") as run:
        train(model(), values, names, tmp_path / "resumed", "test", run, config)
    assert before == (sha256(path), path.stat().st_mtime_ns)
