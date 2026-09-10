"""Optional locked-runtime tests for train-only screening and residual learning."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nfl_trajectory.domain_probe import Probe, arm_mask, screen, transform  # noqa: E402


def test_train_only_screen_and_post_standardization_ablation():
    train = np.array([[1, 2, 2, 8], [2, 3, 3, 8], [3, 4, 4, 8]], dtype=np.float32)
    names = ["base", "speed", "copy", "constant"]
    families = ["control", "motion_state", "arrival_constraints", "field_geometry"]
    fitted = screen(train, names, families)
    assert fitted["retained"] == [True, True, False, False]
    evaluation = np.array([[100, 500, -10, 600]], dtype=np.float32)
    actual = transform(evaluation, fitted) * arm_mask(families, fitted["retained"], "control")
    assert (actual[:, 1:] == 0).all()
    assert fitted["mean"] == [2.0, 3.0, 3.0, 8.0]


def test_probe_can_learn_conditional_opposite_corrections_and_zero_time():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = Probe(4)
    x = torch.zeros((64, 4))
    x[:, 0] = torch.linspace(-1, 1, 64)
    seconds = torch.ones(64)
    target = torch.stack([x[:, 0], -x[:, 0]], dim=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = (model(x, seconds) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.001
    np.testing.assert_array_equal(model(x, seconds * 0).detach().numpy(), np.zeros((64, 2)))


def test_interrupted_probe_resumes_exactly_and_completed_checkpoint_is_reused(
    tmp_path, monkeypatch
):
    from nfl_trajectory import domain_probe
    from nfl_trajectory.runtime import Run, sha256

    rng = np.random.default_rng(14)
    x = rng.normal(size=(96, 4)).astype(np.float32)
    seconds = rng.uniform(0.1, 3, len(x)).astype(np.float32)
    y = x[:, :2] * seconds[:, None]
    with Run(tmp_path, "uninterrupted-probe") as run:
        complete, _ = domain_probe.train_probe(
            x, y, seconds, tmp_path / "complete", "test", run, "control", "test"
        )
    save = domain_probe.save_checkpoint

    def interrupted(path, payload, signature):
        save(path, payload, signature)
        if payload["epoch"] == 4:
            raise RuntimeError("Simulated process interruption after durable checkpoint")

    monkeypatch.setattr(domain_probe, "save_checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="Simulated process interruption"):
        with Run(tmp_path, "interrupted-probe") as run:
            domain_probe.train_probe(
                x, y, seconds, tmp_path / "resume", "test", run, "control", "test"
            )
    monkeypatch.setattr(domain_probe, "save_checkpoint", save)
    with Run(tmp_path, "resumed-probe") as run:
        resumed, state = domain_probe.train_probe(
            x, y, seconds, tmp_path / "resume", "test", run, "control", "test"
        )
    assert state["epoch"] == 12
    for name, expected in complete.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name])
    path = tmp_path / "resume/checkpoint.pt"
    before = (sha256(path), path.stat().st_mtime_ns)
    with Run(tmp_path, "completed-probe") as run:
        domain_probe.train_probe(x, y, seconds, tmp_path / "resume", "test", run, "control", "test")
    assert (sha256(path), path.stat().st_mtime_ns) == before
