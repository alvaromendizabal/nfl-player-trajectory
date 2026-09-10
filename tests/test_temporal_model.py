"""Run under the locked CPU torch environment; no competition data or network needed."""

import numpy as np
import pytest
from test_temporal_data import sample

torch = pytest.importorskip("torch")

from nfl_trajectory.temporal_model import (  # noqa: E402
    TemporalModel,
    collate,
    coordinate_loss,
    load_checkpoint,
    save_checkpoint,
)


def test_player_permutation_and_padding_do_not_change_predictions():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    model = TemporalModel(32).eval()
    torch.nn.init.normal_(model.decode[-1].weight, std=0.03)
    original = sample(3)
    changed = {k: v.copy() for k, v in original.items()}
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
    with torch.no_grad():
        expected = model(collate([original]))
        permuted = model(collate([changed]))
        padded = model(collate([original, sample(5)]))[:1]
    torch.testing.assert_close(expected, permuted, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(expected, padded, atol=1e-6, rtol=1e-5)


def test_padding_never_changes_official_coordinate_loss():
    batch = collate([sample()])
    zero = torch.zeros_like(batch["truth"])
    actual = coordinate_loss(zero, batch)
    expected = np.mean(sample()["truth"] ** 2)
    assert actual.item() == pytest.approx(expected)
    batch["scored"][:, -1] = False
    before = coordinate_loss(zero, batch)
    batch["truth"][:, -1] = 1e8
    torch.testing.assert_close(before, coordinate_loss(zero, batch))


def test_learning_and_exact_optimizer_rng_recovery(tmp_path):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2026)
    model = TemporalModel(32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    batch = collate([sample()])

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = coordinate_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    first = step()
    for _ in range(24):
        last = step()
    assert last < first * 0.25
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        "test",
    )
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    saved = load_checkpoint(path, "test")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="signature or hash"):
        load_checkpoint(path, "wrong")
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="signature or hash"):
        load_checkpoint(path, "test")


def test_model_learns_opposite_role_conditioned_motion_not_only_a_bias():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    first = sample()
    second = {name: value.copy() for name, value in sample().items()}
    second["role"][0] = 2
    second["truth"] *= -1
    batch = collate([first, second])
    model = TemporalModel(32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    initial = float(coordinate_loss(model(batch), batch).detach())
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = coordinate_loss(model(batch), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
    assert float(coordinate_loss(prediction, batch)) < initial * 0.15
    assert float(prediction[0, -1, 0]) > 0
    assert float(prediction[1, -1, 0]) < 0
