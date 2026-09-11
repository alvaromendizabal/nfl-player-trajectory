"""Bitwise resume across epoch boundaries with actual batch and LR contracts."""

import copy

import pytest
from test_supervision_batches import samples

torch = pytest.importorskip("torch")
from nfl_trajectory.motion_supervision import MatchedState, supervised_batch  # noqa: E402
from nfl_trajectory.supervision_batches import (  # noqa: E402
    TrainingBatches,
    scheduled_learning_rate,
)
from nfl_trajectory.supervision_plan import training_plan  # noqa: E402


def advance(state, batches, plan, stop):
    while state.steps < stop:
        features, labels = supervised_batch(batches.batch(state.steps))
        for group in state.optimizer.param_groups:
            group["lr"] = scheduled_learning_rate(state.steps, 9, 2)
        state.step(
            features, labels, plan["coordinate_loss_denominator"], plan["velocity_loss_denominator"]
        )


@pytest.mark.parametrize("weight", [0.0, 0.1])
def test_exact_cross_epoch_restart(weight):
    from test_motion_supervision import assert_tree_equal

    old_threads = torch.get_num_threads()
    old_det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        values = samples(5)
        plan = training_plan(values, 2)
        first = MatchedState(32, 2026, weight, plan["training_only_velocity_rms"])
        advance(first, TrainingBatches(values, 2), plan, 4)
        saved = copy.deepcopy(first.payload())
        advance(first, TrainingBatches(values, 2), plan, 9)
        resumed = MatchedState.restore(saved)
        advance(resumed, TrainingBatches(list(reversed(values)), 2), plan, 9)
        assert_tree_equal(first.payload(), resumed.payload())
        assert first.coordinates == 3 * 2 * sum(len(s["keys"]) for s in values)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_det)
