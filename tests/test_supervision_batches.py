"""Cursor, reflection and fixed-schedule isolation contracts."""

import copy

import numpy as np
import pytest
from test_temporal_data import sample

from nfl_trajectory.supervision_batches import TrainingBatches, scheduled_learning_rate


def samples(n=7):
    values = []
    for i in range(n):
        s = {k: v.copy() for k, v in sample(3).items()}
        s["split"] = np.array("train")
        s["keys"][:, 1] = i + 1
        values.append(s)
    return values


def test_every_play_once():
    b = TrainingBatches(samples(), 3)
    for e in range(3):
        parts = [b.selection(e * 3 + j)[0] for j in range(3)]
        assert sorted(np.concatenate(parts).tolist()) == list(range(7))
        assert [len(p) for p in parts] == [3, 3, 1]


def test_restart_and_reversed_input():
    values = samples()
    a = TrainingBatches(values, 3)
    for step in range(11):
        b = TrainingBatches(list(reversed(values)), 3)
        for x, y in zip(a.batch(step), b.batch(step), strict=True):
            for key in x:
                np.testing.assert_array_equal(x[key], y[key])


def test_no_label_access_for_order():
    values = samples()
    changed = copy.deepcopy(values)
    for s in changed:
        s.pop("truth")
    a, b = TrainingBatches(values, 3), TrainingBatches(changed, 3)
    for step in range(9):
        for x, y in zip(a.selection(step), b.selection(step), strict=True):
            np.testing.assert_array_equal(x, y)
        for x, y in zip(a.batch(step), b.batch(step), strict=True):
            for key in y:
                np.testing.assert_array_equal(x[key], y[key])


def test_readonly_inputs_unchanged():
    values = samples()
    before = copy.deepcopy(values)
    for s in values:
        for arr in s.values():
            arr.flags.writeable = False
    b = TrainingBatches(values, 3)
    for step in range(9):
        b.batch(step)
    for a, b in zip(values, before, strict=True):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_validation_rejected_before_read():
    with pytest.raises(ValueError, match="evaluation"):
        TrainingBatches([{"split": "validation"}], 3)


def test_duplicate_and_negative_rejected():
    s = samples(1)[0]
    with pytest.raises(ValueError, match="Duplicate"):
        TrainingBatches([s, s], 3)
    with pytest.raises(ValueError, match="Negative"):
        TrainingBatches([s], 3).batch(-1)


def test_lr_endpoints():
    for step, expected in [(0, 0.0001), (9, 0.001), (10, 0.001), (99, 0.00001)]:
        assert scheduled_learning_rate(step, 100, 10) == pytest.approx(expected)
    assert all(scheduled_learning_rate(i, 100, 10) > 0 for i in range(100))


@pytest.mark.parametrize("cursor,total,warmup", [(-1, 100, 10), (100, 100, 10), (0, 10, 9)])
def test_bad_schedule(cursor, total, warmup):
    with pytest.raises(ValueError):
        scheduled_learning_rate(cursor, total, warmup)
