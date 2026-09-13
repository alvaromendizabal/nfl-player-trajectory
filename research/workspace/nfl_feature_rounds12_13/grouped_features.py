"""Two matched views of the existing Round 9 goal-relative information.

Both arms share all masks, ages, groups, timing and eight Cartesian channels.
Only the six added numerical goal-frame values are withheld from the control.
No future-coordinate array is accessed by either feature function.
"""
from __future__ import annotations
import numpy as np
from goal_frame import build, NAMES, GROUPS

ARMS = ('cartesian', 'goal')


def make_extension(sample: dict) -> dict[str, np.ndarray]:
    g = build(sample)
    present = g['valid']
    slots = np.arange(20)[None, None, :, None]
    last = np.where(present, slots, -1).max(2)
    age = np.where(last >= 0, (19-last)/10.0, 2.0).astype(np.float32)
    return {'goal': g['values'], 'goal_valid': present, 'groups': g['groups'], 'goal_age': age}


def pair_view(sample: dict, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    x = np.asarray(sample['goal']); valid = np.asarray(sample['goal_valid'])
    if valid.dtype != np.bool_ or x.shape != valid.shape or x.shape[-2:] != (20,6):
        raise ValueError('Invalid goal feature/mask contract')
    if not np.isfinite(x[valid]).all():
        raise ValueError('Nonfinite available goal feature')
    goal = np.where(valid, x, 0) if arm == 'goal' else np.zeros_like(x)
    v = np.concatenate((sample['pair_valid'], valid), -1)
    p = np.concatenate((np.where(sample['pair_valid'],sample['pair'],0), goal), -1).astype(np.float32)
    age = np.concatenate((sample['pair_age'], sample['goal_age']), -1).astype(np.float32)
    if not np.isfinite(p).all() or not np.isfinite(age).all():
        raise ValueError('Nonfinite pair input')
    return p, v, age
