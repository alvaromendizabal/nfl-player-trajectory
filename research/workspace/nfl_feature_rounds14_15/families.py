"""Two 12-channel observed-only families. Prepared code; execution is user-owned.

14: goal-aligned motion. 15: receiver-relative motion response.
All arithmetic uses current/past observed slots. No target arguments, fitted
statistics, interpolation, clipping, or synthetic coverage assignments.
"""
from __future__ import annotations
import numpy as np
from motion_reference import build as reference_motion

DT = 0.1
ARMS = ('mask', 'core', 'full')
NAMES = {
    14: ('goal_parallel_acceleration', 'goal_lateral_acceleration',
         'goal_parallel_jerk', 'goal_lateral_jerk',
         'velocity_goal_alignment', 'velocity_goal_lateral_alignment',
         'goal_parallel_acceleration_trailing3', 'goal_lateral_acceleration_trailing3',
         'goal_parallel_acceleration_trailing5', 'goal_lateral_acceleration_trailing5',
         'closing_speed_change_rate', 'lateral_goal_speed_change_rate'),
    15: ('receiver_relative_ax', 'receiver_relative_ay',
         'receiver_closing_acceleration', 'receiver_lateral_acceleration',
         'receiver_turn_rate_difference', 'receiver_speed_change_difference',
         'receiver_relative_ax_trailing3', 'receiver_relative_ay_trailing3',
         'receiver_relative_ax_trailing5', 'receiver_relative_ay_trailing5',
         'receiver_relative_jerk_x', 'receiver_relative_jerk_y'),
}
SCALES = {
    14: np.array([20, 20, 100, 100, 1, 1, 20, 20, 20, 20, 20, 20], float),
    15: np.array([20, 20, 20, 20, 10, 20, 20, 20, 20, 20, 100, 100], float),
}
UNITS = {
    14: ('yards/s^2',)*2 + ('yards/s^3',)*2 + ('cosine','signed sine') + ('yards/s^2',)*6,
    15: ('yards/s^2',)*4 + ('rad/s','yards/s^2') + ('yards/s^2',)*4 + ('yards/s^3',)*2,
}
ODD = {14: (1, 3, 5, 7, 9, 11), 15: (1, 3, 4, 7, 9, 11)}


def _round(round_no: int) -> None:
    if type(round_no) is not int or round_no not in NAMES:
        raise ValueError('Round must be 14 or 15')


def difference(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-slot backward difference; both endpoints must exist."""
    if x.shape != valid.shape or valid.dtype != np.bool_ or x.ndim < 2:
        raise ValueError('Difference requires aligned values and boolean masks')
    y = np.zeros_like(x, dtype=np.float64)
    m = np.zeros_like(valid)
    clean = np.where(valid, x, 0).astype(np.float64)
    y[:, 1:] = np.diff(clean, axis=1) / DT
    m[:, 1:] = valid[:, 1:] & valid[:, :-1]
    return np.where(m, y, 0), m


def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]


def build(sample: dict, round_no: int) -> dict[str, np.ndarray]:
    """Return 12 candidate values and masks. Round 13 motion is a separate baseline."""
    _round(round_no)
    # Validates observed layouts, finite supported fields, roles and self exclusion.
    ref = reference_motion(sample, 13)
    node, nv = np.asarray(sample['node']), np.asarray(sample['node_valid'])
    n, t = node.shape[:2]
    values = np.zeros((n, t, 12), np.float64)
    mask = np.zeros_like(values, bool)
    acceleration = ref['values'][..., :2].astype(float) * 20
    acceleration_ok = ref['valid'][..., :2].all(-1)
    jerk = ref['values'][..., 10:12].astype(float) * 100
    jerk_ok = ref['valid'][..., 10:12].all(-1)
    if round_no == 14:
        goal = np.where(nv[..., 4:6], node[..., 4:6], 0).astype(float)*20
        distance = np.linalg.norm(goal, axis=-1)
        gok = nv[..., 4:6].all(-1) & (distance >= 0.1)
        axis = goal / np.maximum(distance, 0.1)[..., None]
        for first, vector, ok in ((0, acceleration, acceleration_ok), (2, jerk, jerk_ok)):
            values[..., first] = np.sum(axis*vector, axis=-1)
            values[..., first+1] = cross(axis, vector)
            mask[..., first:first+2] = (gok & ok)[..., None]
        vel = np.where(nv[..., 2:4], node[..., 2:4], 0).astype(float)*10
        speed = np.linalg.norm(vel, axis=-1)
        vok = nv[..., 2:4].all(-1) & gok
        moving = vok & (speed >= 0.1)
        direction = vel / np.maximum(speed, 0.1)[..., None]
        values[..., 4] = np.sum(axis*direction, axis=-1)
        values[..., 5] = cross(axis, direction)
        mask[..., 4:6] = moving[..., None]
        for dest, source in ((6, 6), (8, 8)):
            trailing = ref['values'][..., source:source+2].astype(float)*20
            ok = ref['valid'][..., source:source+2].all(-1) & gok
            # Trailing Cartesian acceleration projected onto the CURRENT goal axis.
            values[..., dest] = np.sum(axis*trailing, axis=-1)
            values[..., dest+1] = cross(axis, trailing)
            mask[..., dest:dest+2] = ok[..., None]
        for dest, x in ((10, np.sum(axis*vel, -1)), (11, cross(axis, vel))):
            values[..., dest], mask[..., dest] = difference(x, vok)
    else:
        receiver = np.flatnonzero(np.asarray(sample['role'])[:, 0] == 1)
        if len(receiver):
            j = int(receiver[0])
            p = np.asarray(sample['pair'])[:, j]
            pv = np.asarray(sample['pair_valid'])[:, j]
            dv = np.where(pv[..., 2:4], p[..., 2:4], 0).astype(float)*10
            joint = pv[..., :4].all(-1)
            acc, ok = difference(dv, np.broadcast_to(joint[..., None], dv.shape).copy())
            values[..., :2], mask[..., :2] = acc, ok
            for dest, source in ((2, 5), (3, 6)):
                x = np.where(pv[..., source], p[..., source], 0).astype(float)*10
                values[..., dest], mask[..., dest] = difference(x, pv[..., source])
            for dest, source, scale in ((4, 4, 10), (5, 5, 20)):
                v = ref['values'][..., source].astype(float)*scale
                m = ref['valid'][..., source]
                values[..., dest] = v[j][None, :] - v
                mask[..., dest] = m & m[j][None, :] & joint
            for window, start in ((3, 6), (5, 8)):
                for ti in range(window-1, t):
                    values[:, ti, start:start+2] = acc[:, ti-window+1:ti+1].mean(axis=1)
                    mask[:, ti, start:start+2] = ok[:, ti-window+1:ti+1].all(axis=1)
            values[..., 10:12], mask[..., 10:12] = difference(acc, ok)
            # No self-neighbor, even for motion-difference channels.
            values[j] = 0
            mask[j] = False
    result = np.where(mask, values/SCALES[round_no], 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Feature overflow; do not silently clip')
    return {'values': result, 'valid': mask}


def view(ext: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    x, m = np.asarray(ext['values']), np.asarray(ext['valid'])
    if x.ndim != 3 or x.shape[1:] != (20,12) or m.shape != x.shape or m.dtype != np.bool_:
        raise ValueError('Invalid family tensor')
    if not np.isfinite(x[m]).all():
        raise ValueError('Nonfinite supported feature')
    out = np.where(m, x, 0).astype(np.float32)
    if arm == 'mask':
        out[:] = 0
    elif arm == 'core':
        out[..., 6:] = 0
    return out, m.copy()


def summarize(arrays: list[dict], round_no: int) -> list[dict]:
    _round(round_no)
    rows = []
    for j, name in enumerate(NAMES[round_no]):
        den = sum(a['values'].shape[0]*20 for a in arrays)
        parts = [a['values'][...,j][a['valid'][...,j]].astype(float)*SCALES[round_no][j] for a in arrays]
        x = np.concatenate(parts) if parts else np.empty(0)
        rows.append({'name':name, 'unit':UNITS[round_no][j], 'block':'core' if j<6 else 'extension',
                     'slots':den, 'valid':len(x), 'support':len(x)/den if den else 0,
                     'rms':float(np.sqrt(np.mean(x*x))) if len(x) else None,
                     'p01':float(np.quantile(x,.01)) if len(x) else None,
                     'p99':float(np.quantile(x,.99)) if len(x) else None,
                     'max_abs':float(np.max(np.abs(x))) if len(x) else None})
    return rows
