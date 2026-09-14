"""Observed passer-to-landing coordinates, with explicit support and time offsets.

No future coordinates, fitted encodings, residuals, or predicted coverage labels
enter this interface. The passing axis is a geometric reference, NOT the actual
ball path. An absent or coincident passer/landing point is masked, never imputed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from origin_features import KEYS, validate_raw, canonical_xy, velocity, flags

OFFSETS = (5, 10, 15)
STATIC_NAMES = (
    'pass_length', 'player_along_fraction', 'player_across', 'ball_along',
    'distance_to_passer', 'velocity_along', 'velocity_across',
    'orientation_along', 'orientation_across',
    'receiver_gap_along', 'receiver_gap_across',
    'receiver_velocity_gap_along', 'receiver_velocity_gap_across',
    'signed_triangle_area', 'behind_passer', 'beyond_landing',
    'axis_present', 'velocity_present', 'orientation_present',
    'receiver_present', 'receiver_velocity_present',
)
HISTORY_FIELDS = ('displacement_along', 'displacement_across',
                  'velocity_change_along', 'velocity_change_across',
                  'receiver_gap_change_along', 'receiver_gap_change_across',
                  'position_endpoints_present', 'velocity_endpoints_present',
                  'receiver_endpoints_present')
HISTORY_NAMES = tuple(f'lag{k:02d}__{n}' for k in OFFSETS for n in HISTORY_FIELDS)
NAMES = STATIC_NAMES + HISTORY_NAMES
ODD_STATIC = tuple(STATIC_NAMES.index(n) for n in ('player_across', 'velocity_across',
    'orientation_across', 'receiver_gap_across', 'receiver_velocity_gap_across', 'signed_triangle_area'))
ODD_HISTORY = tuple(j*len(HISTORY_FIELDS)+i for j in range(len(OFFSETS)) for i in (1, 3, 5))


def cross(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0]*v[1] - u[1]*v[0])


def build_features(raw: pd.DataFrame, keys: np.ndarray, *, cutoff: int) -> dict:
    """Return [forecast rows,21] static and [forecast rows,27] lagged values.

    Landmarks are observed simultaneously with each player's final observation.
    Earlier positions use exact observed frame IDs, not row offsets. Endpoint
    changes are not instantaneous derivatives; missing endpoints are masked.
    The terminal axis is known at prediction time and is held fixed for its
    history, so a moving frame cannot manufacture apparent player motion.
    """
    validate_raw(raw)
    q = np.asarray(keys)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q) or not np.issubdtype(q.dtype, np.integer):
        raise ValueError('Nonempty integer [rows,4] forecast keys required')
    if len(np.unique(q, axis=0)) != len(q):
        raise ValueError('Duplicate forecast key')
    if not isinstance(cutoff, (int, np.integer)) or cutoff < 1 or raw.frame_id.max() > cutoff:
        raise ValueError('Invalid cutoff or future observed frame')
    pair = raw[KEYS[:2]].drop_duplicates().to_numpy(np.int64)
    if pair.shape != (1, 2) or not (q[:, :2] == pair[0]).all():
        raise ValueError('Mixed play or query identity')
    left = raw.play_direction.iloc[0] == 'left'
    by_id, landmarks = {}, {'Passer': {}, 'Targeted Receiver': {}}
    for ident, g in raw.groupby('nfl_id', sort=True):
        g = g.sort_values('frame_id')
        xy = canonical_xy(g[['x', 'y']].to_numpy(float), left)
        vel, ok = velocity(g, left)
        frames = g.frame_id.to_numpy(int)
        rows = g.to_dict('records')
        recs = {int(f): (xy[i], vel[i], bool(ok[i]), rows[i]) for i, f in enumerate(frames)}
        by_id[int(ident)] = recs
        role = rows[-1]['player_role']
        if role in landmarks:
            for f, rec in recs.items():
                if f in landmarks[role]:
                    raise ValueError('Ambiguous simultaneous role landmark')
                landmarks[role][f] = rec
    encoded = {}
    for ident in np.unique(q[:, 2]):
        ident = int(ident)
        if ident not in by_id:
            raise ValueError('Unknown query player')
        recs = by_id[ident]
        f = max(recs)
        p, v, vok, row = recs[f]
        if not flags(pd.Series([row['player_to_predict']]))[0]:
            raise ValueError('Query player is not organizer-designated')
        qs = q[q[:, 2] == ident, 3]
        if (qs < 1).any() or (qs > int(row['num_frames_output'])).any():
            raise ValueError('Query horizon exceeds organizer request')
        st, hist = np.zeros(len(STATIC_NAMES)), np.zeros(len(HISTORY_NAMES))
        passer = landmarks['Passer'].get(f)
        receiver = landmarks['Targeted Receiver'].get(f)
        ball = canonical_xy(np.array([row['ball_land_x'], row['ball_land_y']], float), left)
        if passer is not None:
            qb = passer[0]
            direction = ball - qb
            length = float(np.linalg.norm(direction))
            if length >= 0.1:
                u = direction / length
                rel = p - qb
                along = float(rel @ u)
                ori_ok = 'o' in row and pd.notna(row['o']) and np.isfinite(float(row['o']))
                ori = np.zeros(2)
                if ori_ok:
                    a = np.deg2rad(float(row['o']))
                    ori = np.array([np.sin(a), np.cos(a)]) * (-1 if left else 1)
                rvok = receiver is not None and receiver[2] and vok
                gap = receiver[0]-p if receiver is not None else np.zeros(2)
                dv = receiver[1]-v if rvok else np.zeros(2)
                st[:] = [length/20, along/length, cross(u, rel)/20,
                    float((ball-p)@u)/20, np.linalg.norm(rel)/20,
                    float(v@u)/10 if vok else 0, cross(u, v)/10 if vok else 0,
                    float(ori@u) if ori_ok else 0, cross(u, ori) if ori_ok else 0,
                    float(gap@u)/20, cross(u, gap)/20,
                    float(dv@u)/10, cross(u, dv)/10,
                    cross(rel, receiver[0]-qb)/(length*length) if receiver is not None else 0,
                    float(along < 0), float(along > length), 1., float(vok), float(ori_ok),
                    float(receiver is not None), float(rvok)]
                for j, offset in enumerate(OFFSETS):
                    old = recs.get(f-offset)
                    old_rec = landmarks['Targeted Receiver'].get(f-offset)
                    if old is None:
                        continue
                    dp = p-old[0]
                    vvok = vok and old[2]
                    dvv = v-old[1] if vvok else np.zeros(2)
                    rrok = receiver is not None and old_rec is not None
                    dr = gap-(old_rec[0]-old[0]) if rrok else np.zeros(2)
                    hist[j*9:(j+1)*9] = [float(dp@u)/10, cross(u, dp)/10,
                        float(dvv@u)/10, cross(u, dvv)/10, float(dr@u)/10,
                        cross(u, dr)/10, 1., float(vvok), float(rrok)]
        encoded[ident] = (st, hist)
    static = np.array([encoded[int(i)][0] for i in q[:, 2]], dtype=np.float64)
    history = np.array([encoded[int(i)][1] for i in q[:, 2]], dtype=np.float64)
    if not np.isfinite(static).all() or not np.isfinite(history).all():
        raise ValueError('Nonfinite pass-axis feature')
    return {'static': static, 'history': history, 'keys': q.copy()}
