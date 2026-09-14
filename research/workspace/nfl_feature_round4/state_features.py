"""Observed-state and landing-frame features. No future coordinates or fitted statistics.

This is a representation study, not a claim of new raw information. Legacy state
was multiplied by query time plus observation age. Expose the same observed state
separately, then test compact geometry relative to the organizer-supplied landing
point. All formulas use canonical yards and seconds, with explicit support flags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from origin_features import STATE_NAMES, ALL_NAMES, build_view, query_features, KEYS

DIRECT_NAMES = tuple('observed__' + n for n in STATE_NAMES)
GOAL_NAMES = (
    'goal_distance', 'goal_unit_x', 'goal_unit_y',
    'speed_toward_goal', 'speed_across_goal', 'speed', 'heading_alignment',
    'required_average_speed', 'radial_speed_shortfall',
    'cv_distance_at_query', 'cv_distance_at_endpoint',
    'cv_endpoint_radial_miss', 'cv_endpoint_lateral_miss',
    'closest_approach_time', 'closest_approach_distance',
    'receiver_distance', 'receiver_along_goal', 'receiver_across_goal',
    'receiver_radial_velocity_gap', 'receiver_lateral_velocity_gap',
    'receiver_position_present', 'receiver_velocity_present',
    'velocity_present', 'goal_axis_present',
    'closest_approach_present', 'heading_present',
)
GOAL_UNITS = (
    'yards / 20','unitless','unitless','yards/second / 10','yards/second / 10',
    'yards/second / 10','unitless','yards/second / 10','yards/second / 10',
    'yards / 20','yards / 20','yards / 20','yards / 20','seconds / 5','yards / 20',
    'yards / 20','yards / 20','yards / 20','yards/second / 10','yards/second / 10',
    'boolean','boolean','boolean','boolean','boolean','boolean',
)
GOAL_ODD = tuple(GOAL_NAMES.index(n) for n in (
    'goal_unit_y','speed_across_goal','cv_endpoint_lateral_miss',
    'receiver_across_goal','receiver_lateral_velocity_gap'))
STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1] - a[1]*b[0])


def storage_parity(rebuilt: np.ndarray, saved: np.ndarray) -> None:
    """Repeat each shard's recorded float32/float64 storage, then compare exactly."""
    a, b = np.asarray(rebuilt), np.asarray(saved)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != len(ALL_NAMES) or not len(a):
        raise ValueError('Legacy feature shape/row mismatch')
    if a.dtype != np.dtype('float64') or b.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Unsupported legacy storage dtype')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite legacy feature')
    with np.errstate(over='raise', invalid='raise'):
        candidate = a.astype(b.dtype)
    if not np.array_equal(candidate, b):
        raise ValueError('Raw adapter differs AFTER historical storage conversion; do not relax parity')


def build_features(raw: pd.DataFrame, keys: np.ndarray, *, cutoff: int) -> dict[str, np.ndarray]:
    """Return row-aligned observed inputs; `keys` supplies only forecast requests.

    No target values, prediction errors, player priors or model predictions are
    accepted. Receiver-relative positions are simultaneous observed positions;
    they are not inferred coverage assignments. No player is required to reach
    the landing point. Constant-velocity projections are hypotheses, not truths.
    """
    q = np.asarray(keys)
    if q.ndim != 2 or q.shape[1] != 4 or not len(q) or not np.issubdtype(q.dtype, np.integer):
        raise ValueError('Nonempty integer [rows,4] query keys required')
    if len(np.unique(q, axis=0)) != len(q):
        raise ValueError('Duplicate query keys')
    if not isinstance(cutoff, (int, np.integer)) or int(cutoff) < 1:
        raise ValueError('Positive integer observed cutoff required')
    if raw.empty or raw.frame_id.max() > cutoff:
        raise ValueError('Future observed rows reached feature construction')
    pairs = raw[KEYS[:2]].drop_duplicates().to_numpy(np.int64)
    if pairs.shape != (1,2) or not (q[:,:2] == pairs[0]).all():
        raise ValueError('Query keys belong to a different play')
    view = build_view(raw, origin=int(cutoff), offset=0)
    base, _ = query_features(view, q[:,2], q[:,3].astype(float)/10)
    rows, geometry = [], []
    for ident_raw, frame in q[:,2:]:
        ident = int(ident_raw)
        if ident not in view.scored:
            raise ValueError('A query player is not an organizer-designated scored player')
        state = view.states[ident]
        rows.append(state)
        t = int(frame)/10
        tau = t + view.ages[ident]
        total = view.horizons[ident] + view.ages[ident]
        if total <= 0 or tau <= 0:
            raise ValueError('Invalid observation/query timing')
        b = view.balls[ident]
        v = view.velocities[ident]
        distance, speed = float(np.linalg.norm(b)), float(np.linalg.norm(v))
        axis_ok = distance >= 1e-6
        velocity_ok = bool(state[STATE_INDEX['vx_observed']])
        heading_ok = axis_ok and velocity_ok and speed >= 1e-3
        closest_ok = velocity_ok and speed >= 1e-3
        unit = b/distance if axis_ok else np.zeros(2)
        toward = float(v @ unit) if axis_ok and velocity_ok else 0.0
        lateral = cross(unit,v) if axis_ok and velocity_ok else 0.0
        miss_query, miss_end = b-v*tau, b-v*total
        closest_time = float(np.clip((b@v)/(speed*speed),0,total)) if closest_ok else 0.0
        receiver_position = bool(state[STATE_INDEX['receiver_joint']])
        receiver_velocity = bool(view.receiver_valid[ident] and velocity_ok)
        rr = view.receiver_relative[ident]
        dv = view.receiver_velocity[ident]-v
        required_speed = distance/total
        g = [
            distance/20, unit[0], unit[1], toward/10, lateral/10,
            speed/10 if velocity_ok else 0.0,
            toward/speed if heading_ok else 0.0, required_speed/10,
            (required_speed-toward)/10 if axis_ok and velocity_ok else 0.0,
            np.linalg.norm(miss_query)/20 if velocity_ok else 0.0,
            np.linalg.norm(miss_end)/20 if velocity_ok else 0.0,
            (miss_end@unit)/20 if axis_ok and velocity_ok else 0.0,
            cross(unit,miss_end)/20 if axis_ok and velocity_ok else 0.0,
            closest_time/5,
            np.linalg.norm(b-v*closest_time)/20 if closest_ok else 0.0,
            np.linalg.norm(rr)/20 if receiver_position else 0.0,
            (rr@unit)/20 if receiver_position and axis_ok else 0.0,
            cross(unit,rr)/20 if receiver_position and axis_ok else 0.0,
            (dv@unit)/10 if receiver_velocity and axis_ok else 0.0,
            cross(unit,dv)/10 if receiver_velocity and axis_ok else 0.0,
            float(receiver_position), float(receiver_velocity), float(velocity_ok),
            float(axis_ok), float(closest_ok), float(heading_ok),
        ]
        geometry.append(g)
    state_array = np.asarray(rows,dtype=np.float64)
    goal_array = np.asarray(geometry,dtype=np.float64)
    if state_array.shape != (len(q),len(DIRECT_NAMES)) or goal_array.shape != (len(q),len(GOAL_NAMES)):
        raise ValueError('Feature schema mismatch')
    if not np.isfinite(state_array).all() or not np.isfinite(goal_array).all():
        raise ValueError('Nonfinite observed features')
    return {'state':state_array,'goal':goal_array,'legacy':base,'keys':q.copy()}
