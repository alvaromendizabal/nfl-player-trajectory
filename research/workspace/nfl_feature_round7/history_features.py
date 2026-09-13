"""Strictly earlier-date, trajectory-weighted motion-response histories.

Future player positions are labels only. Training rows never see outcomes from
any game on their current date. Evaluation uses a frozen training table. IDs are
lookup keys, never model inputs. No external data, tuning, or stochastic ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import copy
import hashlib
import json
from typing import Callable

import numpy as np

VERSION = 'ordered-motion-response-v1'
CONFIG = {'phase_edges': [1 / 3, 2 / 3], 'horizon_edges_seconds': [1., 2., 3.],
          'global_smoothing_trajectories': 30., 'role_smoothing_trajectories': 20.,
          'player_smoothing_trajectories': 10., 'zero_history_rate_sd': 2.}
LEVELS = ('global_phase', 'role_phase_horizon', 'player_role_phase_horizon')
SUPPORT_NAMES = tuple(f'{level}__{name}' for level in LEVELS
                      for name in ('log1p_trajectories', 'cold_start', 'log1p_age_days'))
VALUE_NAMES = ('mean_residual_rate_x', 'mean_residual_rate_y', 'rate_sd_x', 'rate_sd_y',
               'expected_residual_x', 'expected_residual_y')
ROLE_NAMES = tuple('role_response__' + n for n in VALUE_NAMES)
PLAYER_NAMES = tuple('player_response__' + n for n in VALUE_NAMES)
NAMES = SUPPORT_NAMES + ROLE_NAMES + PLAYER_NAMES


def state_hash(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Metadata:
    keys: np.ndarray
    role: np.ndarray
    time: np.ndarray
    horizon: np.ndarray

    def take(self, index: np.ndarray) -> 'Metadata':
        return Metadata(self.keys[index].copy(), self.role[index].copy(),
                        self.time[index].copy(), self.horizon[index].copy())

    def validate(self) -> None:
        n = len(self.keys)
        if (self.keys.shape != (n, 4) or not np.issubdtype(self.keys.dtype, np.integer)
                or any(a.shape != (n,) for a in (self.role, self.time, self.horizon))):
            raise ValueError('Metadata keys must be integer [rows,4]; fields must align')
        if len(np.unique(self.keys, axis=0)) != n:
            raise ValueError('Duplicate forecast keys')
        if (not np.isin(self.role, [0, 1, 2, 3]).all()
                or any(not np.isfinite(a).all() for a in (self.time, self.horizon))
                or (self.time <= 0).any() or (self.horizon <= 0).any()
                or (self.time > self.horizon + 1e-10).any()):
            raise ValueError('Invalid role, time, or organizer horizon')
        if not np.allclose(self.time, self.keys[:, 3] / 10, rtol=0, atol=1e-10):
            raise ValueError('Query clock is not the organizer frame clock')
        if (self.keys[:, 2:] < 1).any():
            raise ValueError('Invalid player or future-frame key')
        ordinal_dates(self.keys[:, 0])
        # Role/horizon cannot drift within a trajectory.
        if n:
            order = np.lexsort(tuple(self.keys[:, j] for j in (3, 2, 1, 0)))
            k = self.keys[order]
            same = np.all(k[1:, :3] == k[:-1, :3], axis=1)
            if ((self.role[order][1:] != self.role[order][:-1])[same].any()
                    or (np.abs(np.diff(self.horizon[order])) > 1e-10)[same].any()):
                raise ValueError('Role or horizon changes within a trajectory')


def ordinal_dates(game_ids: np.ndarray) -> np.ndarray:
    days = np.asarray(game_ids) // 100
    mapping = {}
    for d in np.unique(days):
        try:
            mapping[int(d)] = datetime.strptime(str(int(d)), '%Y%m%d').toordinal()
        except (ValueError, OverflowError) as exc:
            raise ValueError('Invalid date-coded game identifier') from exc
    return np.array([mapping[int(d)] for d in days], dtype=np.int64)


def from_arrays(keys: np.ndarray, role: np.ndarray, base: np.ndarray) -> Metadata:
    """Decode organizer clocks only; do not accept targets or fitted estimates."""
    if base.ndim != 2 or len(base) != len(keys) or base.shape[1] < 6:
        raise ValueError('Original base matrix is missing')
    if not np.isfinite(base[:, :6]).all() or not np.all(base[:, 5] == 0):
        raise ValueError('Only original-origin, finite task clocks are accepted')
    if not np.allclose(base[:, 1], keys[:, 3] / 10, rtol=0, atol=1e-5):
        raise ValueError('Stored query clock differs from forecast keys')
    # Recover exact integer horizon despite the historical float32 storage step.
    hraw = (base[:, 1].astype(float) + base[:, 3].astype(float)) * 10
    hframes = np.rint(hraw)
    if not np.allclose(hraw, hframes, rtol=0, atol=1e-4):
        raise ValueError('Organizer horizon cannot be recovered as an integer')
    result = Metadata(keys.copy(), role.copy(), keys[:, 3].astype(float) / 10, hframes / 10)
    result.validate()
    if not np.allclose(base[:, 4], result.time / result.horizon, rtol=0, atol=1e-6):
        raise ValueError('Stored phase differs from organizer query/horizon')
    return result


def group_keys(meta: Metadata) -> tuple[list[str], list[str], list[str]]:
    phase = np.searchsorted(CONFIG['phase_edges'], meta.time / meta.horizon, side='right')
    horizon = np.searchsorted(CONFIG['horizon_edges_seconds'], meta.horizon, side='right')
    a, b, c = [], [], []
    for i, (p, h) in enumerate(zip(phase, horizon)):
        a.append(str(int(p)))
        b.append(f'{int(meta.role[i])}:{int(p)}:{int(h)}')
        c.append(f'{int(meta.keys[i, 2])}:{b[-1]}')
    return a, b, c


def new_state() -> dict:
    return {'version': VERSION, 'config': copy.deepcopy(CONFIG), 'last_training_date': 0,
            'tables': {level: {} for level in LEVELS}}


def validate_state(state: dict) -> None:
    if (state.get('version') != VERSION or state.get('config') != CONFIG
            or set(state.get('tables', {})) != set(LEVELS)):
        raise ValueError('Historical table specification changed')
    latest = state['last_training_date']
    if not isinstance(latest, int) or latest < 0:
        raise ValueError('Invalid historical table cutoff')
    for table in state['tables'].values():
        for record in table.values():
            if (len(record) != 6 or not np.isfinite(record).all() or record[0] < 1
                    or int(record[0]) != record[0] or record[3] < 0 or record[4] < 0
                    or record[5] <= 0 or record[5] > latest):
                raise ValueError('Invalid historical count/moment record')


def _moments(record: list | None, mean: np.ndarray, second: np.ndarray,
             smoothing: float) -> tuple[np.ndarray, np.ndarray]:
    if record is None:
        return mean.copy(), second.copy()
    n, sx, sy, qx, qy, _ = record
    m = (np.array([sx, sy]) + smoothing * mean) / (n + smoothing)
    q = (np.array([qx, qy]) + smoothing * second) / (n + smoothing)
    return m, q


def _transform(meta: Metadata, state: dict) -> np.ndarray:
    n = len(meta.keys)
    values = np.empty((n, len(NAMES)), dtype=np.float64)
    days = ordinal_dates(meta.keys[:, 0])
    if n and state['last_training_date'] >= int(days.min()):
        raise ValueError('History contains a current-date or future-date outcome')
    groups = group_keys(meta)
    strengths = [CONFIG[k] for k in ('global_smoothing_trajectories',
                                    'role_smoothing_trajectories', 'player_smoothing_trajectories')]
    cache = {}
    for i in range(n):
        tag = tuple(g[i] for g in groups) + (int(days[i]),)
        if tag not in cache:
            mean = np.zeros(2)
            second = np.full(2, CONFIG['zero_history_rate_sd'] ** 2)
            support = []
            blocks = []
            for level, g, strength in zip(LEVELS, groups, strengths):
                r = state['tables'][level].get(g[i])
                count, age = (r[0], int(days[i]) - r[5]) if r is not None else (0, 0)
                support += [np.log1p(count), float(count == 0), np.log1p(age)]
                mean, second = _moments(r, mean, second, strength)
                sd = np.sqrt(np.maximum(second - mean * mean, 0))
                blocks.append((mean.copy(), sd.copy()))
            cache[tag] = (support, blocks[1], blocks[2])
        support, (rmean, rsd), (pmean, psd) = cache[tag]
        values[i] = np.r_[support, rmean, rsd, rmean * meta.time[i],
                          pmean, psd, pmean * meta.time[i]]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite historical feature')
    return values


def transform_frozen(meta: Metadata, state: dict) -> np.ndarray:
    """Prediction interface: no targets and no state updates are accepted."""
    meta.validate()
    validate_state(state)
    return _transform(meta, state)


def _update_date(meta: Metadata, target: np.ndarray, state: dict) -> int:
    days = ordinal_dates(meta.keys[:, 0])
    if len(np.unique(days)) != 1 or days[0] <= state['last_training_date']:
        raise ValueError('Updates require one strictly later training date')
    groups = group_keys(meta)
    # One contribution per player/play/phase, not one per 10-Hz forecast row.
    order = np.lexsort(tuple(meta.keys[:, j] for j in (3, 2, 1, 0)))
    buckets = {}
    rate = target / meta.time[:, None]
    for i in order:
        donor = (*map(int, meta.keys[i, :3]), groups[0][i])
        buckets.setdefault(donor, []).append(int(i))
    for rows in buckets.values():
        first = rows[0]
        value = rate[rows].mean(axis=0)
        for level, g in zip(LEVELS, groups):
            r = state['tables'][level].setdefault(g[first], [0, 0., 0., 0., 0., 0])
            r[0] += 1
            r[1] += float(value[0]); r[2] += float(value[1])
            r[3] += float(value[0] ** 2); r[4] += float(value[1] ** 2)
            r[5] = int(days[0])
    state['last_training_date'] = int(days[0])
    return len(buckets)


def fit_ordered(train: Metadata, targets: np.ndarray, evaluation: Metadata,
                on_date: Callable[[dict], None] | None = None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Encode training by date, then freeze for evaluation. No evaluation targets."""
    train.validate(); evaluation.validate()
    y = np.asarray(targets, dtype=float)
    if len(train.keys) == 0 or y.shape != (len(train.keys), 2) or not np.isfinite(y).all():
        raise ValueError('Finite training residual targets are required')
    td, ed = ordinal_dates(train.keys[:, 0]), ordinal_dates(evaluation.keys[:, 0])
    if len(ed) and int(td.max()) >= int(ed.min()):
        raise ValueError('Evaluation dates must be strictly after all training dates')
    state = new_state(); xt = np.empty((len(y), len(NAMES)), dtype=np.float64)
    counts = []
    dates = np.unique(td)
    for step, day in enumerate(dates, 1):
        ix = np.flatnonzero(td == day); batch = train.take(ix)
        xt[ix] = _transform(batch, state)  # ALL same-day queries before any same-day update.
        donors = _update_date(batch, y[ix], state)
        record = {'completed_dates': step, 'total_dates': len(dates), 'rows': len(ix),
                  'trajectory_phase_donors': donors}
        counts.append(record)
        if on_date is not None:
            on_date(record)
    validate_state(state)
    xe = transform_frozen(evaluation, state)
    info = {'training_dates': len(dates), 'training_rows': len(y), 'evaluation_rows': len(ed),
            'trajectory_phase_donors': sum(r['trajectory_phase_donors'] for r in counts),
            'first_training_date_all_cold': bool(np.all(xt[td == td.min()][:, [1, 4, 7]] == 1)),
            'same_date_outcomes_excluded': True, 'evaluation_table_frozen': True,
            'training_support': support_summary(xt), 'evaluation_support': support_summary(xe),
            'training_rate_abs_quantiles': np.quantile(np.abs(y / train.time[:, None]), [.5, .9, .99, 1.], axis=0).tolist(),
            'table_sizes': {k: len(v) for k, v in state['tables'].items()}}
    return xt, xe, state, info


def support_summary(features: np.ndarray) -> list[dict]:
    if features.ndim != 2 or features.shape[1] != len(NAMES):
        raise ValueError('Historical feature shape changed')
    result = []
    for j, level in enumerate(LEVELS):
        n = np.expm1(features[:, j * 3])
        result.append({'level': level, 'rows': len(n),
                       'warm_fraction': float(np.mean(features[:, j * 3 + 1] == 0)) if len(n) else None,
                       'median_prior_trajectories': float(np.median(n)) if len(n) else None,
                       'p90_prior_trajectories': float(np.quantile(n, .9)) if len(n) else None})
    return result
