"""Analytic/synthetic fixtures only. Never represent these metrics as NFL results."""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origin_features import ROLES, KEYS, ALL_NAMES, BASE_NAMES, make_example, chronological_folds, fit_ridge, predict_ridge, rmse
from study import atomic_json, atomic_npz, digest


def play(game=2023090101, play_id=1):
    records, labels = [], []
    for k in range(4):
        base = np.array([30. + 3 * k, 15. + 4 * k])
        v = np.array([3. + 0.2 * k, 0.4 - 0.05 * k])
        a = np.array([0.1, 0.04 * (k + 1)])
        for f in range(1, 31):
            t = (f - 30) / 10
            xy = base + v * t + 0.5 * a * t * t
            velocity = v + a * t
            records.append(dict(game_id=game, play_id=play_id, nfl_id=100+k, frame_id=f,
                                x=xy[0], y=xy[1], s=float(np.linalg.norm(velocity)),
                                dir=float(np.rad2deg(np.arctan2(velocity[0], velocity[1])) % 360),
                                o=80., player_role=ROLES[k], player_side='Defense' if k == 1 else 'Offense',
                                player_to_predict=k < 2, num_frames_output=16,
                                ball_land_x=44., ball_land_y=19., play_direction='right'))
        if k < 2:
            for f in range(1, 17):
                t = f / 10
                xy = base + v * t + 0.5 * a * t * t
                labels.append(dict(game_id=game, play_id=play_id, nfl_id=100+k, frame_id=f, x=xy[0], y=xy[1]))
    return pd.DataFrame(records), pd.DataFrame(labels)


def parent_fixture(root: Path):
    """Construct a minimal authentic Round 1 artifact layout, including index gaps."""
    parent = root / 'parent'
    repo = root / 'repo'
    (repo / 'data/raw/train').mkdir(parents=True)
    parent.mkdir()
    files, raw_frames, all_arrays = [], [], []
    source = 'synthetic-round1-source-not-an-NFL-run'
    for i in range(16):
        day = date(2023, 9, 1) + timedelta(days=i)
        game = int(day.strftime('%Y%m%d')) * 100 + 1
        raw, output = play(game)
        # Deterministic between-game variation, using INPUTS only for X.
        raw['s'] = raw.s * (1 + 0.005 * i)
        raw_frames.append(raw)
        for offset in (0, 5):
            ex = make_example(raw, output, offset)
            ex = {k: (v.astype(np.float32) if k in ('X', 'y', 'cv') else v) for k, v in ex.items() if isinstance(v, np.ndarray)}
            path = parent / 'research/plays' / f'{game}_1_o{offset}.npz'
            atomic_npz(path, **ex)
            files.append({'offset': offset, 'path': path.relative_to(parent/'research').as_posix(), 'sha256': digest(path)})
            all_arrays.append(ex)
    rawfile = repo / 'data/raw/train/input_2023_w01.csv'
    pd.concat(raw_frames, ignore_index=True).to_csv(rawfile, index=False)
    manifest = {'source_signature': source, 'files': files}
    atomic_json(parent/'research/dataset_manifest.json', manifest)
    manifest_sha = digest(parent/'research/dataset_manifest.json')
    sig = hashlib.sha256((manifest_sha + source + 'screen-v1').encode()).hexdigest()
    data = {k: np.concatenate([e[k] for e in all_arrays]) for k in all_arrays[0]}
    original = data['offset'] == 0
    folds = chronological_folds(data['keys'][original, 0])
    protocol = {'signature': sig, 'penalty': 0.01, 'folds': [
        {'fold': f['fold'], 'train_games': f['train_games'].tolist(), 'validation_games': f['validation_games'].tolist()} for f in folds]}
    atomic_json(parent/'research/screen/protocol.json', protocol)
    yy, pp = [], {'control': [], 'arrival': []}
    for fold in folds:
        ti = np.flatnonzero(original & np.isin(data['keys'][:, 0], fold['train_games']))
        ti = np.random.default_rng(20260911+fold['fold']).permutation(ti)
        vi = np.flatnonzero(original & np.isin(data['keys'][:, 0], fold['validation_games']))
        yy.append(data['y'][vi])
        for arm, width in (('control', len(BASE_NAMES)), ('arrival', len(ALL_NAMES))):
            model = fit_ridge(data['X'][ti, :width], data['y'][ti])
            pred = predict_ridge(model, data['X'][vi, :width]); pp[arm].append(pred)
            path = parent/'research/screen'/f'fold_{fold["fold"]}_{arm}.npz'
            atomic_npz(path, **{f'model_{k}': v for k, v in model.items()}, pred=pred, truth=data['y'][vi].astype(float),
                       evaluation_keys=data['keys'][vi], train_indices=ti)
            atomic_json(path.with_suffix('.json'), {'signature': sig, 'sha256': digest(path), 'rmse': rmse(data['y'][vi], pred)})
    summary = {'status': 'SYNTHETIC', 'evaluation_rows': sum(map(len, yy)),
               'pooled_rmse': {a: rmse(np.concatenate(yy), np.concatenate(p)) for a, p in pp.items()}}
    atomic_json(parent/'research/screen_summary.json', summary)
    contract = {'parent_manifest_sha256': manifest_sha, 'parent_source_signature': source,
                'parent_summary_sha256': digest(parent/'research/screen_summary.json'),
                'original_rows': int(original.sum()), 'original_games': 16, 'original_plays': 16,
                'raw_files': [{'path': rawfile.relative_to(repo).as_posix(), 'sha256': digest(rawfile)}]}
    return parent, repo, contract
