"""Temporal feature replication with immutable parents and no first-fold refitting.

Round 5 adds no new candidate definition. It tests the retained 62 direct-state
fields on two pre-existing later chronological folds. No goal-geometry treatment,
raw output CSV, augmentation, new model family, or held-out partition is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from origin_features import BASE_NAMES, REQUIRED, rmse
from state_features import DIRECT_NAMES, build_features, storage_parity
from parent_support import digest, hash_json, safe_file, read_npz, atomic_json, atomic_npz, seal_json
from study import parent_context, replay_control, assemble as assemble_round4, checkpoint_read, json_file

CODE_FILES = ('confirmation.py', 'feature_worker.py', 'run_round.py', 'input_contract.json',
              'parent_contract.json', 'origin_features.py', 'state_features.py',
              'pair_features.py', 'parent_support.py', 'tree_core.py', 'study.py')
ARMS = ('control', 'direct_state')


def validate_folds(data: dict, folds: list[dict]) -> list[dict]:
    """Game-level chronological order and disjoint evaluation populations."""
    if [f['fold'] for f in folds] != [1, 2, 3]:
        raise ValueError('Expected the original three ordered folds')
    seen: set[int] = set()
    populations = []
    previous_train: set[int] = set()
    all_games = set(map(int, data['keys'][:, 0]))
    for f in folds:
        train = set(map(int, f['train_games']))
        evaluation = set(map(int, f['validation_games']))
        if not train or not evaluation or len(train) != len(f['train_games']) or len(evaluation) != len(f['validation_games']):
            raise ValueError('Empty or duplicate game declarations')
        if train & evaluation or seen & evaluation:
            raise ValueError('Training/evaluation or evaluation/evaluation game overlap')
        if max(train) // 100 >= min(evaluation) // 100:
            raise ValueError('Same-date or backwards chronological fold')
        if previous_train and not previous_train.issubset(train):
            raise ValueError('Expanding chronological training contract changed')
        if not train.issubset(all_games) or not evaluation.issubset(all_games):
            raise ValueError('A declared game has no selected rows')
        tr = np.flatnonzero(np.isin(data['keys'][:, 0], sorted(train)))
        va = np.flatnonzero(np.isin(data['keys'][:, 0], sorted(evaluation)))
        populations.append({'fold': f['fold'], 'training_games': len(train),
                            'evaluation_games': len(evaluation), 'training_rows': len(tr),
                            'evaluation_rows': len(va)})
        seen |= evaluation
        previous_train = train
    return populations


def context(kit: Path, round4: Path, round4_kit: Path, round3: Path,
            round3_kit: Path, repo: Path, *, verify_identity: bool = True) -> dict:
    c = json_file(kit, 'input_contract.json')
    old = json_file(kit, 'parent_contract.json')
    for root in (round4, round4_kit):
        if any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError('Symlink parent root')
    if digest(safe_file(round4_kit, 'input_contract.json')) != c['round4_contract_sha256']:
        raise ValueError('Round 4 input contract changed')
    for name, sha in c['round4_source_hashes'].items():
        if digest(safe_file(round4_kit, name)) != sha:
            raise ValueError('Reviewed Round 4 source changed: ' + name)
    for name, sha in c['round4_report_hashes'].items():
        if digest(safe_file(round4, name)) != sha:
            raise ValueError('Reviewed Round 4 report changed: ' + name)
    ctx = parent_context(round3, round3_kit, repo, old, verify_identity=verify_identity)
    result = json_file(round4, 'summary.json')
    replay = json_file(round4, 'replay.json')
    for key in ('source_signature', 'experiment_signature', 'pooled_rmse', 'contrasts',
                'training_rows', 'evaluation_rows', 'evaluation_games', 'slices'):
        if result[key] != replay[key]:
            raise ValueError('Round 4 scientific fit and replay disagree')
    if (result['status'] != 'round4_first_fold_complete'
            or result['source_signature'] != c['round4_source_signature']
            or result['experiment_signature'] != c['round4_experiment_signature']
            or result['pooled_rmse'] != c['round4_metrics']
            or not replay['all_new_prediction_replays_exact']
            or replay['new_coordinate_models'] != 0 or replay['parent_refits'] != 0):
        raise ValueError('Incomplete or changed Round 4 result')
    # Continuation is for the passed representation, not the failed Round 3 arms.
    passed = any(x['treatment'] == 'direct_state' and x['control'] == 'control'
                 and x['screen_gate'] for x in result['contrasts'])
    if not passed:
        raise ValueError('Direct-state feature gate did not authorize this continuation')
    populations = validate_folds(ctx['data'], ctx['source']['folds'])
    wanted = {(p['game'], p['play']) for p in ctx['selection']['plays']}
    if wanted != set(map(tuple, ctx['data']['keys'][:, :2])):
        raise ValueError('Selected play identities differ from the saved dataset')
    ctx.update(round4=round4, round4_summary=result, round5_contract=c,
               populations=populations, kit=kit)
    return ctx


def signature(ctx: dict) -> str:
    return hash_json({'code': {n: digest(safe_file(ctx['kit'], n)) for n in CODE_FILES},
                      'parent_data': ctx['source'],
                      'round4_experiment': ctx['round5_contract']['round4_experiment_signature'],
                      'study': 'direct-state-temporal-replication-v1'})


def replay_discovery(ctx: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Exact forward replay of four original coordinate models; no fit calls."""
    from tree_core import SETTINGS, versions, checked_blob
    control, control_receipt = replay_control(ctx)
    parent = ctx['round4']
    sig = ctx['round5_contract']['round4_source_signature']
    d, feature_hash = assemble_round4(ctx, parent, sig)
    protocol = json_file(parent, 'experiment_protocol.json')
    expected = ctx['round5_contract']['round4_experiment_signature']
    if (hash_json(protocol) != expected or protocol['source_signature'] != sig
            or protocol['features_hash'] != feature_hash or protocol['settings'] != SETTINGS
            or protocol['environment'] != versions()
            or protocol['parent_data_sha256'] != ctx['source']['sha256']):
        raise ValueError('Round 4 fitted feature/model protocol drift')
    f = ctx['fold']
    tr = np.flatnonzero(np.isin(d['keys'][:, 0], f['train_games']))
    va = np.flatnonzero(np.isin(d['keys'][:, 0], f['validation_games']))
    m = np.column_stack([d['X'][:, :len(BASE_NAMES)], d['state']])
    keep = np.ptp(m[tr], axis=0) > 1e-10
    xt = np.ascontiguousarray(m[tr][:, keep], dtype=np.float64)
    xe = np.ascontiguousarray(m[va][:, keep], dtype=np.float64)
    parts, model_receipts = [], []
    for axis in range(2):
        folder = parent / 'models/direct_state' / str(axis)
        receipt = json_file(folder, 'checkpoint.json')
        parent_sig = hash_json({'study': expected, 'arm': 'direct_state',
                                'axis': axis, 'keep': keep.tolist()})
        wanted_sig = hash_json({'parent': parent_sig, 'settings': SETTINGS, 'environment': versions()})
        if receipt['signature'] != wanted_sig or receipt['step'] != SETTINGS['max_iter']:
            raise ValueError('Original direct-state checkpoint is incomplete or changed')
        model = checked_blob(folder, receipt)
        if model.n_iter_ != SETTINGS['max_iter'] or not np.array_equal(model.predict(xt[:64]), receipt['probe']):
            raise ValueError('Original direct-state training probe failed')
        parts.append(model.predict(xe))
        model_receipts.append({'axis': axis, 'sha256': receipt['sha256'], 'trees': model.n_iter_})
    prediction = np.column_stack(parts)
    path = safe_file(parent, 'predictions/direct_state.npz')
    pr = json_file(parent, 'predictions/direct_state.json')
    if pr['signature'] != expected or pr['sha256'] != digest(path):
        raise ValueError('Original direct-state prediction hash/signature mismatch')
    saved = read_npz(path)
    for a, b in ((saved['pred'], prediction), (saved['keys'], d['keys'][va]), (saved['truth'], d['y'][va])):
        if not np.array_equal(a, b):
            raise ValueError('Original direct-state prediction/key/target replay failed')
    if abs(rmse(d['y'][va], prediction) - ctx['round5_contract']['round4_metrics']['direct_state']) > 1e-12:
        raise ValueError('Original direct-state metric cannot be reproduced')
    values = {'control': control, 'direct_state': prediction}
    rec = {'old_coordinate_models_replayed': 4, 'old_models_refitted': 0,
           'all_parent_predictions_exact': True, 'control': control_receipt,
           'direct_models': model_receipts,
           'metrics': {k: rmse(d['y'][va], p) for k, p in values.items()}}
    return values, rec


def old_play(ctx: dict, item: dict) -> tuple[str, Path, str]:
    name = f"{item['game']}_{item['play']}.npz"
    p = safe_file(ctx['parent'], 'features/plays/' + name)
    receipt = json_file(p.parent, p.with_suffix('.json').name)
    sha = digest(p)
    if receipt['signature'] != ctx['source']['signature'] or receipt['sha256'] != sha:
        raise ValueError('Original feature shard identity failed')
    return name, p, sha


def state_checkpoint(ctx: dict, out: Path, item: dict, sig: str) -> tuple[dict, dict] | None:
    name, old, old_sha = old_play(ctx, item)
    r4 = ctx['round4'] / 'features' / name
    if r4.exists() or r4.with_suffix('.json').exists():
        z = checkpoint_read(r4, ctx['round5_contract']['round4_source_signature'], old_sha)
        entry = {'owner': 'round4', 'name': name, 'sha256': digest(r4), 'parent_sha256': old_sha}
    else:
        path = out / 'features' / name
        if not path.exists() and not path.with_suffix('.json').exists():
            return None
        receipt = json_file(path.parent, path.with_suffix('.json').name)
        path = safe_file(path.parent, path.name)
        if (receipt['signature'] != sig or receipt['parent_sha256'] != old_sha or receipt['sha256'] != digest(path)):
            raise ValueError('New direct-state checkpoint identity failed')
        z = read_npz(path)
        if str(z['signature']) != sig:
            raise ValueError('New direct-state array signature failed')
        entry = {'owner': 'round5', 'name': name, 'sha256': digest(path), 'parent_sha256': old_sha}
    original_keys = read_npz(old, ('keys',))['keys']
    if (not np.array_equal(z['keys'], original_keys) or z['state'].shape != (len(original_keys), len(DIRECT_NAMES))
            or z['state'].dtype != np.dtype('float64') or not np.isfinite(z['state']).all()):
        raise ValueError('Direct-state checkpoint key/shape/dtype/finiteness error')
    return z, entry


def fold_indices(data: dict, fold: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.flatnonzero(np.isin(data['keys'][:, 0], fold['train_games'])),
            np.flatnonzero(np.isin(data['keys'][:, 0], fold['validation_games'])))


def prepare(ctx: dict, repo: Path, out: Path, sig: str, *, smoke: bool, event) -> dict:
    start = time.monotonic()
    first_later = ctx['source']['folds'][1]
    selected = ctx['selection']['plays']
    if smoke:
        selected = [p for p in selected if p['game'] in first_later['train_games']][:32]
    if not selected:
        raise ValueError('No eligible plays for the declared stage')
    pending, parent_reused, local_reused = [], 0, 0
    for p in selected:
        found = state_checkpoint(ctx, out, p, sig)
        if found is None:
            pending.append(p)
        elif found[1]['owner'] == 'round4':
            parent_reused += 1
        else:
            local_reused += 1
    # Smoke independently rebuilds raw features, even when a parent cache exists.
    work = selected if smoke else pending
    raw_inventory = {x['path']: x for x in ctx['contract']['raw_input_files']}
    rebuilt, created, parity_rows = 0, 0, 0
    for name in sorted({p['input'] for p in work}):
        if name not in raw_inventory:
            raise ValueError('Unregistered observed-input file')
        raw_path = safe_file(repo, name)
        spec = raw_inventory[name]
        if raw_path.stat().st_size != spec['size'] or digest(raw_path) != spec['sha256']:
            raise ValueError('Observed file checksum/size mismatch')
        wanted = {(p['game'], p['play']): p for p in work if p['input'] == name}
        columns = list(pd.read_csv(raw_path, nrows=0).columns)
        use = [n for n in columns if n in REQUIRED | {'s', 'dir', 'o', 'a'}]
        parts = []
        for chunk in pd.read_csv(raw_path, usecols=use, chunksize=50000):
            mask = pd.MultiIndex.from_frame(chunk[['game_id', 'play_id']]).isin(wanted)
            if mask.any():
                parts.append(chunk.loc[mask])
        if not parts:
            raise ValueError('Selected observed-input rows are absent')
        groups = pd.concat(parts, ignore_index=True).groupby(['game_id', 'play_id'], sort=True)
        found_keys = set()
        for key, raw in groups:
            p = wanted[tuple(map(int, key))]
            found_keys.add(tuple(map(int, key)))
            filename, old, old_sha = old_play(ctx, p)
            z = read_npz(old, ('X', 'keys'))
            feat = build_features(raw.sort_values(['nfl_id', 'frame_id']).reset_index(drop=True),
                                  z['keys'], cutoff=int(raw.frame_id.max()))
            storage_parity(feat['legacy'], z['X'])
            existing = state_checkpoint(ctx, out, p, sig)
            if existing is not None:
                if not np.array_equal(existing[0]['state'], feat['state']):
                    raise ValueError('Direct-state values differ from the retained representation')
            else:
                target = out / 'features' / filename
                atomic_npz(target, state=feat['state'], keys=z['keys'], signature=np.array(sig))
                atomic_json(target.with_suffix('.json'), {'signature': sig, 'parent_sha256': old_sha,
                                                         'sha256': digest(target)})
                state_checkpoint(ctx, out, p, sig)
                created += 1
            parity_rows += len(z['keys'])
            rebuilt += 1
            if rebuilt % 16 == 0:
                event('state_preparation', rebuilt=rebuilt, total_to_rebuild=len(work),
                      new_checkpoints=created, parent_reused=parent_reused)
        if found_keys != set(wanted) or digest(raw_path) != spec['sha256']:
            raise ValueError('Incomplete observed rows or concurrent raw-input modification')
    if rebuilt != len(work):
        raise ValueError('Incomplete state preparation')
    entries = []
    for p in selected:
        got = state_checkpoint(ctx, out, p, sig)
        if got is None:
            raise ValueError('Missing prepared state checkpoint')
        entries.append(got[1])
    result = {'status': 'temporal_state_smoke_passed' if smoke else 'temporal_state_ready',
              'signature': sig, 'selected_plays': len(selected), 'new_checkpoints': created,
              'round4_checkpoints_reused': parent_reused, 'round5_checkpoints_reused': local_reused,
              'raw_feature_plays_rebuilt': rebuilt, 'storage_parity_rows_checked': parity_rows,
              'new_feature_definitions': 0, 'retained_direct_state_columns': len(DIRECT_NAMES),
              'new_model_fits': 0, 'raw_output_csvs_opened': 0,
              'feature_research': 'open', 'elapsed_seconds': round(time.monotonic() - start, 3)}
    if not smoke:
        manifest = {'signature': sig, 'entries': entries, 'parent_data_sha256': ctx['source']['sha256']}
        seal_json(out / 'feature_manifest.json', manifest)
        result['manifest_sha256'] = digest(out / 'feature_manifest.json')
    filename = 'smoke.json' if smoke else 'preparation.json'
    if (out / filename).exists():
        old = json_file(out, filename)
        if old['signature'] != sig or old['status'] != result['status']:
            raise ValueError('Previous preparation belongs to another study')
        filename = filename.removesuffix('.json') + '_reuse.json'
    atomic_json(out / filename, result)
    event('state_preparation_complete', **result)
    return result


def assemble(ctx: dict, out: Path, sig: str) -> tuple[dict, str]:
    manifest = json_file(out, 'feature_manifest.json')
    if manifest['signature'] != sig or manifest['parent_data_sha256'] != ctx['source']['sha256']:
        raise ValueError('Prepared feature manifest lineage mismatch')
    state, keys, entries = [], [], []
    total = 0
    for p in ctx['selection']['plays']:
        got = state_checkpoint(ctx, out, p, sig)
        if got is None:
            raise ValueError('Prepared state missing')
        z, e = got
        state.append(z['state']); keys.append(z['keys']); entries.append(e)
        total += z['state'].nbytes + z['keys'].nbytes
        if total > ctx['contract']['max_feature_bytes']:
            raise ValueError('Feature memory cap exceeded')
    if entries != manifest['entries']:
        raise ValueError('Feature checkpoint receipts changed since preparation')
    key_array = np.concatenate(keys)
    if not np.array_equal(key_array, ctx['data']['keys']):
        raise ValueError('State rows are not aligned to the original data')
    return {**ctx['data'], 'state': np.concatenate(state)}, hash_json(manifest)


def matrices(d: dict) -> dict[str, np.ndarray]:
    base = d['X'][:, :len(BASE_NAMES)]
    return {'control': base, 'direct_state': np.column_stack([base, d['state']])}


def training_mask(matrix: np.ndarray, training_indices: np.ndarray) -> np.ndarray:
    if not len(training_indices):
        raise ValueError('No training rows')
    keep = np.ptp(matrix[training_indices], axis=0) > 1e-10
    if not keep.any():
        raise ValueError('No variable training features')
    return keep


def game_statistics(y: np.ndarray, a: np.ndarray, b: np.ndarray, games: np.ndarray) -> dict:
    if y.shape != a.shape or y.shape != b.shape or y.ndim != 2 or y.shape[1] != 2 or len(y) != len(games):
        raise ValueError('Incompatible coordinate arrays')
    if not len(y) or any(not np.isfinite(v).all() for v in (y, a, b, games)):
        raise ValueError('Empty or nonfinite predictions')
    unique, inv = np.unique(games, return_inverse=True)
    return {'games': unique, 'rows': np.bincount(inv),
            'control_sse': np.bincount(inv, weights=np.square(y - a).sum(1)),
            'direct_state_sse': np.bincount(inv, weights=np.square(y - b).sum(1))}


def compare(stats: dict, c: dict) -> dict:
    n, sa, sb = stats['rows'], stats['control_sse'], stats['direct_state_sse']
    if len(n) < 2 or (n <= 0).any() or sa.sum() <= 0:
        raise ValueError('Need positive control error and at least two evaluation games')
    ca = float(np.sqrt(sa.sum() / (2 * n.sum())))
    cb = float(np.sqrt(sb.sum() / (2 * n.sum())))
    rng = np.random.default_rng(20260911)
    repeats = int(c['bootstrap_resamples'])
    looks = int(c['nominal_total_comparison_looks'])
    if repeats < 100 or looks < 1:
        raise ValueError('Invalid frozen uncertainty configuration')
    draws = []
    for start in range(0, repeats, 500):
        ix = rng.integers(0, len(n), (min(500, repeats - start), len(n)))
        draws.extend((np.sqrt(sb[ix].sum(1) / (2 * n[ix].sum(1))) - np.sqrt(sa[ix].sum(1) / (2 * n[ix].sum(1)))).tolist())
    lo, hi = np.quantile(draws, [.05 / (2 * looks), 1 - .05 / (2 * looks)])
    gain = 1 - cb / ca
    return {'control_rmse': ca, 'direct_state_rmse': cb, 'delta_rmse': cb - ca,
            'relative_gain': gain, 'adjusted_low': float(lo), 'adjusted_high': float(hi),
            'games': len(n), 'rows': int(n.sum()), 'bootstrap_resamples': repeats,
            'nominal_total_comparison_looks': looks,
            'numerical_screen_passed': bool(gain >= c['minimum_relative_gain'] and hi < 0),
            'scope': 'Exploratory game-block interval on reused game splits; not independent confirmation.'}


def influence(stats: dict) -> dict:
    n, a, b = stats['rows'], stats['control_sse'], stats['direct_state_sse']
    remaining = n.sum() - n
    valid = remaining > 0
    ca = np.sqrt(np.maximum(a.sum() - a[valid], 0) / (2 * remaining[valid]))
    cb = np.sqrt(np.maximum(b.sum() - b[valid], 0) / (2 * remaining[valid]))
    gains = 1 - cb / np.maximum(ca, 1e-30)
    return {'delete_one_game_min_gain': float(gains.min()), 'delete_one_game_max_gain': float(gains.max()),
            'delete_one_game_nonpositive_count': int((gains <= 0).sum()), 'games': len(n)}


def slice_statistics(y: np.ndarray, predictions: dict, keys: np.ndarray, role: np.ndarray) -> list[dict]:
    rows = []
    masks = [('first_second', keys[:, 3] <= 10), ('after_first_second', keys[:, 3] > 10)]
    masks += [(f'role_{r}', role == r) for r in range(4)]
    for arm, pred in predictions.items():
        for label, mask in masks:
            if mask.any():
                sse = float(np.square(y[mask] - pred[mask]).sum())
                rows.append({'arm': arm, 'slice': label, 'rows': int(mask.sum()), 'sse': sse,
                             'rmse': float(np.sqrt(sse / (2 * mask.sum())))})
    return rows


def may_continue_fold3(fold2: dict, c: dict) -> bool:
    # A cost guard, not a statistical acceptance gate. Modest negative folds remain visible.
    return fold2['comparison']['relative_gain'] > -c['futility_relative_deterioration']


def protocol(ctx: dict, feature_hash: str, sig: str) -> dict:
    from tree_core import SETTINGS, versions
    return {'signature': sig, 'feature_manifest_hash': feature_hash, 'settings': SETTINGS,
            'environment': versions(), 'folds': ctx['source']['folds'], 'arms': list(ARMS),
            'max_new_coordinate_models': 8, 'old_models_refitted': 0,
            'schema': {'control': list(BASE_NAMES), 'direct_state': list(BASE_NAMES) + list(DIRECT_NAMES)},
            'futility_relative_deterioration': ctx['round5_contract']['futility_relative_deterioration'],
            'raw_output_csvs_opened': 0, 'outer_holdout_used': False}


def verified_fold_report(out: Path, fold: int) -> dict:
    path = safe_file(out, f'fold_{fold}/summary.json')
    receipt = json_file(out, f'fold_{fold}/summary.receipt.json')
    report = json_file(path.parent, path.name)
    if (receipt['sha256'] != digest(path)
            or receipt['experiment_signature'] != report['experiment_signature']):
        raise ValueError('Completed temporal report checksum/identity mismatch')
    return report


def run_fold(ctx: dict, out: Path, sig: str, fold: int, *, replay_only: bool = False) -> dict:
    from tree_core import fit_axis, event
    if fold not in (2, 3):
        raise ValueError('Round 5 may only fit folds 2 and 3; first-fold models are read-only')
    d, feature_hash = assemble(ctx, out, sig)
    spec = protocol(ctx, feature_hash, sig)
    seal_json(out / 'experiment_protocol.json', spec)
    experiment = hash_json(spec)
    if fold == 3:
        first = verified_fold_report(out, 2)
        if first['experiment_signature'] != experiment or first['status'] != 'temporal_fold_complete':
            raise ValueError('Complete and verify fold 2 before fold 3')
        if not may_continue_fold3(first, ctx['round5_contract']):
            result = {'status': 'stopped_for_futility', 'fold': 3, 'new_coordinate_models': 0,
                      'signature': sig, 'reason': 'Fold 2 is at least 5% worse. No additional fits are authorized.',
                      'feature_research': 'open'}
            if replay_only:
                raise ValueError('Fold 3 was stopped before fitting; replay cannot fit it')
            atomic_json(out / 'fold_3/futility.json', result)
            event('futility_stop', **result)
            return result
    f = ctx['source']['folds'][fold - 1]
    tr, va = fold_indices(d, f)
    predictions, fits, retained = {}, [], {}
    start = time.monotonic()
    for arm, matrix in matrices(d).items():
        keep = training_mask(matrix, tr)
        retained[arm] = int(keep.sum())
        x = np.ascontiguousarray(matrix[tr][:, keep], dtype=np.float64)
        xe = np.ascontiguousarray(matrix[va][:, keep], dtype=np.float64)
        parts = []
        for axis in range(2):
            axis_sig = hash_json({'study': experiment, 'fold': fold, 'arm': arm, 'axis': axis, 'keep': keep.tolist()})
            folder = out / f'fold_{fold}' / 'models' / arm / str(axis)
            if replay_only and not folder.is_dir():
                raise ValueError('Replay refuses a missing coordinate model')
            pred, rec = fit_axis(folder, x, d['y'][tr, axis], xe, axis_sig, replay_only=replay_only)
            parts.append(pred); fits.append({'arm': arm, 'axis': axis, **rec})
        prediction = np.column_stack(parts)
        predictions[arm] = prediction
        path = out / f'fold_{fold}' / 'predictions' / (arm + '.npz')
        rp = path.with_suffix('.json')
        if path.exists() or rp.exists():
            rec = json_file(rp.parent, rp.name)
            if rec['signature'] != experiment or rec['sha256'] != digest(safe_file(path.parent, path.name)):
                raise ValueError('Temporal prediction checksum/source mismatch')
            saved = read_npz(path)
            for a, b in ((saved['pred'], prediction), (saved['keys'], d['keys'][va]), (saved['truth'], d['y'][va])):
                if not np.array_equal(a, b):
                    raise ValueError('Temporal forecast/key/target replay mismatch')
        elif replay_only:
            raise ValueError('Replay cannot replace missing predictions')
        else:
            atomic_npz(path, pred=prediction, keys=d['keys'][va], truth=d['y'][va])
            atomic_json(rp, {'signature': experiment, 'sha256': digest(path)})
    stats = game_statistics(d['y'][va], predictions['control'], predictions['direct_state'], d['keys'][va, 0])
    result = {'status': 'temporal_fold_complete', 'fold': fold, 'signature': sig,
              'experiment_signature': experiment, 'training_rows': len(tr), 'evaluation_rows': len(va),
              'evaluation_games': len(stats['rows']),
              'metrics': {a: rmse(d['y'][va], p) for a, p in predictions.items()},
              'comparison': compare(stats, ctx['round5_contract']), 'influence': influence(stats),
              'slices': slice_statistics(d['y'][va], predictions, d['keys'][va], d['role'][va]),
              'retained_columns': retained, 'fits': fits,
              'new_coordinate_models': sum(not r['reused_complete'] for r in fits),
              'old_models_refitted': 0, 'all_coordinate_replays_exact': True,
              'elapsed_seconds': round(time.monotonic() - start, 3),
              'feature_research': 'open', 'github_updated': False, 'new_kaggle_score': None}
    name = 'replay.json' if replay_only else 'summary.json'
    old_path = out / f'fold_{fold}' / 'summary.json'
    if old_path.exists():
        old = verified_fold_report(out, fold)
        for k in ('signature', 'experiment_signature', 'metrics', 'comparison', 'slices', 'retained_columns'):
            if old[k] != result[k]:
                raise ValueError('Completed temporal result no longer reproduces')
        if not replay_only:
            name = 'reuse.json'
    atomic_json(out / f'fold_{fold}' / name, result)
    if name == 'summary.json':
        atomic_json(out / f'fold_{fold}/summary.receipt.json',
                    {'sha256': digest(out / f'fold_{fold}/summary.json'),
                     'experiment_signature': experiment})
    event('temporal_fold_complete', fold=fold, metrics=result['metrics'], new_models=result['new_coordinate_models'])
    return result


def gather_predictions(ctx: dict, out: Path, fold: int, d: dict, experiment: str) -> dict:
    _, va = fold_indices(d, ctx['source']['folds'][fold - 1])
    if fold == 1:
        preds, _ = replay_discovery(ctx)
    else:
        preds = {}
        for arm in ARMS:
            path = safe_file(out, f'fold_{fold}/predictions/{arm}.npz')
            rec = json_file(path.parent, arm + '.json')
            if rec['signature'] != experiment or rec['sha256'] != digest(path):
                raise ValueError('Pooled prediction identity failed')
            z = read_npz(path)
            if not np.array_equal(z['keys'], d['keys'][va]) or not np.array_equal(z['truth'], d['y'][va]):
                raise ValueError('Pooled target/key alignment failed')
            preds[arm] = z['pred']
    return {'keys': d['keys'][va], 'y': d['y'][va], 'role': d['role'][va], **preds}


def summarize(ctx: dict, out: Path, sig: str) -> dict:
    d, fh = assemble(ctx, out, sig)
    experiment = hash_json(protocol(ctx, fh, sig))
    present = [f for f in (2, 3) if (out / f'fold_{f}/summary.json').is_file()]
    if not present:
        raise ValueError('No completed later-fold experiment')
    reports = {f: verified_fold_report(out, f) for f in present}
    for r in reports.values():
        if r['experiment_signature'] != experiment:
            raise ValueError('Later-fold result identity mismatch')
    packs = {f: gather_predictions(ctx, out, f, d, experiment) for f in [1, *present]}
    pools = {}
    for label, fs in (('later_folds_only', present), ('all_available_folds', [1, *present])):
        joined = {k: np.concatenate([packs[f][k] for f in fs]) for k in packs[1]}
        keys = joined['keys']
        if len(np.unique(keys, axis=0)) != len(keys):
            raise ValueError('Evaluation rows overlap; refuse double-counted pooled score')
        game_sets = [set(packs[f]['keys'][:, 0]) for f in fs]
        if sum(map(len, game_sets)) != len(set.union(*game_sets)):
            raise ValueError('Evaluation game folds overlap')
        stats = game_statistics(joined['y'], joined['control'], joined['direct_state'], keys[:, 0])
        pools[label] = {'folds': fs, **compare(stats, ctx['round5_contract']),
                       'influence': influence(stats),
                       'slices': slice_statistics(joined['y'], {a: joined[a] for a in ARMS}, keys, joined['role'])}
    complete = present == [2, 3]
    directions = all(r['comparison']['relative_gain'] > 0 for r in reports.values())
    gate = bool(complete and directions and pools['later_folds_only']['numerical_screen_passed'])
    stopped = (out / 'fold_3/futility.json').exists()
    result = {'status': 'temporal_replication_complete' if complete else ('stopped_for_futility' if stopped else 'temporal_replication_partial'),
              'signature': sig, 'completed_later_folds': present,
              'discovery_metrics': ctx['round4_summary']['pooled_rmse'],
              'fold_results': [{'fold': f, 'metrics': reports[f]['metrics'],
                               'comparison': reports[f]['comparison']} for f in present],
              'pools': pools, 'both_later_folds_improve': directions if complete else None,
              'feature_replication_gate_passed': gate, 'first_fold_cannot_rescue_later_failure': True,
              'new_coordinate_models_recorded': sum(r['new_coordinate_models'] for r in reports.values()),
              'old_models_refitted': 0, 'additional_fits_in_summary': 0,
              'feature_research': 'open', 'github_updated': False, 'new_kaggle_score': None,
              'scope': 'Same 1,024-play selection and previously used training-side game splits. Not untouched confirmation, cross-season evidence, or Kaggle scoring.'}
    result['milestone_review'] = {
        'attempted': 'Temporal replication of the 62 retained direct observed-state fields',
        'completed': result['status'], 'passed': {'replication_gate': gate},
        'failed_or_underperformed': [] if gate else ['The predefined later-only replication gate has not passed'],
        'actual_metric': pools['later_folds_only'],
        'saved_artifacts': 'Immutable parent references, new per-play states, per-coordinate models and predictions, fold reports, replay',
        'github_updated': False,
        'learned': 'Later-only evidence, not the inspected first-fold result, controls the decision.',
        'next': 'Review for a controlled integration into a replayable forecasting pipeline' if gate else 'Diagnose representation stability before new features or additional scaling',
        'compute_justification': 'No further fits authorized by this report.'}
    atomic_json(out / 'summary.json', result)
    return result
