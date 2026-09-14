"""Bounded CPU-only Round 2. No network, installs, Git writes, cloud jobs or submissions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

from motion_features import NAMES, FEATURES, build_features
from origin_features import KEYS
from study import atomic_json, atomic_npz, digest, hash_json, load_parent, read_npz, safe_file, screen, seal_json

KIT = Path(__file__).resolve().parent
HOME = Path('/home/sagemaker-user')
DEFAULT_REPO = HOME / 'nfl-player-trajectory'
DEFAULT_PARENT = HOME / 'nfl-feature-round1-results'
DEFAULT_OUT = HOME / 'nfl-feature-round2-results'


def log(event: str, **payload: object) -> None:
    print(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **payload}, allow_nan=False), flush=True)


def signature(contract: dict) -> str:
    files = ['run_round.py', 'study.py', 'motion_features.py', 'origin_features.py', 'input_contract.json']
    return hash_json({'files': {n: digest(KIT / n) for n in files}, 'parent': contract['parent_manifest_sha256'],
                      'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__})


def context(args):
    contract = json.loads((KIT / 'input_contract.json').read_text())
    observed = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}
    if observed != contract['parent_environment']:
        raise ValueError('Environment differs from the successful Round 1 environment. Use its existing Python 3.11 venv; do not reinstall or silently refit controls. ' + json.dumps(observed))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=args.repo, check=True, capture_output=True,
                          text=True, timeout=10).stdout.strip()
    if head != contract['repo_commit']:
        raise ValueError('Repository differs from reviewed main. Do not reset, pull or switch automatically.')
    log('checking_parent_artifacts', fitting=False)
    parent = load_parent(args.parent, contract)
    sig = signature(contract)
    seal_json(args.out / 'study_plan.json', {'signature': sig, 'parent_manifest': contract['parent_manifest_sha256'],
                                           'attribution_new_fit_cap': 18, 'motion_new_fit_cap': 9,
                                           'new_features': 60, 'original_origins_only': True})
    return parent, contract, sig


def preflight(args, parent, contract, sig):
    result = {'status': 'parent_replay_verified', 'signature': sig, 'repo_head': contract['repo_commit'],
              'parent_models_replayed': 6, 'parent_models_refitted': 0,
              'parent_checkpoints_hash_verified': parent['verified_checkpoints'],
              'original_training_side_plays': contract['original_plays'], 'original_rows': contract['original_rows'],
              'parent_control_rmse': parent['parent_summary']['pooled_rmse']['control'],
              'parent_arrival_rmse': parent['parent_summary']['pooled_rmse']['arrival'],
              'outer_validation_used': False, 'feature_research': 'open',
              'next': 'Decompose arrival features, then test three new observed-only motion families.'}
    atomic_json(args.out / 'preflight.json', result)
    log('preflight_complete', **result)


def feature_paths(out: Path, game: int, play: int):
    path = out / 'features' / 'plays' / f'{game}_{play}.npz'
    return path, path.with_suffix('.json')


def check_feature_checkpoint(path: Path, expected_keys: np.ndarray, sig: str):
    if path.with_suffix('.json').exists():
        receipt = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != receipt['sha256']:
            raise ValueError('New feature checkpoint checksum mismatch')
    z = read_npz(path)
    if str(z['signature'].item()) != sig or not np.array_equal(z['keys'], expected_keys):
        raise ValueError('New feature checkpoint source/key drift')
    if z['X'].shape != (len(expected_keys), len(NAMES)) or not np.isfinite(z['X']).all():
        raise ValueError('New feature checkpoint shape/value failure')
    if not path.with_suffix('.json').exists():
        atomic_json(path.with_suffix('.json'), {'sha256': digest(path), 'rows': len(expected_keys), 'signature': sig})
    return z


def prepare_features(args, parent, contract, sig, *, limit: int):
    """Reuse parent X/y and row order; read only original input CSVs for new signals."""
    started = time.monotonic()
    data = parent['data']
    pairs = [tuple(map(int, p)) for p in np.unique(data['keys'][:, :2], axis=0)]
    selected = pairs[:limit]
    row_by_pair = {p: np.flatnonzero((data['keys'][:, 0] == p[0]) & (data['keys'][:, 1] == p[1])) for p in selected}
    missing, reused = set(), 0
    for game, play in selected:
        path, _ = feature_paths(args.out, game, play)
        if path.exists():
            check_feature_checkpoint(path, data['keys'][row_by_pair[(game, play)]], sig)
            reused += 1
        else:
            missing.add((game, play))
    log('feature_preparation', selected_plays=len(selected), reused_plays=reused, new_features=len(NAMES), labels_read_from_raw=False)
    raw_manifest = {r['path']: r for r in contract['raw_files']}
    completed = reused
    for week in range(1, 19):
        if not missing:
            break
        name = f'data/raw/train/input_2023_w{week:02d}.csv'
        meta = raw_manifest.get(name)
        if not meta:
            raise ValueError('Verified input-file manifest is incomplete')
        rawpath = safe_file(args.repo, name)
        if digest(rawpath) != meta['sha256']:
            raise ValueError('Original observed tracking checksum changed: ' + name)
        raw = pd.read_csv(rawpath)
        pairs_index = pd.MultiIndex.from_frame(raw[KEYS[:2]])
        raw = raw.loc[pairs_index.isin(pd.MultiIndex.from_tuples(sorted(missing)))].copy()
        for raw_key, observed in raw.groupby(KEYS[:2], sort=True):
            key = tuple(map(int, raw_key))
            indices = row_by_pair[key]
            requests = pd.DataFrame(data['keys'][indices], columns=KEYS)
            features, support = build_features(observed, requests)
            path, receipt = feature_paths(args.out, *key)
            atomic_npz(path, X=features.astype(np.float32), keys=data['keys'][indices],
                       signature=np.array(sig), support=np.array(json.dumps(support, sort_keys=True)))
            check_feature_checkpoint(path, data['keys'][indices], sig)
            if not receipt.exists():
                atomic_json(receipt, {'sha256': digest(path), 'rows': len(indices), 'signature': sig})
            missing.remove(key)
            completed += 1
            if completed % 8 == 0 or not missing:
                log('feature_checkpoint', completed=completed, total=len(selected), elapsed_seconds=round(time.monotonic() - started, 2))
    if missing:
        raise ValueError('Some selected plays are absent from the verified raw input files')
    # Availability review uses only first-fold TRAIN games, not its evaluation games.
    train_games = set(parent['folds'][0]['train_games'].tolist())
    totals = {}
    manifest = []
    for key in selected:
        path, _ = feature_paths(args.out, *key)
        z = check_feature_checkpoint(path, data['keys'][row_by_pair[key]], sig)
        manifest.append({'path': path.relative_to(args.out).as_posix(), 'sha256': digest(path),
                         'game': key[0], 'play': key[1]})
        if key[0] in train_games:
            for record in json.loads(str(z['support'].item())):
                group = (record['family'], record['role'], record['window'])
                agg = totals.setdefault(group, {'players': 0, 'supported': 0, 'support_sum': 0.0, 'age_sum': 0.0})
                agg['players'] += 1; agg['supported'] += record['supported']
                agg['support_sum'] += record['fraction']; agg['age_sum'] += record['age_seconds']
    support_rows = [{'family': f, 'role': r, 'window': w, **v,
                     'available_fraction': v['supported'] / v['players'],
                     'mean_adjacent_support': v['support_sum'] / v['players']} for (f, r, w), v in sorted(totals.items())]
    name = 'smoke' if limit == 32 else 'preparation'
    if limit != 32:
        seal_json(args.out / 'features' / 'manifest.json', {'signature': sig, 'files': manifest})
    result = {'status': 'motion_smoke_complete' if limit == 32 else 'motion_features_ready',
              'signature': sig, 'plays': len(selected), 'reused_play_checkpoints': reused,
              'new_play_checkpoints': len(selected) - reused, 'candidate_features': len(NAMES),
              'candidate_families': {'turn': 24, 'brake': 24, 'orientation': 12},
              'scientific_fits': 0, 'raw_labels_read': False, 'raw_data_modified': False,
              'support_review_scope': 'first parent fold training games only',
              'training_only_support': support_rows, 'outer_validation_used': False,
              'elapsed_seconds': round(time.monotonic() - started, 3), 'feature_research': 'open'}
    atomic_json(args.out / 'features' / f'{name}_summary.json', result)
    log('feature_preparation_complete', status=result['status'], plays=len(selected), new_fits=0)
    return result


def load_extra(parent, out, sig):
    manifest = json.loads(safe_file(out, 'features/manifest.json').read_text())
    if manifest['signature'] != sig:
        raise ValueError('New feature preparation signature drift')
    keys = parent['data']['keys']
    extra = np.zeros((len(keys), len(NAMES)), np.float32)
    filled = np.zeros(len(keys), bool)
    for record in manifest['files']:
        path = safe_file(out, record['path'])
        if digest(path) != record['sha256']:
            raise ValueError('New feature manifest checksum failed')
        indices = np.flatnonzero((keys[:, 0] == record['game']) & (keys[:, 1] == record['play']))
        if not len(indices) or filled[indices].any():
            raise ValueError('Duplicate or unknown feature play')
        z = check_feature_checkpoint(path, keys[indices], sig)
        extra[indices] = z['X']; filled[indices] = True
    if not filled.all():
        raise ValueError('Not every parent forecast row has new features; do not discard missing rows')
    return extra


def review(result, args):
    passed = [d['arm'] for d in result['decisions'] if d['gate_passed']]
    unsuccessful = [d['arm'] for d in result['decisions'] if not d['gate_passed']]
    value = {'attempted': result['phase'] + ': fixed-ridge matched feature ablations',
             'completed': result['status'], 'passed': {'gates': passed, 'all_forward_replays_exact': True},
             'failed_or_underperformed': unsuccessful, 'actual_metric': result['pooled_rmse'],
             'saved': result['phase'] + '/fold_*.npz plus summaries and frozen protocol',
             'github_updated': False, 'learned': 'See family-specific additions and removals; repeated folds are exploratory.',
             'next': 'Inspect attribution and new-family evidence; integrate justified representation into the established model with matched controls, not indefinite cheap probes.',
             'compute_justification': 'Only surviving, attributable families earn a larger established-model experiment.',
             'new_fits': result['new_fits_this_invocation'], 'parent_refits': 0,
             'kaggle_score': None, 'feature_research': 'open'}
    atomic_json(args.out / result['phase'] / 'milestone_review.json', value)


def export_report(out: Path):
    names = ['preflight.json', 'last_command.json', 'last_error.json',
             'features/smoke_summary.json', 'features/preparation_summary.json']
    for phase in ('attribution', 'motion'):
        names += [f'{phase}/{name}.json' for name in ('summary', 'replay_summary', 'milestone_review')]
    path = out / 'nfl_feature_round2_report.zip'
    temporary = path.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            p = out / name
            if p.is_file() and not p.is_symlink():
                z.write(p, name)
        z.writestr('CONTENTS.txt', 'Aggregate evidence only. Excludes tracking, future coordinates, prediction keys, models, protocols with row indices, logs and credentials.\n')
    os.replace(temporary, path)
    log('report_exported', path=str(path))


def worker(args):
    if args.command == 'report':
        export_report(args.out); return
    parent, contract, sig = context(args)
    if args.command == 'preflight':
        preflight(args, parent, contract, sig)
    elif args.command == 'attribute':
        result = screen(parent, args.out, 'attribution', sig, event=log); review(result, args)
    elif args.command in ('smoke', 'prepare'):
        prepare_features(args, parent, contract, sig, limit=32 if args.command == 'smoke' else 256)
    elif args.command == 'screen':
        result = screen(parent, args.out, 'motion', sig, load_extra(parent, args.out, sig), event=log); review(result, args)
    elif args.command == 'replay':
        screen(parent, args.out, 'attribution', sig, replay_only=True, event=log)
        screen(parent, args.out, 'motion', sig, load_extra(parent, args.out, sig), replay_only=True, event=log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'attribute', 'smoke', 'prepare', 'screen', 'replay', 'report'])
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        parser.error('Hard time budget must be 1..600 seconds per command')
    args.out.mkdir(parents=True, exist_ok=True)
    for protected in (args.repo, args.parent, KIT):
        a, b = args.out.resolve(), protected.resolve()
        if a == b or a.is_relative_to(b) or b.is_relative_to(a):
            parser.error('Keep Round 2 outputs separate from repository, parent results and kit')
    if shutil.disk_usage(args.out).free < 1024**3:
        parser.error('At least 1 GiB free output storage required')
    if args.worker:
        try:
            worker(args)
        except Exception as exc:
            atomic_json(args.out / 'last_error.json', {'command': args.command, 'error_type': type(exc).__name__,
                                                      'error': str(exc), 'completed_checkpoints_preserved': True})
            raise
        return
    import fcntl
    lock = (args.out / '.round.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        parser.error('Another Round 2 command is running; do not run stages concurrently')
    env = os.environ.copy()
    env.update({k: '2' for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    env['PYTHONUNBUFFERED'] = '1'
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    (args.out / 'logs').mkdir(exist_ok=True)
    logname = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + args.command + '.log'
    start, beat = time.monotonic(), time.monotonic()
    state, rc = 'failed', 1
    with (args.out / 'logs' / logname).open('wb') as logfile:
        proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                if time.monotonic() - start >= args.seconds:
                    raise subprocess.TimeoutExpired(command, args.seconds)
                events = selector.select(timeout=0.25)
                for key, _ in events:
                    payload = os.read(key.fileobj.fileno(), 65536)
                    if payload:
                        logfile.write(payload); logfile.flush()
                        sys.stdout.write(payload.decode('utf-8', errors='replace')); sys.stdout.flush()
                if proc.poll() is not None:
                    remaining = proc.stdout.read()
                    logfile.write(remaining)
                    sys.stdout.write(remaining.decode('utf-8', errors='replace')); sys.stdout.flush()
                    rc = proc.returncode; state = 'completed' if rc == 0 else 'failed'; break
                if time.monotonic() - beat >= 15:
                    log('heartbeat', stage=args.command, elapsed_seconds=round(time.monotonic() - start, 1), limit_seconds=args.seconds)
                    beat = time.monotonic()
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            state = 'stopped_budget' if isinstance(exc, subprocess.TimeoutExpired) else 'interrupted'
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            rc = 124
        finally:
            selector.close()
            proc.stdout.close()
            atomic_json(args.out / 'last_command.json', {'command': args.command, 'status': state, 'exit_code': rc,
                        'hard_limit_seconds': args.seconds, 'elapsed_seconds': round(time.monotonic() - start, 3),
                        'cloud_resource_changes': 0, 'github_writes': 0, 'new_kaggle_submissions': 0,
                        'partial_checkpoints_retained': True})
    lock.close()
    if rc:
        raise SystemExit(rc)


if __name__ == '__main__':
    main()
