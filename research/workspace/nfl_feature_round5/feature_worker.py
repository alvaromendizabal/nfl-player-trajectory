# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Run one explicitly selected stage, never an unbounded research loop."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from confirmation import (context, signature, replay_discovery, prepare, assemble,
                          run_fold, summarize, json_file, influence, game_statistics, fold_indices)
from parent_support import atomic_json, seal_json
from tree_core import SETTINGS, versions, event


def execute(args: argparse.Namespace) -> dict:
    ctx = context(args.kit, args.round4, args.round4_kit, args.round3,
                  args.round3_kit, args.repo)
    sig = signature(ctx)
    seal_json(args.out / 'plan.json', {
        'signature': sig, 'parent_data_sha256': ctx['source']['sha256'],
        'round4_experiment_signature': ctx['round5_contract']['round4_experiment_signature'],
        'settings': SETTINGS, 'folds': ctx['source']['folds'], 'new_folds': [2, 3],
        'arms': ['control', 'direct_state'], 'max_new_coordinate_models': 8,
        'new_feature_definitions': 0, 'old_models_refitted': 0})
    if args.command == 'preflight':
        predictions, replay = replay_discovery(ctx)
        _, va = fold_indices(ctx['data'], ctx['source']['folds'][0])
        stats = game_statistics(ctx['data']['y'][va], predictions['control'],
                                predictions['direct_state'], ctx['data']['keys'][va, 0])
        free = shutil.disk_usage(args.out).free / 1024**3
        if free < 2:
            raise ValueError('At least 2 GiB of free storage is required; do not delete old results')
        result = {'status': 'temporal_preflight_passed', 'signature': sig,
                  'parent_replay': replay, 'discovery_leave_one_game_out': influence(stats),
                  'fold_populations': ctx['populations'], 'environment': versions(),
                  'raw_output_csvs_opened': 0, 'new_model_fits': 0, 'free_gib': round(free, 2),
                  'feature_research': 'open', 'github_updated': False, 'new_kaggle_score': None}
        filename = 'preflight_reuse.json' if (args.out / 'preflight.json').exists() else 'preflight.json'
        if filename != 'preflight.json' and json_file(args.out, 'preflight.json')['signature'] != sig:
            raise ValueError('Preflight belongs to another study')
        atomic_json(args.out / filename, result)
    else:
        pre = json_file(args.out, 'preflight.json')
        if pre['status'] != 'temporal_preflight_passed' or pre['signature'] != sig:
            raise ValueError('Complete the current preflight first')
        if args.command in ('smoke', 'prepare'):
            if args.command == 'prepare':
                sm = json_file(args.out, 'smoke.json')
                if sm['status'] != 'temporal_state_smoke_passed' or sm['signature'] != sig:
                    raise ValueError('Complete the 32-play smoke before feature preparation')
            result = prepare(ctx, args.repo, args.out, sig,
                             smoke=args.command == 'smoke', event=event)
        else:
            prepared = json_file(args.out, 'preparation.json')
            if prepared['status'] != 'temporal_state_ready' or prepared['signature'] != sig:
                raise ValueError('Complete state preparation first')
            if args.command == 'fit':
                result = run_fold(ctx, args.out, sig, args.fold)
            elif args.command == 'replay':
                _, old = replay_discovery(ctx)
                folds = [f for f in (2, 3) if (args.out / f'fold_{f}/summary.json').is_file()]
                if not folds:
                    raise ValueError('Replay refuses to fit missing temporal results')
                # A partially fitted fold must be diagnosed/resumed, not hidden by replay.
                for f in (2, 3):
                    if (args.out / f'fold_{f}/models').exists() and f not in folds:
                        raise ValueError('An incomplete fold has model checkpoints; complete it before replay')
                receipts = [run_fold(ctx, args.out, sig, f, replay_only=True) for f in folds]
                if sum(r['new_coordinate_models'] for r in receipts) != 0:
                    raise ValueError('Replay unexpectedly fitted a model')
                summary = summarize(ctx, args.out, sig)
                result = {'status': 'temporal_replay_exact', 'signature': sig,
                          'completed_later_folds': folds, 'new_coordinate_models': 0,
                          'old_models_refitted': 0, 'parent_replay': old,
                          'new_models_replayed': len(folds) * 4,
                          'feature_replication_gate_passed': summary['feature_replication_gate_passed']}
                atomic_json(args.out / 'replay.json', result)
            else:
                result = summarize(ctx, args.out, sig)
    event('stage_complete', command=args.command, status=result['status'])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'smoke', 'prepare', 'fit', 'replay', 'summarize'])
    p.add_argument('--fold', type=int, choices=[2, 3], default=2)
    for name in ('kit', 'round4', 'round4-kit', 'round3', 'round3-kit', 'repo', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    execute(p.parse_args())


if __name__ == '__main__':
    main()
