# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Run one explicitly selected stage; no cloud writes or automatic submissions."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from history_study import (context,signature,read,replay_parent,smoke,prepare,
                           run_fold,summarize,build_fold)
from parent_support import seal_json,atomic_json
from history_features import CONFIG,NAMES


def execute(args):
    ctx=context(args);sig=signature(ctx)
    from tree_core import SETTINGS,versions,event
    plan={'signature':sig,'study':'ordered-role-player-response-v1',
          'parent_data_sha256':ctx['source']['sha256'],'parent_experiment':ctx['r5_experiment'],
          'folds':[2,3],'settings':SETTINGS,'environment':versions(),'history_config':CONFIG,
          'max_new_coordinate_models':12,'new_feature_definitions':len(NAMES),
          'selected_plays':len(ctx['selection']['plays']),'old_models_refitted':0,
          'external_data':False,'kaggle_submissions':0,'raw_csv_reads':0}
    seal_json(args.out/'plan.json',plan)
    if args.command=='preflight':
        if shutil.disk_usage(args.out).free<2*1024**3:raise ValueError('Need 2 GiB free; preserve older artifacts')
        records=[]
        for fold in (2,3):
            _,r=replay_parent(ctx,fold);records.extend(r)
        result={'status':'history_preflight_passed','signature':sig,
                'parent_models_replayed':len(records),'parent_replay':records,
                'old_models_refitted':0,'new_coordinate_models':0,'raw_csv_reads':0,
                'environment':versions(),'fold_populations':ctx['populations'],
                'feature_research':'open','github_updated':False}
        name='preflight_reuse.json' if (args.out/'preflight.json').exists() else 'preflight.json'
        atomic_json(args.out/name,result)
    else:
        pre=read(args.out,'preflight.json')
        if pre['signature']!=sig or pre['status']!='history_preflight_passed':raise ValueError('Run preflight first')
        if args.command=='smoke':result=smoke(ctx,args.out,sig)
        elif args.command=='prepare':
            r=read(args.out,'smoke.json')
            if r['signature']!=sig or r['status']!='history_smoke_passed':raise ValueError('Complete training smoke first')
            result=prepare(ctx,args.out,sig)
        else:
            r=read(args.out,'preparation.json')
            if r['signature']!=sig or r['status']!='history_features_ready':raise ValueError('Complete feature preparation first')
            if args.command=='fit':result=run_fold(ctx,args.out,sig,args.fold)
            elif args.command=='replay':
                present=[f for f in (2,3) if (args.out/f'fold_{f}/summary.json').exists()]
                if not present:raise ValueError('Replay cannot fit missing studies')
                for f in (2,3):
                    if (args.out/f'fold_{f}/models').exists() and f not in present:raise ValueError('Resume/diagnose an incomplete fold before replay')
                enc=[build_fold(ctx,args.out,sig,f,replay_only=True) for f in (2,3)]
                atomic_json(args.out/'encoder_replay.json',{'signature':sig,'status':'ordered_encoder_replay_exact','folds':enc})
                reports=[run_fold(ctx,args.out,sig,f,replay_only=True) for f in present]
                if any(r['new_coordinate_models'] for r in reports):raise ValueError('Replay unexpectedly fitted')
                summarize(ctx,args.out,sig)
                result={'status':'history_replay_exact','signature':sig,'folds':present,
                        'new_models_replayed':len(present)*6,'new_coordinate_models':0,
                        'old_models_refitted':0,'historical_encoders_reconstructed_exactly':2}
                atomic_json(args.out/'replay.json',result)
            else:result=summarize(ctx,args.out,sig)
    event('stage_complete',stage=args.command,status=result['status']);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','smoke','prepare','fit','replay','summarize'])
    p.add_argument('--fold',type=int,choices=[2,3],default=2)
    for name in ('kit','round6','round5','round5-kit','round4','round4-kit','round3','round3-kit','repo','out'):
        p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())

if __name__=='__main__':main()
