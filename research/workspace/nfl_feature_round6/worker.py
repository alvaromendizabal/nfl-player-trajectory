# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Execute one selected pass-axis stage; never launch cloud resources."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
from experiment import parent,signature,read,replay_parent,prepare,run_fold,summarize,assemble
from parent_support import seal_json,atomic_json


def execute(args):
    ctx=parent(args);sig=signature(ctx)
    from tree_core import SETTINGS,versions,event
    plan={'signature':sig,'study':'pass-axis-state-and-history-v1',
        'parent_data_sha256':ctx['source']['sha256'],'parent_experiment':ctx['r5_experiment'],
        'folds':[2,3],'settings':SETTINGS,'environment':versions(),'max_new_coordinate_models':8,
        'selected_plays':len(ctx['selection']['plays']),'new_feature_definitions':48,
        'old_models_refitted':0,'kaggle_submissions':0}
    seal_json(args.out/'plan.json',plan)
    if args.command=='preflight':
        if shutil.disk_usage(args.out).free < 2*1024**3:raise ValueError('Need 2 GiB free; preserve previous artifacts')
        records=[]
        for f in (2,3):
            _,r=replay_parent(ctx,f,('control','direct_state'));records.extend(r)
        result={'status':'pass_axis_preflight_passed','signature':sig,'parent_models_replayed':len(records),
                'parent_replay':records,'old_models_refitted':0,'new_model_fits':0,
                'environment':versions(),'fold_populations':ctx['populations'],
                'round5_replication_gate_passed':False,'feature_research':'open','github_updated':False}
        name='preflight_reuse.json' if (args.out/'preflight.json').exists() else 'preflight.json'
        atomic_json(args.out/name,result)
    else:
        pre=read(args.out,'preflight.json')
        if pre['signature']!=sig or pre['status']!='pass_axis_preflight_passed':raise ValueError('Run preflight first')
        if args.command in ('smoke','prepare'):
            if args.command=='prepare':
                sm=read(args.out,'smoke.json')
                if sm['signature']!=sig or sm['status']!='pass_axis_smoke_passed':raise ValueError('Run 32-play smoke first')
            result=prepare(ctx,args.out,sig,smoke=args.command=='smoke')
        else:
            pr=read(args.out,'preparation.json')
            if pr['signature']!=sig or pr['status']!='pass_axis_features_ready':raise ValueError('Complete feature preparation first')
            if args.command=='fit':result=run_fold(ctx,args.out,sig,args.fold)
            elif args.command=='replay':
                folds=[f for f in (2,3) if (args.out/f'fold_{f}/summary.json').exists()]
                if not folds:raise ValueError('Replay cannot fit missing studies')
                for f in (2,3):
                    if (args.out/f'fold_{f}/models').exists() and f not in folds:raise ValueError('An incomplete fold must be resumed/diagnosed before replay')
                rr=[run_fold(ctx,args.out,sig,f,replay_only=True) for f in folds]
                if any(r['new_coordinate_models'] for r in rr):raise ValueError('Replay fitted unexpectedly')
                summarize(ctx,args.out,sig)
                result={'status':'pass_axis_replay_exact','signature':sig,'folds':folds,
                        'new_models_replayed':len(folds)*4,'new_coordinate_models':0,'old_models_refitted':0}
                atomic_json(args.out/'replay.json',result)
            else:result=summarize(ctx,args.out,sig)
    event('stage_complete',stage=args.command,status=result['status'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','smoke','prepare','fit','replay','summarize'])
    p.add_argument('--fold',type=int,choices=[2,3],default=2)
    for name in ('kit','round5','round5-kit','round4','round4-kit','round3','round3-kit','repo','out'):
        p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())

if __name__=='__main__':main()
