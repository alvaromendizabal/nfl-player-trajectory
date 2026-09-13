# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Frozen first-fold feature attribution. No training of old models or extra folds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from origin_features import BASE_NAMES, rmse
from state_features import DIRECT_NAMES, GOAL_NAMES
from parent_support import (atomic_json,atomic_npz,seal_json,hash_json,digest,safe_file,read_npz)
from tree_core import SETTINGS, versions, fit_axis, paired, event
from study import (parent_context,replay_control,study_signature,prepare,assemble,json_file)

ARMS=('direct_state','goal_geometry')


def new_matrices(d:dict)->dict[str,np.ndarray]:
    base=d['X'][:,:len(BASE_NAMES)]
    direct=np.column_stack([base,d['state']])
    return {'direct_state':direct,'goal_geometry':np.column_stack([direct,d['goal']])}


def fit_study(ctx:dict,out:Path,sig:str,*,replay_only:bool=False)->dict:
    started=time.monotonic()
    control,replay=replay_control(ctx)
    d,features_hash=assemble(ctx,out,sig)
    tr=np.flatnonzero(np.isin(d['keys'][:,0],ctx['fold']['train_games']))
    va=np.flatnonzero(np.isin(d['keys'][:,0],ctx['fold']['validation_games']))
    if not np.array_equal(d['keys'][va],ctx['data']['keys'][ctx['evaluation']]):
        raise ValueError('The new comparison changes the parent evaluation row order')
    protocol={'source_signature':sig,'parent_data_sha256':ctx['source']['sha256'],
              'features_hash':features_hash,'settings':SETTINGS,'environment':versions(),
              'fold':ctx['fold'],'arms':list(ARMS),'new_coordinate_models_max':4,
              'selected_from_errors':False,'outer_holdout_used':False,
              'schema':{'direct':list(DIRECT_NAMES),'goal':list(GOAL_NAMES)}}
    seal_json(out/'experiment_protocol.json',protocol)
    study_sig=hash_json(protocol)
    predictions={'control':control};fits=[];keeps={}
    for arm,m in new_matrices(d).items():
        keep=np.ptp(m[tr],axis=0)>1e-10
        if not keep.any():raise ValueError('No variable training features')
        keeps[arm]=keep
        xt=np.ascontiguousarray(m[tr][:,keep],dtype=np.float64)
        xe=np.ascontiguousarray(m[va][:,keep],dtype=np.float64)
        pair_pred=[]
        for axis in range(2):
            fit_sig=hash_json({'study':study_sig,'arm':arm,'axis':axis,'keep':keep.tolist()})
            folder=out/'models'/arm/str(axis)
            if replay_only and not folder.is_dir():raise ValueError('Replay refuses a missing model')
            pred,receipt=fit_axis(folder,xt,d['y'][tr,axis],xe,fit_sig,replay_only=replay_only)
            pair_pred.append(pred);fits.append({'arm':arm,'axis':axis,**receipt})
        prediction=np.column_stack(pair_pred);predictions[arm]=prediction
        path=out/'predictions'/(arm+'.npz');receipt_path=path.with_suffix('.json')
        if path.exists() or receipt_path.exists():
            receipt=json_file(path.parent,receipt_path.name)
            if receipt['signature']!=study_sig or digest(safe_file(path.parent,path.name))!=receipt['sha256']:
                raise ValueError('Prediction receipt/hash mismatch')
            old=read_npz(path)
            for a,b in ((old['pred'],prediction),(old['keys'],d['keys'][va]),(old['truth'],d['y'][va])):
                if not np.array_equal(a,b):raise ValueError('Prediction numerical replay or key/label parity failed')
        elif replay_only:raise ValueError('Replay cannot replace missing prediction evidence')
        else:
            atomic_npz(path,pred=prediction,keys=d['keys'][va],truth=d['y'][va])
            atomic_json(receipt_path,{'signature':study_sig,'sha256':digest(path)})
    contrasts=[]
    for treatment,control_name in [('direct_state','control'),('goal_geometry','direct_state'),('goal_geometry','control')]:
        contrast=paired(d['y'][va],predictions[control_name],predictions[treatment],d['keys'][va,0])
        contrasts.append({'treatment':treatment,'control':control_name,**contrast})
    promoted=[]
    if contrasts[0]['screen_gate']:promoted.append('direct_state')
    if contrasts[1]['screen_gate'] and contrasts[2]['screen_gate']:promoted.append('goal_geometry')
    slices=[]
    for arm,p in predictions.items():
        masks=[('first_second',d['keys'][va,3]<=10),('after_first_second',d['keys'][va,3]>10)]
        masks += [(f'role_{i}',d['role'][va]==i) for i in range(4)]
        for name,mask in masks:
            if mask.any():slices.append({'arm':arm,'slice':name,'rows':int(mask.sum()),
                'sse':float(np.square(d['y'][va][mask]-p[mask]).sum()),'rmse':rmse(d['y'][va][mask],p[mask])})
    result={'status':'round4_first_fold_complete','scope':'Same reused first-fold rows as Round 3; exploratory feature attribution, not a Kaggle score.',
            'source_signature':sig,'experiment_signature':study_sig,'fold':1,
            'training_rows':len(tr),'evaluation_rows':len(va),'evaluation_games':len(np.unique(d['keys'][va,0])),
            'pooled_rmse':{arm:rmse(d['y'][va],p) for arm,p in predictions.items()},
            'retained_columns':{arm:int(k.sum()) for arm,k in keeps.items()},
            'candidate_columns':{'direct_state':len(BASE_NAMES)+len(DIRECT_NAMES),
                                 'goal_geometry':len(BASE_NAMES)+len(DIRECT_NAMES)+len(GOAL_NAMES)},
            'contrasts':contrasts,'screen_passed_arms':promoted,'slices':slices,'fits':fits,
            'parent_replay':replay,'new_coordinate_models':sum(not f['reused_complete'] for f in fits),
            'parent_refits':0,'all_new_prediction_replays_exact':True,
            'new_kaggle_score':None,'github_updated':False,'aws_resource_changes':0,
            'elapsed_seconds':round(time.monotonic()-started,3),'feature_research':'open'}
    result['milestone_review']={'attempted':'Direct observed state, followed by compact landing-frame geometry',
        'completed':result['status'],'passed':promoted,'failed_or_underperformed':[a for a in ARMS if a not in promoted],
        'actual_metrics':result['pooled_rmse'],'saved_artifacts':'Per-coordinate models, prediction arrays, summary and no-fit replay',
        'github_updated':False,'learned':'Use paired differences; two raw input feature groups, fixed estimator, identical rows.',
        'next':'Return report before any larger or additional-fold run.',
        'compute_justification':'Only a supported representation merits a broader existing-model validation; no unchanged failures are scaled.'}
    summary_path=out/'summary.json'
    if summary_path.exists():
        old=json.loads(summary_path.read_text())
        for key in ('source_signature','experiment_signature','pooled_rmse','contrasts','slices','training_rows','evaluation_rows'):
            if old[key]!=result[key]:raise ValueError('Completed scientific summary no longer reproduces')
    name='replay.json' if replay_only else ('reuse.json' if summary_path.exists() else 'summary.json')
    atomic_json(out/name,result)
    event('round4_complete',metrics=result['pooled_rmse'],new_models=result['new_coordinate_models'],screen_passed_arms=promoted)
    return result


def execute(args):
    c=json.loads(safe_file(args.kit,'input_contract.json').read_text())
    ctx=parent_context(args.parent,args.legacy,args.repo,c)
    sig=study_signature(args.kit,ctx)
    seal_json(args.out/'plan.json',{'signature':sig,'parent_source_signature':c['parent_source_signature'],
        'parent_data_sha256':ctx['source']['sha256'],'settings':SETTINGS,'fold':1,
        'feature_groups':[len(DIRECT_NAMES),len(GOAL_NAMES)],'new_coordinate_models_max':4,
        'no_further_fold_execution':True,'no_augmented_rows':True})
    if args.command=='preflight':
        _,rec=replay_control(ctx)
        free=shutil.disk_usage(args.out).free/1024**3
        if free<2:raise ValueError('At least 2 GiB free space is required')
        result={'status':'round4_preflight_passed','signature':sig,'parent_replay':rec,
                'training_rows':len(ctx['train']),'evaluation_rows':len(ctx['evaluation']),
                'environment':versions(),'free_gib':round(free,2),'raw_output_csvs_opened':0,
                'new_model_fits':0,'github_updated':False,'new_kaggle_score':None,'feature_research':'open'}
        atomic_json(args.out/'preflight.json',result);event('preflight_complete',**result)
    else:
        pre=json_file(args.out,'preflight.json')
        if pre['signature']!=sig or pre['status']!='round4_preflight_passed':raise ValueError('Pass the current preflight first')
        if args.command in ('smoke','prepare'):
            if args.command=='prepare':
                s=json_file(args.out,'smoke.json')
                if s['status']!='state_smoke_complete' or s['signature']!=sig:raise ValueError('Pass the bounded training smoke first')
            prepare(ctx,args.repo,args.out,sig,smoke=args.command=='smoke',event=event)
        elif args.command in ('fit','replay'):
            p=json_file(args.out,'preparation.json')
            if p['status']!='state_features_ready' or p['signature']!=sig:raise ValueError('Feature preparation incomplete')
            fit_study(ctx,args.out,sig,replay_only=args.command=='replay')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','smoke','prepare','fit','replay'])
    for name in ('kit','parent','legacy','repo','out'):p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())

if __name__=='__main__':main()
