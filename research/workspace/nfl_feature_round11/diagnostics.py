"""Frozen final-checkpoint training audit. No optimizer is created or called."""
from __future__ import annotations
import hashlib
import platform
import numpy as np
import torch
from audit_io import hash_json, safe_file, read_json, seal_json, checkpoint_npz, digest
from core import context, load_play, unchanged
from grouped_model import Forecast, collate, setup


def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'device':'cpu','threads':2,'machine':platform.machine()}

def aggregate(y,p,frame,role):
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite matching target/prediction rows required')
    sse=(y.astype(float)-p.astype(float))**2
    total={'rows':len(y),'sse':float(sse.sum()),'rmse':float(np.sqrt(sse.mean()))}
    slices=[]
    masks={'first_second':frame<=10,'after_first_second':frame>10}
    masks.update({f'role_{int(r)}':role==r for r in np.unique(role)})
    for name,mask in masks.items():
        n=int(mask.sum())
        if n:slices.append({'slice':name,'rows':n,'sse':float(sse[mask].sum()),'rmse':float(np.sqrt(sse[mask].mean()))})
    return total,slices

def run_audit(a,replay=False):
    v=context(a);setup()
    if environment()!=v['protocol']['environment']:raise ValueError('Exact parent numerical runtime required; no automatic installation')
    if read_json(safe_file(a.out,'feature_summary.json'))['signature']!=v['signature']:
        raise ValueError('Current feature audit must finish first')
    models={}
    for arm,x in v['checkpoints'].items():
        model=Forecast()
        state=torch.load(x['path'],map_location='cpu',weights_only=True)
        if state['signature']!=x['receipt']['signature'] or state['step']!=x['receipt']['step'] or state['initial_hash']!=x['complete']['initial_hash']:
            raise ValueError('Loaded checkpoint lineage differs')
        model.load_state_dict(state['model'],strict=True);model.eval();models[arm]=model
        del state
    pred={k:[] for k in models};targets=[];allkeys=[];allroles=[]
    norm=v['protocol']['normalization'];records=v['train']
    for start in range(0,len(records),8):
        chunk=records[start:start+8];plays=[load_play(a,r,v,labels=True) for r in chunk]
        keys=np.concatenate([p['keys'] for p in plays]);y=np.concatenate([p['y'] for p in plays]).astype(float)
        roles=np.concatenate([p['role_query'] for p in plays]);batch_preds={}
        for arm,model in models.items():
            with torch.no_grad():
                out=model(**collate(plays,arm,norm)).numpy().astype(np.float64)
            if start==0 and hashlib.sha256(out.tobytes()).hexdigest()!=v['checkpoints'][arm]['complete']['probe_sha256']:
                raise ValueError('Original first-eight training probe differs: '+arm)
            pred[arm].append(out);batch_preds[arm]=out
        folder=a.out/'training_predictions';path=folder/f'{start:05d}.npz'
        if replay and not path.is_file():raise ValueError('Replay refuses missing prediction checkpoints')
        checkpoint_npz(path,{'keys':keys,**batch_preds})
        targets.append(y);allkeys.append(keys);allroles.append(roles)
        if start==0 or (start//8+1)%16==0 or start+8>=len(records):
            import json
            print(json.dumps({'event':'frozen_training_replay','plays':min(start+8,len(records)),'total_plays':len(records),'optimizer_steps':0}),flush=True)
    keys=np.concatenate(allkeys);y=np.concatenate(targets);role=np.concatenate(allroles)
    if len(keys)!=v['contract']['training_rows'] or len(np.unique(keys,axis=0))!=len(keys):raise ValueError('Training rows lost or duplicated')
    metrics={};slices=[]
    for arm,pp in pred.items():
        total,sl=aggregate(y,np.concatenate(pp),keys[:,3],role)
        metrics[arm]=total
        slices.extend({'arm':arm,**s} for s in sl)
    z,sl=aggregate(y,np.zeros_like(y),keys[:,3],role);metrics['zero_residual_reference']=z
    slices.extend({'arm':'zero_residual_reference',**s} for s in sl)
    report={'status':'frozen_training_audit_complete','signature':v['signature'],
      'environment':environment(),'training_rows':len(keys),'training_plays':len(records),
      'training_games':len(np.unique(keys[:,0])),'metrics':metrics,'slices':slices,
      'historical_evaluation_metrics':v['summary']['metrics'],'historical_evaluation_rows':v['summary']['evaluation_rows'],
      'evaluation_predictions_recomputed':False,'evaluation_arrays_opened':False,'training_labels_read':True,
      'numerical_replay_scope':'Every selected training row. Historical evaluation scores are read from the completed summary only.',
      'new_optimizer_steps':0,'new_model_fits':0,'candidate_features_used_by_models':False,
      'interpretation':'In-sample error is descriptive, not a validation gain. No universal train/evaluation-gap threshold identifies the cause.',
      'zero_reference_meaning':'Zero prediction in the existing stored residual target space; not a fitted tree or a leaderboard baseline.',
      'first_eight_training_probes_exact':True,'feature_research':'open'}
    if replay:
        if read_json(safe_file(a.out,'training_audit.json'))!=report:raise ValueError('Frozen training summary changed')
    else:seal_json(a.out/'training_audit.json',report)
    unchanged(v['protected'])
    return report
