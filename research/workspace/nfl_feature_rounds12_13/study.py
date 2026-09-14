"""Two predeclared independent 3-arm studies; no selection using the other round.

The same 699 training plays and 14 reused evaluation games are retained.
Six predeclared contrasts share a Bonferroni family; this is not a correction for
all historical adaptive choices in this project. One seed; no Kaggle result.
"""
from __future__ import annotations
import hashlib,json,platform,time
import numpy as np
import torch
from audit_io import (digest,hash_json,safe_file,read_json,atomic_json,seal_json,checkpoint_npz)
from bridge import load_split,evaluation_data
from checkpoints import save_checkpoint,load_checkpoint,weight_hash
from model import CONFIG,Forecast,setup,normalization,ordered_batches,step,collate,predict
from families import ARMS

CONTRASTS=(('mask','core'),('core','full'),('mask','full'))

def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'device':'cpu','threads':CONFIG['threads'],'machine':platform.machine()}

def protocol(a):
    setup();train,v=load_split(a,True)
    if environment()!=v['contract']['environment']:raise ValueError('Pinned parent environment differs; do not install different versions')
    games={int(p['keys'][0,0]) for p in train};eg=set(v['evaluation_games'])
    if games&eg or max(games)//100>=min(eg)//100:raise ValueError('Invalid chronological partition')
    p={'study':'direct-player-family-v1','round':a.round_no,'config':CONFIG,'environment':environment(),
       'data_sha256':digest(a.out/'dataset.json'),'normalization':normalization(train),'arms':list(ARMS),
       'parameters':sum(x.numel() for x in Forecast().parameters()),
       'optimizer_steps_per_arm':len(ordered_batches(len(train),0))*CONFIG['epochs'],
       'six_contrast_family':True,'mask_policy':'Identical availability masks in mask/core/full; only numeric values ablated',
       'selection':'fixed final optimizer exposure, no validation checkpoint selection',
       'independence':'No Round 12 scores/features/models feed Round 13, or vice versa'}
    seal_json(a.out/'protocol.json',p);return train,p,hash_json(p)

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)

def profile(a):
    train,p,psig=protocol(a)
    # Profile maximum-shape rows among the selected training examples only.
    items=sorted(train,key=lambda x:(len(x['ids']),len(x['base'])),reverse=True)[:8]
    y=torch.tensor(np.concatenate([x['y'] for x in items]),dtype=torch.float32)
    timings={};gradient=None
    for arm in ARMS:
        setup();model=Forecast();opt=optimizer(model);durations=[]
        for i in range(8):
            before=time.monotonic();b=collate(items,arm,p['normalization']);step(model,opt,b,y,2*len(y))
            if i:durations.append(time.monotonic()-before)
        sec=float(np.quantile(durations,.9));timings[arm]={'p90_seconds':sec,'projected_seconds':sec*p['optimizer_steps_per_arm']*1.5+60}
        if arm=='full':
            model.zero_grad(set_to_none=True);b=collate(items,arm,p['normalization']);b['node'].requires_grad_()
            pred=model(**b);(pred[:,0].sum()-pred[:,1].sum()).backward()
            g=b['node'].grad[...,10:];valid=b['node_valid'][...,10:]
            if not valid.any() or not torch.isfinite(g).all() or not (g[valid]!=0).any():raise ValueError('No finite local pathway through the new values')
            gradient=float(g[valid].square().mean().sqrt())
    passed=all(t['projected_seconds']<=300 for t in timings.values())
    r={'status':'family_profile_passed' if passed else 'family_profile_budget_stop','round':a.round_no,
       'protocol_signature':psig,'timings':timings,'training_plays':len(items),'disposable_optimizer_steps':24,
       'local_input_gradient_rms':gradient,'gradient_is_predictive_evidence':False,
       'evaluation_arrays_opened':False,'scientific_models_fitted':0,'budget_guard_seconds':300}
    atomic_json(a.out/'profile.json',r);return r

def train_arm(a,arm,max_steps=None):
    if arm not in ARMS:raise ValueError('Unknown arm')
    train,p,psig=protocol(a);g=read_json(safe_file(a.out,'profile.json'))
    if g['status']!='family_profile_passed' or g['protocol_signature']!=psig:raise ValueError('Current profile must pass')
    setup();model=Forecast();initial=weight_hash(model);opt=optimizer(model)
    sig=hash_json({'protocol':psig,'arm':arm});folder=a.out/'models'/arm
    nb=len(ordered_batches(len(train),0));total=nb*CONFIG['epochs'];cursor=0;losses=[]
    if (folder/'checkpoint.json').is_file():
        state=load_checkpoint(folder,model,opt,sig);cursor=state['step'];losses=state['losses']
        if state['initial_hash']!=initial or not 0<=cursor<=total or len(losses)!=cursor:raise ValueError('Invalid resumed initialization/cursor')
    limit=total if max_steps is None else min(total,cursor+max_steps)
    den=2*sum(len(x['y']) for x in train)/nb;began=time.monotonic()
    for i in range(cursor,limit):
        ix=ordered_batches(len(train),i//nb)[i%nb];items=[train[j] for j in ix]
        y=torch.tensor(np.concatenate([x['y'] for x in items]),dtype=torch.float32)
        losses.append(step(model,opt,collate(items,arm,p['normalization']),y,den))
        if (i+1)%CONFIG['checkpoint_steps']==0 or (i+1)%nb==0 or i+1==limit:
            save_checkpoint(folder,model,opt,i+1,sig,losses,initial)
            print(json.dumps({'event':'optimizer_checkpoint','round':a.round_no,'arm':arm,'steps':i+1,'total':total,
                              'elapsed_seconds':round(time.monotonic()-began,2)}),flush=True)
    done=max(cursor,limit)
    if done<total:return {'status':'incomplete_checkpoint_preserved','new_optimizer_steps':done-cursor,'steps':done}
    probe=predict(model,train[:8],arm,p['normalization']);other=Forecast()
    load_checkpoint(folder,other,optimizer(other),sig)
    if not np.array_equal(probe,predict(other,train[:8],arm,p['normalization'])):raise ValueError('Fresh model training-probe replay failed')
    r={'status':'family_arm_complete','round':a.round_no,'arm':arm,'protocol_signature':psig,'initial_hash':initial,
       'steps':total,'new_optimizer_steps':total-cursor,'parameters':p['parameters'],
       'training_epoch_objectives':[sum(losses[i:i+nb])/nb for i in range(0,total,nb)],
       'probe_sha256':hashlib.sha256(probe.tobytes()).hexdigest(),'evaluation_arrays_opened':False}
    if (folder/'complete.json').exists():
        old=read_json(folder/'complete.json')
        for k in ('protocol_signature','initial_hash','steps','probe_sha256','training_epoch_objectives'):
            if r[k]!=old[k]:raise ValueError('Completed arm changed on reuse')
        atomic_json(folder/'reuse.json',r)
    else:seal_json(folder/'complete.json',r)
    return r

def metric(y,p):
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite nonempty aligned coordinate rows required')
    sse=float(((y.astype(float)-p.astype(float))**2).sum());return {'rows':len(y),'sse':sse,'rmse':float(np.sqrt(sse/(2*len(y))))}

def contrast(y,a,b,games):
    ca,cb=metric(y,a)['rmse'],metric(y,b)['rmse']
    if len(games)!=len(y) or ca==0:raise ValueError('Invalid game keys or zero control RMSE')
    u,ix=np.unique(games,return_inverse=True)
    if len(u)<2:raise ValueError('At least two games required')
    n=np.bincount(ix);sa=np.bincount(ix,weights=((y-a)**2).sum(1));sb=np.bincount(ix,weights=((y-b)**2).sum(1))
    rng=np.random.default_rng(20260912);sample=rng.integers(0,len(u),(20000,len(u)))
    d=np.sqrt(sb[sample].sum(1)/(2*n[sample].sum(1)))-np.sqrt(sa[sample].sum(1)/(2*n[sample].sum(1)))
    lo,hi=np.quantile(d,[.05/(2*6),1-.05/(2*6)]);gain=1-cb/ca
    return {'delta_rmse':cb-ca,'relative_gain':gain,'low_adjusted':float(lo),'high_adjusted':float(hi),
            'resamples':20000,'planned_family_comparisons':6,'passes':bool(gain>=.01 and hi<0),
            'scope':'Exploratory reused games, one seed; adjustment only for these two frozen rounds'}

def evaluate(a,replay=False):
    train,p,psig=protocol(a)
    records={arm:read_json(safe_file(a.out,f'models/{arm}/complete.json')) for arm in ARMS}
    if len({r['initial_hash'] for r in records.values()})!=1:raise ValueError('Unmatched initialization')
    if any(r['steps']!=p['optimizer_steps_per_arm'] or r['protocol_signature']!=psig for r in records.values()):raise ValueError('Unmatched exposure')
    ev,ref=evaluation_data(a);preds={};training_metrics={}
    for arm in ARMS:
        setup();m=Forecast();state=load_checkpoint(a.out/'models'/arm,m,optimizer(m),hash_json({'protocol':psig,'arm':arm}))
        if state['step']!=p['optimizer_steps_per_arm']:raise ValueError('Checkpoint is incomplete')
        probe=predict(m,train[:8],arm,p['normalization'])
        if hashlib.sha256(probe.tobytes()).hexdigest()!=records[arm]['probe_sha256']:raise ValueError('Final training-probe mismatch')
        # Final training predictions are descriptive; no checkpoint/model selection.
        tr=predict(m,train,arm,p['normalization']);path=a.out/'predictions'/f'{arm}_train.npz'
        if replay and not path.is_file():raise ValueError('Replay refuses missing training predictions')
        checkpoint_npz(path,{'pred':tr,'keys':np.concatenate([s['keys'] for s in train])})
        training_metrics[arm]=metric(np.concatenate([s['y'] for s in train]),tr)
        pred=predict(m,ev,arm,p['normalization']);path=a.out/'predictions'/f'{arm}.npz'
        if replay and not path.is_file():raise ValueError('Replay refuses missing evaluation predictions')
        checkpoint_npz(path,{'pred':pred,'keys':ref['keys']});preds[arm]=pred
    y=ref['truth'];allpred={**preds,'preserved_tree':ref['pred']};metrics={arm:metric(y,pred) for arm,pred in allpred.items()}
    cmp={f'{b}_vs_{c}':contrast(y,preds[c],preds[b],ref['keys'][:,0]) for c,b in CONTRASTS}
    role=np.concatenate([s['role_query'] for s in ev]);h=ref['keys'][:,3]/10;slices=[]
    for arm,pred in allpred.items():
        for name,mask in [('first_second',h<=1),('after_first_second',h>1)]+[(f'role_{r}',role==r) for r in np.unique(role)]:
            if mask.any():slices.append({'arm':arm,'slice':name,**metric(y[mask],pred[mask])})
    result={'status':'family_study_complete','round':a.round_no,'protocol_signature':psig,'metrics':metrics,
       'training_metrics':training_metrics,'contrasts':cmp,'slices':slices,'evaluation_rows':len(y),
       'evaluation_games':len(np.unique(ref['keys'][:,0])),'training_rows':sum(len(s['y']) for s in train),
       'core_ready_for_replication':bool(cmp['core_vs_mask']['passes'] and metrics['core']['rmse']<=metrics['preserved_tree']['rmse']),
       'full_ready_for_replication':bool(cmp['full_vs_core']['passes'] and cmp['full_vs_mask']['passes'] and metrics['full']['rmse']<=metrics['preserved_tree']['rmse']),
       'models_total':3,'old_models_refitted':0,'new_kaggle_score':None,'feature_research':'open',
       'boundary':'Within-round numerical feature effects conditional on expanded player encoder; no claim masks or architecture outperform parent'}
    if replay and not (a.out/'summary.json').is_file():raise ValueError('Existing evaluation required for replay')
    seal_json(a.out/'summary.json',result)
    if replay:
        r={'status':'family_replay_exact','round':a.round_no,'new_optimizer_steps':0,'summary_sha256':digest(a.out/'summary.json'),'models_replayed':3}
        seal_json(a.out/'replay.json',r);return r
    return result
