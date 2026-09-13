"""Two matched goal-feature arms; no tuning, ensembling or automatic extra folds."""
from __future__ import annotations
import hashlib
import json
import platform
import time
import numpy as np
import torch
from audit_io import digest,hash_json,safe_file,read_json,atomic_json,seal_json,checkpoint_npz
from data_io import load_training,evaluation_data
from checkpoints import save_checkpoint,load_checkpoint,weight_hash
from grouped_model import CONFIG,Forecast,setup,normalization,collate,ordered_batches,step,predict
from grouped_features import ARMS


def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'device':'cpu','threads':CONFIG['threads'],'machine':platform.machine()}


def protocol(a):
    setup();train,m=load_training(a)
    norm=normalization(train)
    games={int(p['keys'][0,0]) for p in train}
    evalgames={int(r['parent_path'].split('/')[-1].split('_')[0]) for r in m['files'] if not r['train']}
    if games&evalgames or max(games)//100>=min(evalgames)//100:
        raise ValueError('Evaluation must be on strictly later, disjoint game dates')
    p={'study':'grouped-goal-frame-v1','config':CONFIG,'data_sha256':digest(a.out/'dataset.json'),
       'environment':environment(),'normalization':norm,'arms':list(ARMS),
       'parameters':sum(x.numel() for x in Forecast().parameters()),
       'mask_policy':'same values/masks/ages except six numerical goal channels zero in cartesian',
       'selection':'fixed final step; no evaluation checkpoint selection',
       'optimizer_steps_per_arm':len(ordered_batches(len(train),0))*CONFIG['epochs']}
    seal_json(a.out/'protocol.json',p)
    return train,p,hash_json(p)


def new_optimizer(model):
    return torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)


def profile(a):
    train,p,psig=protocol(a)
    items=sorted(train,key=lambda x:(len(x['ids']),len(x['base'])),reverse=True)[:8]
    target=torch.tensor(np.concatenate([x['y'] for x in items]),dtype=torch.float32)
    timing={}; sensitivity={}; done=0
    for arm in ARMS:
        setup();model=Forecast();opt=new_optimizer(model);durations=[]
        for i in range(8):
            before=time.monotonic()
            b=collate(items,arm,p['normalization']); step(model,opt,b,target,2*len(target))
            done+=1
            if i>0:durations.append(time.monotonic()-before)
        seconds=float(np.quantile(durations,.9))
        timing[arm]={'p90_step_seconds':seconds,'conservative_seconds':seconds*p['optimizer_steps_per_arm']*1.5+60}
        with torch.no_grad():
            normal=model(**b)
            diff=model(**b,zero_context=True)-normal
            sensitivity[arm]={'training_context_removal_rms':float(diff.square().mean().sqrt())}
        if arm=='goal':
            model.zero_grad(set_to_none=True)
            q=collate(items,arm,p['normalization']);q['pair'].requires_grad_()
            # Deterministic signed output projection; not a target-dependent gate.
            pred=model(**q);(pred[:,0].sum()-pred[:,1].sum()).backward()
            grad=q['pair'].grad[...,8:]
            valid=q['pair_valid'][...,8:]
            if not valid.any() or not torch.isfinite(grad).all() or not (grad[valid]!=0).any():
                raise ValueError('New goal-feature pathway has no finite nonzero local gradient')
            sensitivity[arm]['goal_input_gradient_rms']=float(grad[valid].square().mean().sqrt())
    passed=all(x['conservative_seconds']<=300 for x in timing.values())
    r={'status':'grouped_profile_passed' if passed else 'grouped_profile_budget_stop',
       'protocol_signature':psig,'timings':timing,'pathway_check':sensitivity,
       'profile_training_plays':len(items),'disposable_optimizer_steps':done,
       'evaluation_arrays_opened':False,'budget_guard_seconds':300,'scientific_fits':0,
       'warning':'Nonzero sensitivity is an engineering check, not feature value; no threshold tuned on data.'}
    atomic_json(a.out/'profile.json',r)
    return r


def train_arm(a,arm,max_steps=None):
    if arm not in ARMS:raise ValueError('Unknown arm')
    train,p,psig=protocol(a)
    g=read_json(safe_file(a.out,'profile.json'))
    if g['status']!='grouped_profile_passed' or g['protocol_signature']!=psig:
        raise ValueError('Current training-only profile must pass; no budget bypass')
    setup();model=Forecast();initial=weight_hash(model);opt=new_optimizer(model)
    sig=hash_json({'protocol':psig,'arm':arm});folder=a.out/'models'/arm
    nb=len(ordered_batches(len(train),0));total=nb*CONFIG['epochs'];cursor=0;losses=[]
    if (folder/'checkpoint.json').exists():
        state=load_checkpoint(folder,model,opt,sig);cursor=state['step'];losses=state['losses']
        if state['initial_hash']!=initial or not 0<=cursor<=total or len(losses)!=cursor:
            raise ValueError('Initialization/cursor mismatch')
    limit=total if max_steps is None else min(total,cursor+max_steps)
    began=time.monotonic();denominator=2*sum(len(t['y']) for t in train)/nb
    for i in range(cursor,limit):
        selected=ordered_batches(len(train),i//nb)[i%nb]
        items=[train[j] for j in selected]
        y=torch.tensor(np.concatenate([s['y'] for s in items]),dtype=torch.float32)
        losses.append(step(model,opt,collate(items,arm,p['normalization']),y,denominator))
        if (i+1)%CONFIG['checkpoint_steps']==0 or (i+1)%nb==0 or i+1==limit:
            save_checkpoint(folder,model,opt,i+1,sig,losses,initial)
            print(json.dumps({'event':'optimizer_checkpoint','arm':arm,'steps':i+1,'total_steps':total,
                              'elapsed_seconds':round(time.monotonic()-began,2)}),flush=True)
    done=max(cursor,limit)
    if done<total:return {'status':'incomplete_checkpoint_preserved','steps':done,'new_optimizer_steps':done-cursor}
    # Training-only checkpoint replay; evaluation arrays remain unopened here.
    probe=predict(model,train[:8],arm,p['normalization'])
    other=Forecast();load_checkpoint(folder,other,new_optimizer(other),sig)
    if not np.array_equal(probe,predict(other,train[:8],arm,p['normalization'])):
        raise ValueError('Independent training-probe checkpoint replay failed')
    r={'status':'grouped_arm_complete','arm':arm,'protocol_signature':psig,'initial_hash':initial,
       'steps':total,'new_optimizer_steps':total-cursor,'parameters':p['parameters'],
       'training_epoch_objectives':[sum(losses[i:i+nb])/nb for i in range(0,total,nb)],
       'probe_sha256':hashlib.sha256(probe.tobytes()).hexdigest(),
       'evaluation_arrays_opened':False,'fixed_final_step':True}
    if (folder/'complete.json').exists():
        old=read_json(folder/'complete.json')
        for key in ('protocol_signature','initial_hash','steps','probe_sha256','training_epoch_objectives'):
            if r[key]!=old[key]:raise ValueError('Completed arm changed')
        atomic_json(folder/'reuse.json',r)
    else:atomic_json(folder/'complete.json',r)
    return r


def rmse(y,p):
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Finite aligned nonempty coordinate rows required')
    return float(np.sqrt(np.mean((y-p)**2)))


def contrast(y,control,treatment,games):
    rmse(y,control);rmse(y,treatment)
    if len(games)!=len(y):raise ValueError('Game-key mismatch')
    u,inv=np.unique(games,return_inverse=True)
    if len(u)<2:raise ValueError('At least two evaluation games required')
    n=np.bincount(inv);sa=np.bincount(inv,weights=((y-control)**2).sum(1));sb=np.bincount(inv,weights=((y-treatment)**2).sum(1))
    if sa.sum()<=0:raise ValueError('Zero control error makes relative gate undefined')
    rng=np.random.default_rng(20260912);ix=rng.integers(0,len(u),(10000,len(u)))
    delta=np.sqrt(sb[ix].sum(1)/(2*n[ix].sum(1)))-np.sqrt(sa[ix].sum(1)/(2*n[ix].sum(1)))
    lo,hi=np.quantile(delta,[.025,.975]);gain=1-rmse(y,treatment)/rmse(y,control)
    return {'relative_gain':gain,'delta_rmse':rmse(y,treatment)-rmse(y,control),
            'low95':float(lo),'high95':float(hi),'resamples':10000,
            'feature_gate_passed':bool(gain>=.01 and hi<0),
            'scope':'One planned comparison on reused games; not experiment-series-wide error control.'}


def evaluate(a,replay=False):
    train,p,psig=protocol(a)
    # Refuse evaluation until both arms are complete and matched.
    records={arm:read_json(safe_file(a.out,f'models/{arm}/complete.json')) for arm in ARMS}
    if len({r['initial_hash'] for r in records.values()})!=1:
        raise ValueError('Arms do not share initialization')
    for r in records.values():
        if r['steps']!=p['optimizer_steps_per_arm'] or r['protocol_signature']!=psig:
            raise ValueError('Incomplete or unmatched exposure')
    ev,ref=evaluation_data(a);predictions={}
    for arm in ARMS:
        setup();model=Forecast();state=load_checkpoint(a.out/'models'/arm,model,new_optimizer(model),hash_json({'protocol':psig,'arm':arm}))
        if state['step']!=p['optimizer_steps_per_arm']:raise ValueError('Incomplete checkpoint')
        probe=predict(model,train[:8],arm,p['normalization'])
        if hashlib.sha256(probe.tobytes()).hexdigest()!=records[arm]['probe_sha256']:
            raise ValueError('Training checkpoint replay drift')
        pred=predict(model,ev,arm,p['normalization'])
        path=a.out/'predictions'/f'{arm}.npz'
        if replay and not path.is_file():raise ValueError('Replay cannot create missing predictions')
        checkpoint_npz(path,{'pred':pred,'keys':ref['keys']})
        predictions[arm]=pred
    y=ref['truth'];metrics={arm:rmse(y,pred) for arm,pred in predictions.items()}
    metrics['preserved_tree']=rmse(y,ref['pred'])
    cmp=contrast(y,predictions['cartesian'],predictions['goal'],ref['keys'][:,0])
    slices=[];roles=np.concatenate([p['role_query'] for p in ev]);h=ref['keys'][:,3]/10
    for name,pred in {**predictions,'preserved_tree':ref['pred']}.items():
        for label,mask in [('first_second',h<=1),('after_first_second',h>1)]+[(f'role_{int(r)}',roles==r) for r in np.unique(roles)]:
            if mask.any():slices.append({'arm':name,'slice':label,'rows':int(mask.sum()),
                                        'sse':float(((y[mask]-pred[mask])**2).sum()),'rmse':rmse(y[mask],pred[mask])})
    result={'status':'grouped_goal_study_complete','protocol_signature':psig,'metrics':metrics,'contrast':cmp,
            'training_rows':sum(len(p['y']) for p in train),'evaluation_rows':len(y),
            'evaluation_games':len(np.unique(ref['keys'][:,0])),'slices':slices,
            'ready_for_later_fold':bool(cmp['feature_gate_passed'] and metrics['goal']<=metrics['preserved_tree']),
            'scientific_models_total':2,'old_models_refitted':0,'new_kaggle_score':None,'feature_research':'open',
            'comparison_boundary':'Goal vs Cartesian attributes six goal-frame values conditional on the shared new encoder. Comparisons to Round 8 or the tree change architecture too.'}
    if replay and not (a.out/'summary.json').is_file():raise ValueError('Replay requires existing evaluation summary')
    seal_json(a.out/'summary.json',result)
    if replay:
        r={'status':'grouped_goal_replay_exact','new_optimizer_steps':0,'models_replayed':2,
           'summary_sha256':digest(a.out/'summary.json')}
        atomic_json(a.out/'replay.json',r)
        return r
    return result
