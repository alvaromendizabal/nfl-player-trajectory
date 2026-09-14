"""Finite one-fold two-arm study with optimizer/RNG/cursor preservation."""
from __future__ import annotations
import io,json,os,platform,time
from pathlib import Path
import numpy as np
import torch
from sequence_model import CONFIG,Forecast,setup,normalization,collate,ordered_batches,step,predict
from parent_support import digest,hash_json,safe_file,atomic_json,atomic_npz,read_npz,seal_json,_atomic
from sequence_data import code_signature
from origin_features import rmse


def event(name,**fields):
    from datetime import datetime,timezone
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':name,**fields},allow_nan=False),flush=True)


def environment():
    return {'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
            'threads':CONFIG['threads'],'device':'cpu','machine':platform.machine()}


def load_data(out,kit):
    manifest=json.loads(safe_file(out,'dataset.json').read_text())
    if manifest['code']!=code_signature(kit):raise ValueError('Source changed after preparation')
    train=[];evaluation=[];total=0
    for r in manifest['files']:
        p=safe_file(out,r['path'])
        if digest(p)!=r['sha256']:raise ValueError('Tensor hash changed')
        z=read_npz(p);total+=sum(x.nbytes for x in z.values())
        if total>1024**3:raise ValueError('Input memory budget exceeded')
        if str(z['signature'])!=manifest['signature'] or bool(z['train'])!=r['train']:raise ValueError('Tensor lineage drift')
        (train if r['train'] else evaluation).append(z)
    if not train or not evaluation:raise ValueError('Missing split')
    tg={int(p['keys'][0,0]) for p in train};eg={int(p['keys'][0,0]) for p in evaluation}
    if tg&eg or max(tg)//100>=min(eg)//100:raise ValueError('Nonchronological or overlapping split')
    for plays,label in [(train,'training_rows'),(evaluation,'evaluation_rows')]:
        kk=np.concatenate([p['keys'] for p in plays]);yy=np.concatenate([p['y'] for p in plays])
        if len(kk)!=manifest[label] or len(np.unique(kk,axis=0))!=len(kk) or not np.isfinite(yy).all():raise ValueError('Row coverage/keys changed')
    ref=safe_file(out,'reference.npz')
    if digest(ref)!=manifest['reference_sha256']:raise ValueError('Tree reference changed')
    rr=read_npz(ref);kk=np.concatenate([p['keys'] for p in evaluation]);yy=np.concatenate([p['y'] for p in evaluation])
    if not np.array_equal(rr['keys'],kk) or not np.array_equal(rr['truth'],yy):raise ValueError('Reference row/label mismatch')
    return train,evaluation,manifest,rr


def protocol(out,kit):
    setup();train,evaluation,m,ref=load_data(out,kit);norm=normalization(train)
    p={'study':'learned-pair-history-v1','data_sha256':digest(out/'dataset.json'),'signature':m['signature'],
       'config':CONFIG,'environment':environment(),'normalization':norm,'arms':['terminal','history'],
       'parameters':sum(x.numel() for x in Forecast().parameters()),'selection':'fixed_final_step_no_validation_selection'}
    seal_json(out/'neural_protocol.json',p)
    return train,evaluation,m,ref,p,hash_json(p)


def save_checkpoint(folder,model,opt,stepno,signature,training_loss,initial_hash):
    folder.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':stepno,
           'torch_rng':torch.get_rng_state(),'signature':signature,'losses':training_loss,'initial_hash':initial_hash}
    buf=io.BytesIO();torch.save(state,buf);payload=buf.getvalue()
    import hashlib
    h=hashlib.sha256(payload).hexdigest();name=f'{stepno:06d}-{h}.pt';path=folder/name
    if path.exists():
        if digest(path)!=h:raise ValueError('Existing checkpoint corrupted')
    else:_atomic(path,lambda f:f.write(payload))
    if digest(path)!=h:raise ValueError('Checkpoint readback hash failed')
    atomic_json(folder/'checkpoint.json',{'signature':signature,'step':stepno,'blob':name,'sha256':h,'bytes':len(payload),'initial_hash':initial_hash})
    return h


def load_checkpoint(folder,model,opt,signature):
    r=json.loads(safe_file(folder,'checkpoint.json').read_text());p=safe_file(folder,r['blob'])
    if r['signature']!=signature or digest(p)!=r['sha256']:raise ValueError('Checkpoint identity drift')
    state=torch.load(p,map_location='cpu',weights_only=True)
    if state['signature']!=signature or state['step']!=r['step']:raise ValueError('Checkpoint cursor mismatch')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
    return state


def weight_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


def train_arm(out,kit,arm,*,replay_only=False,max_steps=None):
    if arm not in ('terminal','history'):raise ValueError('Unknown arm')
    train,evaluation,m,ref,p,psig=protocol(out,kit)
    if not replay_only:
        gate=json.loads(safe_file(out,'profile.json').read_text())
        if gate['protocol_signature']!=psig or gate['status']!='training_profile_passed':raise ValueError('Complete training-only profile first')
    setup();model=Forecast();initial=weight_hash(model);opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)
    sig=hash_json({'protocol':psig,'arm':arm});folder=out/'models'/arm
    cursor=0;losses=[];nb=len(ordered_batches(len(train),0));total=nb*CONFIG['epochs'];norm=p['normalization']
    if (folder/'checkpoint.json').exists():
        state=load_checkpoint(folder,model,opt,sig);cursor=state['step'];losses=state['losses']
        if state['initial_hash']!=initial or not 0<=cursor<=total:raise ValueError('Initialization/cursor changed')
    elif replay_only:raise ValueError('Replay cannot fit a missing model')
    if replay_only and cursor!=total:raise ValueError('Replay cannot train an incomplete model')
    start=cursor;began=time.monotonic();denom=2*sum(len(x['y']) for x in train)/nb
    limit=total if max_steps is None else min(total,cursor+max_steps)
    for i in range(cursor,limit):
        batch_ix=ordered_batches(len(train),i//nb)[i%nb];items=[train[j] for j in batch_ix]
        target=torch.tensor(np.concatenate([x['y'] for x in items]),dtype=torch.float32)
        loss=step(model,opt,collate(items,arm,norm),target,denom);losses.append(loss)
        if (i+1)%CONFIG['checkpoint_steps']==0 or (i+1)%nb==0 or i+1==limit:
            save_checkpoint(folder,model,opt,i+1,sig,losses,initial)
            event('optimizer_checkpoint',arm=arm,steps=i+1,total_steps=total,epoch=1+i//nb,elapsed_seconds=round(time.monotonic()-began,2))
    done=limit if limit>cursor else cursor
    if done<total:return {'status':'incomplete_checkpoint_preserved','step':done,'total':total}
    # Individual arms never score validation; primary evaluation waits for both.
    pred=predict(model,evaluation,arm,norm);keys=np.concatenate([x['keys'] for x in evaluation])
    path=out/'predictions'/f'{arm}.npz';rp=path.with_suffix('.json')
    if path.exists() or rp.exists():
        r=json.loads(safe_file(rp.parent,rp.name).read_text());old=read_npz(safe_file(path.parent,path.name))
        if r['signature']!=sig or digest(path)!=r['sha256'] or not np.array_equal(old['pred'],pred) or not np.array_equal(old['keys'],keys):raise ValueError('Prediction replay is not exact')
    elif replay_only:raise ValueError('Replay cannot create missing prediction evidence')
    else:
        atomic_npz(path,pred=pred,keys=keys);atomic_json(rp,{'signature':sig,'sha256':digest(path)})
    # Reload final weights in-process; command replay later is a fresh process.
    other=Forecast();oo=torch.optim.AdamW(other.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)
    load_checkpoint(folder,other,oo,sig)
    if not np.array_equal(predict(other,evaluation,arm,norm),pred):raise ValueError('Final checkpoint forward replay failed')
    epoch_losses=[sum(losses[i:i+nb])/nb for i in range(0,total,nb)]
    result={'status':'encoder_prediction_replay_exact' if replay_only else 'encoder_arm_complete',
            'arm':arm,'protocol_signature':psig,'steps':total,'new_optimizer_steps':total-start,'initial_hash':initial,
            'parameter_count':p['parameters'],'training_epoch_objectives':epoch_losses,'evaluation_rows':len(pred),
            'validation_scored_here':False,'new_scientific_models':int(start<total),'elapsed_seconds':round(time.monotonic()-began,3)}
    name='replay.json' if replay_only else 'complete.json'
    if (folder/name).exists() and not replay_only:name='reuse.json'
    atomic_json(folder/name,result);return result


def compare(y,a,b,games):
    u,inv=np.unique(games,return_inverse=True);n=np.bincount(inv)
    sa=np.bincount(inv,weights=((y-a)**2).sum(1));sb=np.bincount(inv,weights=((y-b)**2).sum(1))
    if len(u)<2 or sa.sum()<=0:raise ValueError('Insufficient paired games')
    rng=np.random.default_rng(20260912);ix=rng.integers(0,len(u),(10000,len(u)))
    delta=np.sqrt(sb[ix].sum(1)/(2*n[ix].sum(1)))-np.sqrt(sa[ix].sum(1)/(2*n[ix].sum(1)))
    lo,hi=np.quantile(delta,[.025,.975]);gain=1-rmse(y,b)/rmse(y,a)
    return {'delta_rmse':rmse(y,b)-rmse(y,a),'relative_gain':gain,'low95':float(lo),'high95':float(hi),
            'resamples':10000,'feature_screen_gate':bool(gain>=.01 and hi<0),
            'scope':'One exploratory prespecified contrast on reused games; not sequential-family error control'}


def evaluate(out,kit):
    train,ev,m,ref,p,psig=protocol(out,kit);pred={};arms=[]
    for arm in ('terminal','history'):
        r=json.loads(safe_file(out,f'models/{arm}/complete.json').read_text())
        ck=json.loads(safe_file(out,f'models/{arm}/checkpoint.json').read_text())
        blob=safe_file(out/f'models/{arm}',ck['blob'])
        if digest(blob)!=ck['sha256']:raise ValueError('Final model checksum drift')
        path=safe_file(out,f'predictions/{arm}.npz');pr=json.loads(safe_file(path.parent,arm+'.json').read_text())
        if r['protocol_signature']!=psig or ck['step']!=r['steps'] or pr['signature']!=hash_json({'protocol':psig,'arm':arm}) or digest(path)!=pr['sha256']:raise ValueError('Incomplete or changed treatment')
        z=read_npz(path)
        if not np.array_equal(z['keys'],ref['keys']):raise ValueError('Treatment forecast rows differ')
        pred[arm]=z['pred'];arms.append(r)
    if arms[0]['initial_hash']!=arms[1]['initial_hash'] or arms[0]['steps']!=arms[1]['steps']:raise ValueError('Unmatched initialization/exposure')
    y=ref['truth'];metrics={a:rmse(y,v) for a,v in pred.items()};metrics['preserved_tree']=rmse(y,ref['pred']);metrics['constant_velocity']=rmse(y,np.zeros_like(y))
    contrast=compare(y,pred['terminal'],pred['history'],ref['keys'][:,0]);slices=[]
    role=np.concatenate([x['role_query'] for x in ev]);h=ref['keys'][:,3]/10
    for arm,yp in {**pred,'preserved_tree':ref['pred']}.items():
        for name,mask in [('first_second',h<=1),('after_first_second',h>1)]+[(f'role_{int(r)}',role==r) for r in np.unique(role)]:
            if mask.any():slices.append({'arm':arm,'slice':name,'rows':int(mask.sum()),'sse':float(((y[mask]-yp[mask])**2).sum()),'rmse':rmse(y[mask],yp[mask])})
    result={'status':'matched_encoder_study_complete','protocol_signature':psig,'metrics':metrics,'contrast':contrast,
            'evaluation_rows':len(y),'evaluation_games':len(np.unique(ref['keys'][:,0])),
            'training_rows':m['training_rows'],'new_scientific_models':2,'old_models_refitted':0,'slices':slices,
            'conditional_feature_evidence':contrast['feature_screen_gate'],
            'ready_for_later_fold':bool(contrast['feature_screen_gate'] and metrics['history']<=metrics['preserved_tree']),
            'comparison_boundary':'History versus terminal isolates pair-sequence information; neural versus tree changes model and shared representation.',
            'feature_research':'open','new_kaggle_score':None,'github_updated':False}
    if (out/'summary.json').exists():
        if json.loads((out/'summary.json').read_text())!=result:raise ValueError('Completed evaluation drift')
    else:atomic_json(out/'summary.json',result)
    return result
