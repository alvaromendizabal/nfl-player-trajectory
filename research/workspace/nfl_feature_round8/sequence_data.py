"""Verify existing parents, then export one fold as observed-only play tensors."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from parent_bridge import parent,read,indices,replay_parent
from parent_support import digest,hash_json,safe_file,atomic_json,atomic_npz,read_npz,seal_json
from origin_features import BASE_NAMES,REQUIRED,build_view,query_features
from sequence_features import build,NODE_NAMES,PAIR_NAMES


def code_signature(kit):
    return hash_json({p.name:digest(p) for p in sorted(Path(kit).glob('*.py'))}|{'contract':digest(Path(kit)/'input_contract.json')})


def context(args):
    ctx=parent(args);c=read(args.kit,'input_contract.json')
    for n,h in c['round7_report_hashes'].items():
        if digest(safe_file(args.round7,n))!=h:raise ValueError('Reviewed Round 7 receipt changed: '+n)
    r=read(args.round7,'summary.json');rr=read(args.round7,'replay.json')
    if r['status']!='history_study_complete' or r['role_history_earned_further_testing'] or r['player_history_earned_further_testing'] or rr['new_coordinate_models']!=0:
        raise ValueError('Expected complete negative historical-feature study')
    tr,va=indices(ctx,2);ctx.update(tr=tr,va=va)
    sig=hash_json({'code':code_signature(args.kit),'parent':ctx['r5_experiment'],'data':ctx['source'],'fold':2})
    ctx['signature']=sig;return ctx


def storage_parity(rebuilt,saved):
    if saved.dtype not in (np.dtype('float32'),np.dtype('float64')) or rebuilt.shape!=saved.shape:
        raise ValueError('Storage dtype/shape mismatch')
    if not np.isfinite(rebuilt).all() or not np.array_equal(rebuilt.astype(saved.dtype),saved):
        raise ValueError('Observed adapter differs AFTER original storage conversion')


def read_feature(out,name,sig):
    path=safe_file(out,'plays/'+name);r=read(path.parent,name.replace('.npz','.json'))
    if r['signature']!=sig or r['sha256']!=digest(path):raise ValueError('Sequence checkpoint identity changed')
    z=read_npz(path)
    if str(z['signature'])!=sig:raise ValueError('Sequence numerical source mismatch')
    if not all(np.isfinite(v).all() for k,v in z.items() if k!='signature'):raise ValueError('Nonfinite checkpoint')
    return z,r


def prepare(args,ctx,*,smoke=False,event=print):
    sig=ctx['signature'];d=ctx['data'];tr,va=ctx['tr'],ctx['va']
    tgames=set(map(int,d['keys'][tr,0]));egames=set(map(int,d['keys'][va,0]))
    chosen=[p for p in ctx['selection']['plays'] if p['game'] in (tgames if smoke else tgames|egames)]
    if smoke:chosen=chosen[:32]
    if not chosen or (smoke and len(chosen)!=32):raise ValueError('Expected selected training plays')
    entries={x['path']:x for x in ctx['contract']['raw_input_files']}
    new=0;reused=0;rebuilt=0;support=[];missing=[];keys_index={}
    for i,k in enumerate(d['keys'][:,:2]):keys_index.setdefault(tuple(map(int,k)),[]).append(i)
    for p in chosen:
        name=f"{p['game']}_{p['play']}.npz";target=args.out/'plays'/name
        if target.exists():read_feature(args.out,name,sig);reused+=1
        if smoke or not target.exists():missing.append(p)
    for name in sorted({p['input'] for p in missing}):
        e=entries[name];rawpath=safe_file(args.repo,name)
        if rawpath.stat().st_size!=e['size'] or digest(rawpath)!=e['sha256']:raise ValueError('Raw observed input drift')
        wanted={(p['game'],p['play']) for p in missing if p['input']==name}
        cols=[n for n in pd.read_csv(rawpath,nrows=0).columns if n in REQUIRED|{'s','dir','o'}]
        parts=[]
        for ch in pd.read_csv(rawpath,usecols=cols,chunksize=50000):
            mask=pd.MultiIndex.from_frame(ch[['game_id','play_id']]).isin(wanted)
            if mask.any():parts.append(ch.loc[mask])
        if not parts:raise ValueError('Selected observed play missing')
        seen=set()
        for (g,p),raw in pd.concat(parts,ignore_index=True).groupby(['game_id','play_id'],sort=True):
            g,p=int(g),int(p);seen.add((g,p));ix=np.array(keys_index[(g,p)],dtype=np.int64)
            raw=raw.sort_values(['nfl_id','frame_id']).reset_index(drop=True);origin=int(raw.frame_id.max())
            old=safe_file(ctx['parent'],f'features/plays/{g}_{p}.npz')
            receipt=read(old.parent,old.with_suffix('.json').name)
            if digest(old)!=receipt['sha256']:raise ValueError('Saved feature shard hash failed')
            oz=read_npz(old,('X','keys','y'));view=build_view(raw,origin=origin)
            rebuilt_x=query_features(view,oz['keys'][:,2],oz['keys'][:,3]/10.0)
            if isinstance(rebuilt_x,tuple):rebuilt_x=rebuilt_x[0]
            storage_parity(rebuilt_x,oz['X'])
            if not np.array_equal(oz['keys'],d['keys'][ix]) or not np.array_equal(oz['y'],d['y'][ix]):raise ValueError('Old targets/keys changed')
            f=build(raw,origin=origin);pid_index={int(v):i for i,v in enumerate(f['ids'])}
            qi=np.array([pid_index[int(v)] for v in d['keys'][ix,2]],np.int64)
            if len(np.unique(d['keys'][ix],axis=0))!=len(ix):raise ValueError('Duplicate requests')
            arrays={**f,'query':qi,'base':d['X'][ix,:len(BASE_NAMES)],'y':d['y'][ix],
                    'keys':d['keys'][ix],'role_query':d['role'][ix],
                    'train':np.array(g in tgames),'signature':np.array(sig)}
            target=args.out/'plays'/f'{g}_{p}.npz'
            if target.exists():
                oldz,_=read_feature(args.out,target.name,sig)
                if oldz.keys()!=arrays.keys() or any(not np.array_equal(oldz[k],v) for k,v in arrays.items()):raise ValueError('Reconstructed sequence mismatch')
            else:
                atomic_npz(target,**arrays);atomic_json(target.with_suffix('.json'),{'signature':sig,'sha256':digest(target),'parent_shard_sha256':digest(old)})
                new+=1
            rebuilt+=1
            if rebuilt%16==0:event('sequence_progress',rebuilt=rebuilt,new_checkpoints=new,reused_checkpoints=reused,total_plays=len(chosen))
        if seen!=wanted or digest(rawpath)!=e['sha256']:raise ValueError('Observed inventory changed during reading')
    records=[];totalbytes=0;counts={'training_rows':0,'evaluation_rows':0};paircounts=np.zeros(8);pairdenom=0;nodecounts=np.zeros(10);nodedenom=0
    for p in chosen:
        name=f"{p['game']}_{p['play']}.npz";z,r=read_feature(args.out,name,sig)
        train=bool(z['train']);counts['training_rows' if train else 'evaluation_rows']+=len(z['keys']);totalbytes+=sum(v.nbytes for v in z.values())
        records.append({'path':'plays/'+name,'sha256':r['sha256'],'train':train,'rows':len(z['keys']),'players':len(z['ids'])})
        if train:
            # Per-channel denominator is joint-observation pairs, not padded player slots.
            joint=z['pair_valid'][...,0];pairdenom+=joint.sum();paircounts+=z['pair_valid'].sum(axis=(0,1,2))
            present=z['node_valid'][...,0];nodedenom+=present.sum();nodecounts+=z['node_valid'].sum(axis=(0,1))
    if totalbytes>1024**3:raise ValueError('Prepared tensors exceed 1 GiB decoded limit')
    if not smoke:
        if counts!={'training_rows':len(tr),'evaluation_rows':len(va)}:raise ValueError('Frozen row population changed')
        pred,parents=replay_parent(ctx,2)
        ref=args.out/'reference.npz'
        arrays={'keys':d['keys'][va],'truth':d['y'][va],'pred':pred['control']}
        if ref.exists():
            old=read_npz(ref)
            if any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('Reference parity drift')
        else:atomic_npz(ref,**arrays)
        manifest={'signature':sig,'code':code_signature(args.kit),'fold':2,'files':records,'reference_sha256':digest(ref),'decoded_bytes':totalbytes,**counts}
        seal_json(args.out/'dataset.json',manifest)
    r={'status':'sequence_smoke_passed' if smoke else 'sequence_features_ready','signature':sig,'plays':len(chosen),**counts,
       'new_checkpoints':new,'reused_checkpoints':reused,'raw_adapter_rebuilt':rebuilt,'decoded_bytes':totalbytes,
       'node_channel_names':list(NODE_NAMES),'pair_channel_names':list(PAIR_NAMES),
       'training_pair_support':(paircounts/max(pairdenom,1)).tolist(),'training_node_support':(nodecounts/max(nodedenom,1)).tolist(),
       'scientific_fits':0,'raw_output_csv_reads':0,'feature_research':'open'}
    filename='smoke.json' if smoke else 'preparation.json'
    atomic_json(args.out/(filename.replace('.json','_reuse.json') if (args.out/filename).exists() else filename),r)
    event('sequence_stage_complete',**{k:r[k] for k in ['status','plays','new_checkpoints']});return r
