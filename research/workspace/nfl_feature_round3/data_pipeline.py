"""Extend the preserved training-side study, never the reserved holdout.

Raw labels are used only to create targets for newly selected plays. Selection
uses observed-input keys only. Old feature/target rows are reused byte-for-byte.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from origin_features import ALL_NAMES, BASE_NAMES, KEYS, REQUIRED, make_example, build_view, query_features
from pair_features import NAMES, pair_views
from parent_support import atomic_json, atomic_npz, digest, read_npz, safe_file, seal_json, hash_json

TOTAL_PLAYS=1024
SEED=20260911


def choose_plays(candidates, old, total=TOTAL_PLAYS):
    old=set(tuple(map(int,p)) for p in old)
    candidates=set(tuple(map(int,p)) for p in candidates)
    if not old.issubset(candidates): raise ValueError('An original play is missing from observed inputs')
    if total<len(old) or len(candidates)<total: raise ValueError('Insufficient eligible plays for frozen sample size')
    groups={}
    for g,p in candidates-old:
        score=hashlib.sha256(f'{SEED}:{g}:{p}'.encode()).hexdigest()
        groups.setdefault(g,[]).append((score,p))
    for values in groups.values():values.sort(reverse=True)
    selected=set(old)
    # Round-robin adds games broadly; no role, outcome, error, or horizon selects a play.
    while len(selected)<total:
        advanced=False
        for g in sorted(groups):
            if groups[g]:
                _,p=groups[g].pop();selected.add((g,p));advanced=True
                if len(selected)==total:break
        if not advanced:raise ValueError('Selection exhausted')
    return sorted(selected)


def verify_raw(repo, entry):
    p=safe_file(repo,entry['path'])
    if p.stat().st_size!=entry['size'] or digest(p)!=entry['sha256']:
        raise ValueError('Raw input checksum mismatch: '+entry['path'])
    return p


def make_plan(repo:Path, out:Path, parent:dict, contract:dict, sig:str, event=print, total=TOTAL_PLAYS):
    path=out/'selection.json'
    if path.exists():
        receipt=path.with_name('selection.receipt.json')
        if not receipt.is_file() or json.loads(receipt.read_text()).get('sha256')!=digest(path):
            raise ValueError('Frozen selection receipt is missing or changed; preserve the file')
        d=json.loads(path.read_text())
        if d['signature']!=sig or d['target_plays']!=total:raise ValueError('Selection source drift')
        return d
    data=parent['data'];old=np.unique(data['keys'][:,:2],axis=0)
    allowed=set(map(int,np.unique(data['keys'][:,0]))); candidates=set(); source={}
    entries=[e for e in contract['raw_files'] if '/train/input_' in e['path']]
    for k,e in enumerate(entries,1):
        file=verify_raw(repo,e)
        pairs=set()
        for part in pd.read_csv(file,usecols=['game_id','play_id'],chunksize=100000):
            sub=part.loc[part.game_id.isin(allowed)].drop_duplicates()
            pairs.update(map(tuple,sub.to_numpy(np.int64)))
        for pair in pairs:
            if pair in source:raise ValueError('A play occurs in multiple weekly input files')
            source[pair]=e['path']
        candidates.update(pairs)
        event('input_inventory',completed_files=k,total_files=len(entries),eligible_plays=len(candidates))
    chosen=choose_plays(candidates,old,total)
    d={'signature':sig,'target_plays':total,'original_plays':len(old),'eligible_plays':len(candidates),
       'allowed_games':sorted(allowed),'plays':[{'game':g,'play':p,'input':source[(g,p)],'old':(g,p) in set(map(tuple,old))} for g,p in chosen],
       'folds':[{'fold':f['fold'],'train_games':f['train_games'].tolist(),'validation_games':f['validation_games'].tolist()} for f in parent['folds']],
       'selection_uses_outcomes':False,'formal_outer_validation_used':False,'reused_game_folds':True}
    seal_json(path,d)
    atomic_json(path.with_name('selection.receipt.json'),{'signature':sig,'sha256':digest(path)})
    return d


def play_path(out,g,p):return out/'features'/'plays'/f'{g}_{p}.npz'


def verify_shard(path,sig):
    receipt=path.with_suffix('.json')
    if not path.is_file() or path.is_symlink() or not receipt.is_file() or receipt.is_symlink():
        raise ValueError('Incomplete feature checkpoint; preserve it for diagnosis')
    r=json.loads(receipt.read_text())
    if r['signature']!=sig or digest(path)!=r['sha256']:raise ValueError('Feature checkpoint checksum/source mismatch')
    z=read_npz(path)
    n=len(z['keys'])
    if str(z['signature'])!=sig or z['X'].shape!=(n,len(ALL_NAMES)) or z['y'].shape!=(n,2):raise ValueError('Feature checkpoint shape/source mismatch')
    if any(not np.isfinite(z[k]).all() for k in ('X','y','terminal','history','support')):raise ValueError('Nonfinite checkpoint')
    if z['terminal'].shape!=(n,len(NAMES)) or z['history'].shape!=z['terminal'].shape:raise ValueError('Pair width mismatch')
    return z



def verify_parent_storage_parity(rebuilt, saved):
    """Replay the original float32 storage operation, then require exact values.

    Round 1 run_round.py persisted X as float32. Comparing unrounded float64
    recomputation at 1e-12 falsely rejects correctly preserved rows. This guard
    permits only the original storage conversion: no tolerance, row filtering,
    feature replacement or retraining. Any representable float32 mismatch stops.
    """
    rebuilt, saved = np.asarray(rebuilt), np.asarray(saved)
    if (rebuilt.ndim != 2 or rebuilt.shape != saved.shape
            or rebuilt.shape[1] != len(ALL_NAMES) or len(rebuilt) == 0):
        raise ValueError('Parent storage parity: invalid or misaligned feature shapes')
    if rebuilt.dtype != np.dtype('float64') or saved.dtype != np.dtype('float32'):
        raise ValueError('Parent storage parity: expected rebuilt float64 and saved float32')
    if not np.isfinite(rebuilt).all() or not np.isfinite(saved).all():
        raise ValueError('Parent storage parity: nonfinite features')
    with np.errstate(over='raise', invalid='raise'):
        stored = rebuilt.astype(np.float32)
    if not np.array_equal(stored, saved):
        changed = np.count_nonzero(stored != saved)
        raise ValueError(
            f'Parent storage parity: {changed} feature values differ AFTER float32 '
            'storage conversion; stop and diagnose, do not relax the comparison')
    return {'rows': len(saved), 'columns': saved.shape[1],
            'stored_dtype': 'float32', 'exact_after_storage_conversion': True,
            'max_abs_before_conversion': float(np.max(np.abs(rebuilt - saved.astype(np.float64))))}


def prepare(repo,out,parent,contract,sig,*,smoke=False,event=print,total=TOTAL_PLAYS):
    start=time.monotonic(); plan=make_plan(repo,out,parent,contract,sig,event,total)
    first_train=set(plan['folds'][0]['train_games'])
    chosen=[p for p in plan['plays'] if p['game'] in first_train][:32] if smoke else plan['plays']
    if not chosen:raise ValueError('No training plays for smoke')
    done,new,old_reused=0,0,0; missing={};inputs={e['path']:e for e in contract['raw_files']}
    for p in chosen:
        f=play_path(out,p['game'],p['play'])
        if f.exists() or f.with_suffix('.json').exists():verify_shard(f,sig);done+=1
        else:missing[(p['game'],p['play'])]=p
    parent_data=parent['data']; old_indices={}
    for g,p in np.unique(parent_data['keys'][:,:2],axis=0):
        old_indices[(int(g),int(p))]=np.flatnonzero((parent_data['keys'][:,0]==g)&(parent_data['keys'][:,1]==p))
    for name in sorted({p['input'] for p in missing.values()}):
        file=verify_raw(repo,inputs[name]);wanted={key for key,p in missing.items() if p['input']==name}
        cols=list(pd.read_csv(file,nrows=0).columns); use=[c for c in cols if c in REQUIRED|{'s','dir','o','a'}]
        parts=[]
        for chunk in pd.read_csv(file,usecols=use,chunksize=50000):
            mask=pd.MultiIndex.from_frame(chunk[['game_id','play_id']]).isin(wanted)
            if mask.any():parts.append(chunk.loc[mask])
        if not parts:raise ValueError('Selected input rows absent')
        raw=pd.concat(parts,ignore_index=True); future={}
        fresh={key for key in wanted if key not in old_indices}
        if fresh:
            oname=name.replace('/input_','/output_');op=verify_raw(repo,inputs[oname])
            # Output files are modest; filter immediately to declared original TRAIN games/plays.
            yy=pd.read_csv(op,usecols=KEYS+['x','y'])
            yy=yy.loc[pd.MultiIndex.from_frame(yy[['game_id','play_id']]).isin(fresh)]
            future={tuple(map(int,key)):g for key,g in yy.groupby(['game_id','play_id'],sort=True)}
        for key,r in raw.groupby(['game_id','play_id'],sort=True):
            key=tuple(map(int,key));r=r.sort_values(['nfl_id','frame_id']).reset_index(drop=True)
            if key in old_indices:
                ix=old_indices[key];keys=parent_data['keys'][ix]
                z={k:parent_data[k][ix] for k in ('X','y','keys','role')}
                view=build_view(r,origin=int(r.frame_id.max()))
                replay,_=query_features(view,keys[:,2],keys[:,3]/10)
                verify_parent_storage_parity(replay,z['X'])
                old_reused+=1
            else:
                if key not in future:raise ValueError('Missing future label rows for new play')
                z=make_example(r,future[key],0);keys=z['keys']
            pair=pair_views(r,keys,cutoff=int(r.frame_id.max()))
            f=play_path(out,*key)
            atomic_npz(f,**{k:z[k] for k in ('X','y','keys','role')},**pair,signature=np.array(sig))
            atomic_json(f.with_suffix('.json'),{'signature':sig,'sha256':digest(f),'rows':len(keys),'reused_parent_rows':key in old_indices})
            verify_shard(f,sig);done+=1;new+=1
            if done%16==0 or done==len(chosen):event('feature_progress',completed_plays=done,total_plays=len(chosen))
    if done!=len(chosen):raise ValueError('Some selected plays did not produce checkpoints')
    support=[]
    for p in chosen:
        if p['game'] not in first_train:continue
        z=verify_shard(play_path(out,p['game'],p['play']),sig)
        _,index=np.unique(z['keys'][:,2],return_index=True);support.append(z['support'][index])
    s=np.concatenate(support)
    result={'status':'pair_smoke_complete' if smoke else 'expanded_features_ready','signature':sig,'plays':done,
            'new_checkpoints':new,'reused_checkpoints':done-new,'old_feature_target_plays_reused_this_invocation':old_reused,
            'pair_columns_per_view':len(NAMES),'support_scope':'first fold training players only',
            'mean_pair_history_support':s.mean(0).tolist(),'supported_players':len(s),
            'outer_validation_used':False,'old_fits_repeated':0,'new_model_fits':0,'raw_data_modified':False,
            'elapsed_seconds':round(time.monotonic()-start,3),'feature_research':'open'}
    atomic_json(out/('smoke.json' if smoke else 'preparation.json'),result)
    return result


def assemble(out,sig):
    path=out/'selection.json'
    if json.loads((out/'selection.receipt.json').read_text())['sha256']!=digest(path):raise ValueError('Selection integrity check failed')
    plan=json.loads(path.read_text())
    if plan['signature']!=sig:raise ValueError('Selection signature changed')
    fields=('X','y','keys','role','terminal','history');parts={k:[] for k in fields};total=0; hashes={}
    for p in plan['plays']:
        f=play_path(out,p['game'],p['play']);z=verify_shard(f,sig)
        for k in fields:parts[k].append(z[k]);total+=z[k].nbytes
        hashes[f.name]=digest(f)
        if total>768*1024**2:raise ValueError('Feature memory budget exceeded')
    data={k:np.concatenate(v) for k,v in parts.items()}
    if len(data['keys'])>100000 or len(np.unique(data['keys'],axis=0))!=len(data['keys']):raise ValueError('Row cap or duplicate keys')
    if not np.isin(data['keys'][:,0],plan['allowed_games']).all():raise ValueError('Unexpected game')
    return data,plan,hash_json(hashes)
