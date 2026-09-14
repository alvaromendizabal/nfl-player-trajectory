"""Round 4 private-data preparation and exact, read-only parent replay."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from origin_features import ALL_NAMES, BASE_NAMES, REQUIRED, KEYS
from state_features import DIRECT_NAMES, GOAL_NAMES, build_features, storage_parity
from parent_support import (digest, hash_json, safe_file, read_npz, atomic_json,
                            atomic_npz, seal_json)

CODE_FILES = ('run_round.py','study.py','feature_worker.py','state_features.py',
              'origin_features.py','parent_support.py','pair_features.py','tree_core.py',
              'input_contract.json')


def json_file(root: Path, name: str) -> dict:
    return json.loads(safe_file(root,name).read_text())


def validate_arrays(data: dict, maximum: int) -> None:
    keys, x, y, role = (data[k] for k in ('keys','X','y','role'))
    n = len(keys)
    if not 1 <= n <= maximum or keys.shape != (n,4) or not np.issubdtype(keys.dtype,np.integer):
        raise ValueError('Parent forecast key shape/count/dtype changed')
    if len(np.unique(keys,axis=0)) != n or (keys[:,3] < 1).any():
        raise ValueError('Duplicate or invalid parent query keys')
    if x.shape != (n,len(ALL_NAMES)) or y.shape != (n,2) or role.shape != (n,):
        raise ValueError('Parent feature, label or role shape changed')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(role,[0,1,2,3]).all():
        raise ValueError('Nonfinite features/labels or invalid parent role')


def parent_context(parent: Path, legacy: Path, repo: Path, c: dict, *, verify_identity=True) -> dict:
    """Verify known receipts and source before loading only needed numerical arrays.

    Targets are reused from the completed parent cache. Raw output CSVs and
    future/held-out season data are never opened here.
    """
    for root in (parent,legacy,repo):
        if any(p.is_symlink() for p in (root,*root.parents)):
            raise ValueError('Symlink workspace roots are not accepted')
    if verify_identity:
        env = {'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}
        if env != c['parent_environment']:
            raise ValueError('Preserved numerical environment changed; return the report, do not reinstall: '+str(env))
        head = subprocess.run(['git','rev-parse','HEAD'],cwd=repo,check=True,capture_output=True,text=True,timeout=10).stdout.strip()
        if head != c['repo_commit']:
            raise ValueError('Git HEAD differs from the reviewed source; no reset is allowed')
    for n,sha in c['parent_source_hashes'].items():
        if digest(safe_file(legacy,n)) != sha:
            raise ValueError('Reviewed Round 3 source changed: '+n)
    for n,field in [('fold_1/summary.json','reviewed_summary_sha256'),
                    ('fold_1/replay.json','reviewed_replay_sha256'),
                    ('preparation.json','reviewed_preparation_sha256')]:
        if digest(safe_file(parent,n)) != c[field]:
            raise ValueError('Parent report changed since review: '+n)
    summary = json_file(parent,'fold_1/summary.json')
    replay = json_file(parent,'fold_1/replay.json')
    if summary['source_signature'] != c['parent_source_signature'] or summary['status'] != 'first_fold_complete':
        raise ValueError('Parent scientific run is incomplete or has changed lineage')
    for key in ('pooled_rmse','contrasts','training_rows','evaluation_rows','evaluation_games','source_signature'):
        if summary[key] != replay[key]:
            raise ValueError('Parent fit and replay disagree')
    if not replay['all_model_replays_exact'] or replay['new_coordinate_models'] != 0:
        raise ValueError('Parent replay did not finish without fitting')
    plan = json_file(parent,'plan.json')
    if plan['signature'] != c['parent_source_signature'] or plan['source_hashes'] != c['parent_source_hashes']:
        raise ValueError('Parent frozen source plan changed')
    selection = json_file(parent,'selection.json')
    if digest(safe_file(parent,'selection.json')) != json_file(parent,'selection.receipt.json')['sha256']:
        raise ValueError('Parent selection checksum failed')
    if selection['signature'] != plan['signature'] or len(selection['plays']) > c['max_selected_plays']:
        raise ValueError('Parent selection lineage/count changed')
    playkeys = [(p['game'],p['play']) for p in selection['plays']]
    if len(set(playkeys)) != len(playkeys):
        raise ValueError('Duplicate selected play')
    source = json_file(parent,'model_input.json')
    data_path = safe_file(parent,'model_input.npz')
    if source['signature'] != plan['signature'] or source['folds'] != selection['folds'] or digest(data_path) != source['sha256']:
        raise ValueError('Parent prepared data hash/lineage/folds changed')
    with zipfile.ZipFile(data_path) as z:
        if sum(x.file_size for x in z.infolist()) > 768*1024**2:
            raise ValueError('Parent decoded-data safety limit exceeded')
    with np.load(data_path,allow_pickle=False) as z:
        data = {k:z[k] for k in ('X','y','keys','role')}
    validate_arrays(data,c['max_parent_rows'])
    if len(np.unique(data['keys'][:,:2],axis=0)) != len(playkeys):
        raise ValueError('Parent play coverage mismatch')
    fold = next(f for f in source['folds'] if f['fold'] == 1)
    tr = np.flatnonzero(np.isin(data['keys'][:,0],fold['train_games']))
    va = np.flatnonzero(np.isin(data['keys'][:,0],fold['validation_games']))
    if not len(tr) or not len(va) or max(fold['train_games'])//100 >= min(fold['validation_games'])//100:
        raise ValueError('Chronological train/evaluation separation failed')
    if (len(tr),len(va),len(np.unique(data['keys'][va,0]))) != (c['training_rows'],c['evaluation_rows'],c['evaluation_games']):
        raise ValueError('First-fold population differs from the reviewed report')
    protocol = json_file(parent,'tree_protocol.json')
    if protocol['source'] != source or protocol['holdout_used'] is not False:
        raise ValueError('Parent model protocol is inconsistent')
    return dict(data=data,source=source,selection=selection,fold=fold,train=tr,evaluation=va,
                summary=summary,protocol=protocol,parent=parent,contract=c)


def replay_control(ctx: dict) -> tuple[np.ndarray,dict]:
    """Load only completed, hash-checked parent control models. No old-path writes."""
    from tree_core import SETTINGS, versions, checked_blob
    protocol, data, parent = ctx['protocol'],ctx['data'],ctx['parent']
    if protocol['environment'] != versions() or protocol['settings'] != SETTINGS:
        raise ValueError('Parent model environment or fixed settings differ')
    study = hash_json(protocol)
    tr,va = ctx['train'],ctx['evaluation']
    m = data['X'][:,:len(BASE_NAMES)]
    keep = np.ptp(m[tr],axis=0) > 1e-10
    xt = np.ascontiguousarray(m[tr][:,keep],dtype=np.float64)
    xe = np.ascontiguousarray(m[va][:,keep],dtype=np.float64)
    models,pred = [],[]
    for axis in range(2):
        folder = parent/'fold_1/control'/str(axis)
        receipt = json_file(folder,'checkpoint.json')
        axis_sig = hash_json({'study':study,'fold':1,'arm':'control','axis':axis,'keep':keep.tolist()})
        expected = hash_json({'parent':axis_sig,'settings':SETTINGS,'environment':versions()})
        if receipt['signature'] != expected or receipt['step'] != SETTINGS['max_iter']:
            raise ValueError('Parent control model signature/exposure mismatch')
        blob = safe_file(folder,receipt['blob'])
        model = checked_blob(folder,receipt)
        if model.n_iter_ != SETTINGS['max_iter'] or not np.array_equal(model.predict(xt[:64]),receipt['probe']):
            raise ValueError('Parent training probe cannot be replayed')
        pred.append(model.predict(xe))
        models.append({'coordinate':axis,'sha256':digest(blob),'step':model.n_iter_})
    prediction = np.column_stack(pred)
    path = safe_file(parent,'fold_1/control.npz')
    receipt = json_file(parent,'fold_1/control.json')
    if receipt['signature'] != study or receipt['sha256'] != digest(path):
        raise ValueError('Parent prediction hash mismatch')
    z = read_npz(path)
    for got,want in ((prediction,z['pred']),(data['keys'][va],z['keys']),(data['y'][va],z['truth'])):
        if not np.array_equal(got,want):
            raise ValueError('Parent control forecast/keys/targets fail exact replay')
    from origin_features import rmse
    value = rmse(data['y'][va],prediction)
    if abs(value-ctx['contract']['control_rmse']) > 1e-12:
        raise ValueError('Parent control RMSE differs from the reviewed score')
    return prediction,{'parent_models_replayed':2,'parent_refits':0,'control_rmse':value,
                       'control_replay_exact':True,'models':models}


def study_signature(kit: Path, ctx: dict) -> str:
    return hash_json({'code':{n:digest(safe_file(kit,n)) for n in CODE_FILES},
                      'parent_source':ctx['source'],'study':'direct-state-and-goal-frame-v1'})


def checkpoint_read(path: Path, sig: str, parent_sha: str) -> dict:
    r=json_file(path.parent,path.with_suffix('.json').name)
    if r['signature'] != sig or r['parent_sha256'] != parent_sha or digest(safe_file(path.parent,path.name)) != r['sha256']:
        raise ValueError('Feature checkpoint changed; preserve and diagnose')
    z=read_npz(path)
    n=len(z['keys'])
    if z['state'].shape != (n,len(DIRECT_NAMES)) or z['goal'].shape != (n,len(GOAL_NAMES)) or str(z['signature']) != sig:
        raise ValueError('New feature shape/signature mismatch')
    if not np.isfinite(z['state']).all() or not np.isfinite(z['goal']).all():
        raise ValueError('Nonfinite new features')
    return z


def prepare(ctx: dict, repo: Path, out: Path, sig: str, *, smoke: bool, event) -> dict:
    """Read observed CSVs only, and generate features on fixed first-fold plays."""
    start=time.monotonic()
    data,parent = ctx['data'],ctx['parent']
    train=set(ctx['fold']['train_games']); evaluation=set(ctx['fold']['validation_games'])
    selected=[p for p in ctx['selection']['plays'] if p['game'] in (train if smoke else train|evaluation)]
    if smoke:selected=selected[:32]
    if not selected:raise ValueError('No declared plays')
    oldfiles={}; missing=[]; reused=0
    for p in selected:
        name=f"{p['game']}_{p['play']}.npz"
        path=safe_file(parent,'features/plays/'+name)
        receipt=json_file(path.parent,path.with_suffix('.json').name)
        sha=digest(path)
        if sha!=receipt['sha256'] or receipt['signature']!=ctx['source']['signature']:
            raise ValueError('Parent play checkpoint failed checksum/source verification')
        oldfiles[name]=(path,sha)
        target=out/'features'/name
        if target.exists() or target.with_suffix('.json').exists():
            checkpoint_read(target,sig,sha);reused+=1
        else:missing.append(p)
    raw_entries={e['path']:e for e in ctx['contract']['raw_input_files']}
    built=0;parity_rows=0
    for name in sorted({p['input'] for p in missing}):
        entry=raw_entries[name];rawpath=safe_file(repo,name)
        if rawpath.stat().st_size!=entry['size'] or digest(rawpath)!=entry['sha256']:
            raise ValueError('Observed raw file hash/size mismatch')
        wanted={(p['game'],p['play']) for p in missing if p['input']==name}
        columns=list(pd.read_csv(rawpath,nrows=0).columns)
        use=[col for col in columns if col in REQUIRED|{'s','dir','o','a'}]
        parts=[]
        for part in pd.read_csv(rawpath,usecols=use,chunksize=50000):
            mask=pd.MultiIndex.from_frame(part[['game_id','play_id']]).isin(wanted)
            if mask.any():parts.append(part.loc[mask])
        if not parts:raise ValueError('Selected observed rows absent')
        for key,raw in pd.concat(parts,ignore_index=True).groupby(['game_id','play_id'],sort=True):
            filename=f'{int(key[0])}_{int(key[1])}.npz'
            oldpath,oldsha=oldfiles[filename]
            z=read_npz(oldpath,('X','keys'))
            features=build_features(raw.sort_values(['nfl_id','frame_id']).reset_index(drop=True),
                                    z['keys'],cutoff=int(raw.frame_id.max()))
            storage_parity(features['legacy'],z['X']);parity_rows+=len(z['keys'])
            target=out/'features'/filename
            atomic_npz(target,state=features['state'],goal=features['goal'],keys=z['keys'],signature=np.array(sig))
            atomic_json(target.with_suffix('.json'),{'signature':sig,'parent_sha256':oldsha,'sha256':digest(target),'rows':len(z['keys'])})
            checkpoint_read(target,sig,oldsha);built+=1
            if built%16==0:event('feature_progress',completed_plays=reused+built,total_plays=len(selected))
        # Read again to detect an external writer during preprocessing.
        if digest(rawpath)!=entry['sha256']:raise ValueError('Observed input changed during preparation')
    if built+reused!=len(selected):raise ValueError('Incomplete selected play coverage')
    statestate=[];stategoal=[]
    for p in selected:
        if p['game'] not in train:continue
        n=f"{p['game']}_{p['play']}.npz";z=checkpoint_read(out/'features'/n,sig,oldfiles[n][1])
        _,ix=np.unique(z['keys'][:,2],return_index=True)
        statestate.append(z['state'][ix]);stategoal.append(z['goal'][ix])
    st,gg=np.concatenate(statestate),np.concatenate(stategoal)
    result={'status':'state_smoke_complete' if smoke else 'state_features_ready',
            'signature':sig,'plays':len(selected),'new_checkpoints':built,'reused_checkpoints':reused,
            'parity_rows_rechecked':parity_rows,'direct_state_columns':len(DIRECT_NAMES),'goal_columns':len(GOAL_NAMES),
            'training_only_support_players':len(st),'support_columns':list(GOAL_NAMES[-6:]),
            'training_support_fractions':gg[:,-6:].mean(0).tolist(),
            'state_train_min':st.min(0).tolist(),'state_train_max':st.max(0).tolist(),
            'new_model_fits':0,'old_models_refitted':0,'raw_output_csvs_opened':0,
            'outer_validation_used':False,'elapsed_seconds':round(time.monotonic()-start,3),'feature_research':'open'}
    atomic_json(out/('smoke.json' if smoke else 'preparation.json'),result)
    event('features_complete',status=result['status'],plays=len(selected),new_checkpoints=built,reused_checkpoints=reused)
    return result


def assemble(ctx: dict, out: Path, sig: str) -> tuple[dict,str]:
    indices=np.sort(np.r_[ctx['train'],ctx['evaluation']]);data=ctx['data']
    wanted=set(map(tuple,data['keys'][indices,:2]));parts={'state':[],'goal':[],'keys':[]};hashes={}
    decoded=0
    for p in ctx['selection']['plays']:
        if (p['game'],p['play']) not in wanted:continue
        name=f"{p['game']}_{p['play']}.npz"
        old=safe_file(ctx['parent'],'features/plays/'+name)
        path=safe_file(out,'features/'+name);z=checkpoint_read(path,sig,digest(old))
        for k in parts:parts[k].append(z[k]);decoded+=z[k].nbytes
        hashes[name]=digest(path)
        if decoded>ctx['contract']['max_feature_bytes']:raise ValueError('Feature memory cap exceeded')
    extra={k:np.concatenate(v) for k,v in parts.items()}
    if not np.array_equal(extra['keys'],data['keys'][indices]):raise ValueError('New features are not exactly aligned to preserved rows')
    d={k:data[k][indices] for k in ('X','y','keys','role')}
    d.update({k:extra[k] for k in ('state','goal')})
    return d,hash_json(hashes)
