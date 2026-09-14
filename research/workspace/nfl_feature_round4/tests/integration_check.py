"""Synthetic-only integration harness. Not invoked by the production runner.

Identity is fixture-specific; checksums, chronology, feature parity, source
signatures and numerical model recovery use the actual production functions.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
KIT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(KIT));sys.path.insert(0,str(KIT/'tests/base'))
from fixtures import fixture
from origin_features import make_example,chronological_folds,BASE_NAMES,rmse
from parent_support import digest,hash_json,atomic_json,atomic_npz
from tree_core import SETTINGS,versions,fit_axis
from study import parent_context,study_signature,prepare,replay_control,assemble
from feature_worker import fit_study


def event(n,**kw):print(n,kw,flush=True)


def setup(root):
    repo=root/'repo';parent=root/'parent';legacy=root/'legacy';out=root/'out'
    for p in (repo,parent,legacy,out):p.mkdir(parents=True,exist_ok=True)
    (parent/'.run.lock').write_text('')
    c=json.loads((KIT/'input_contract.json').read_text())
    shutil.copy2(KIT/'origin_features.py',legacy/'origin_features.py')
    shutil.copy2(KIT/'tree_core.py',legacy/'tree_worker.py')
    c['parent_source_hashes']={p.name:digest(p) for p in legacy.iterdir()}
    c['parent_source_signature']='synthetic_round3'
    rawparts=[];outputs=[]
    for date in range(1,17):
        for play in range(4):
            r,q=fixture();game=2023090001+date*100;p=100+play
            r.game_id=game;r.play_id=p;r.x+=play;r.ball_land_x+=play+.2*date
            q[:,0]=game;q[:,1]=p
            y=pd.DataFrame(q,columns=['game_id','play_id','nfl_id','frame_id'])
            y['x']=np.where(y.nfl_id==1,36,34)+play+3*y.frame_id/10+.03*(y.frame_id/10)**2*(8+date*.3)
            y['y']=np.where(y.nfl_id==1,20,22)-.13*(y.frame_id/10)**2
            rawparts.append(r);outputs.append(y)
    path=repo/'data/raw/train/input_2023_w01.csv';path.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(rawparts).to_csv(path,index=False)
    # Only fixture labels are materialized here, never read by Round 4 preparation.
    yy=pd.concat(outputs);raw=pd.read_csv(path)
    c['raw_input_files']=[{'path':str(path.relative_to(repo)),'size':path.stat().st_size,'sha256':digest(path)}]
    entries=[];parts={k:[] for k in ('X','y','keys','role')}
    for key,r in raw.groupby(['game_id','play_id'],sort=True):
        y=yy.loc[(yy.game_id==key[0])&(yy.play_id==key[1])]
        z=make_example(r,y,0)
        if int(key[1])%2==0:
            z['X']=z['X'].astype(np.float32);z['y']=z['y'].astype(np.float32)
        for k in parts:parts[k].append(z[k])
        name=f'{int(key[0])}_{int(key[1])}.npz';f=parent/'features/plays'/name
        atomic_npz(f,**{k:z[k] for k in parts})
        atomic_json(f.with_suffix('.json'),{'signature':c['parent_source_signature'],'sha256':digest(f)})
        entries.append({'game':int(key[0]),'play':int(key[1]),'input':str(path.relative_to(repo))})
    d={k:np.concatenate(v) for k,v in parts.items()}
    folds=[{'fold':f['fold'],'train_games':f['train_games'].tolist(),'validation_games':f['validation_games'].tolist()} for f in chronological_folds(d['keys'][:,0])]
    selection={'signature':c['parent_source_signature'],'plays':entries,'folds':folds}
    atomic_json(parent/'selection.json',selection);atomic_json(parent/'selection.receipt.json',{'sha256':digest(parent/'selection.json')})
    atomic_json(parent/'plan.json',{'signature':c['parent_source_signature'],'source_hashes':c['parent_source_hashes']})
    atomic_npz(parent/'model_input.npz',**d)
    source={'signature':c['parent_source_signature'],'sha256':digest(parent/'model_input.npz'),'folds':folds}
    atomic_json(parent/'model_input.json',source)
    protocol={'source':source,'settings':SETTINGS,'environment':versions(),'arms':['control'], 'holdout_used':False}
    atomic_json(parent/'tree_protocol.json',protocol)
    study=hash_json(protocol);fold=folds[0]
    tr=np.flatnonzero(np.isin(d['keys'][:,0],fold['train_games']));va=np.flatnonzero(np.isin(d['keys'][:,0],fold['validation_games']))
    m=d['X'][:,:len(BASE_NAMES)];keep=np.ptp(m[tr],axis=0)>1e-10
    xt=np.ascontiguousarray(m[tr][:,keep],dtype=float);xe=np.ascontiguousarray(m[va][:,keep],dtype=float)
    preds=[]
    for axis in range(2):
        sig=hash_json({'study':study,'fold':1,'arm':'control','axis':axis,'keep':keep.tolist()})
        pred,_=fit_axis(parent/'fold_1/control'/str(axis),xt,d['y'][tr,axis],xe,sig)
        preds.append(pred)
    pred=np.column_stack(preds)
    atomic_npz(parent/'fold_1/control.npz',pred=pred,truth=d['y'][va],keys=d['keys'][va])
    atomic_json(parent/'fold_1/control.json',{'signature':study,'sha256':digest(parent/'fold_1/control.npz')})
    summary={'status':'first_fold_complete','source_signature':c['parent_source_signature'],
             'pooled_rmse':{'control':rmse(d['y'][va],pred)},'contrasts':[], 'all_model_replays_exact':True,
             'new_coordinate_models':0,'training_rows':len(tr),'evaluation_rows':len(va),
             'evaluation_games':len(np.unique(d['keys'][va,0])),'fixture_only':True}
    for n in ('summary','replay'):atomic_json(parent/f'fold_1/{n}.json',summary)
    atomic_json(parent/'preparation.json',{'status':'expanded_features_ready','fixture_only':True})
    c.update(training_rows=len(tr),evaluation_rows=len(va),evaluation_games=len(np.unique(d['keys'][va,0])),control_rmse=summary['pooled_rmse']['control'])
    for n,field in [('fold_1/summary.json','reviewed_summary_sha256'),('fold_1/replay.json','reviewed_replay_sha256'),('preparation.json','reviewed_preparation_sha256')]:c[field]=digest(parent/n)
    atomic_json(root/'fixture_contract.json',c)


def context(root):
    c=json.loads((root/'fixture_contract.json').read_text())
    return parent_context(root/'parent',root/'legacy',root/'repo',c,verify_identity=False)


def main():
    root=Path(sys.argv[1]);stage=sys.argv[2] if len(sys.argv)>2 else 'all'
    if stage=='setup':setup(root);return
    if stage=='all':setup(root)
    ctx=context(root);out=root/'out';sig=study_signature(KIT,ctx)
    if stage=='preflight':
        _,r=replay_control(ctx);atomic_json(out/'preflight.json',{'status':'round4_preflight_passed','signature':sig,'parent_replay':r,'synthetic_only':True});return
    if stage=='smoke':prepare(ctx,root/'repo',out,sig,smoke=True,event=event);return
    if stage=='prepare':prepare(ctx,root/'repo',out,sig,smoke=False,event=event);return
    if stage in ('fit','replay'):
        fit_study(ctx,out,sig,replay_only=stage=='replay');return
    if stage=='report':
        from run_round import export_report
        export_report(out);return
    original={str(p):(digest(p),p.stat().st_mtime_ns) for directory in (root/'parent',root/'repo',root/'legacy') for p in directory.rglob('*') if p.is_file()}
    _,replay=replay_control(ctx)
    a=prepare(ctx,root/'repo',out,sig,smoke=True,event=event)
    b=prepare(ctx,root/'repo',out,sig,smoke=False,event=event)
    c=prepare(ctx,root/'repo',out,sig,smoke=False,event=event)
    assert a['plays']==32 and b['reused_checkpoints']==32 and c['new_checkpoints']==0
    result=fit_study(ctx,out,sig)
    assert result['new_coordinate_models']==4 and result['parent_refits']==0
    weights={str(p):(digest(p),p.stat().st_mtime_ns) for p in (out/'models').rglob('*.pkl')}
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(root),'replay'],check=True,timeout=60)
    replay=json.loads((out/'replay.json').read_text())
    assert replay['new_coordinate_models']==0
    assert all((digest(Path(p)),Path(p).stat().st_mtime_ns)==v for p,v in weights.items())
    assert all((digest(Path(p)),Path(p).stat().st_mtime_ns)==v for p,v in original.items())
    receipt={'synthetic_only':True,'selected_plays':64,'first_fold_plays':b['plays'],'new_coordinate_models':4,
             'fresh_process_replay_new_models':0,'parent_models_replayed':2,'parent_refits':0,
             'parent_raw_source_hashes_and_mtimes_unchanged':True,'new_model_hashes_and_mtimes_unchanged_on_replay':True,
             'float32_and_float64_parent_storage_exercised':True,'feature_checkpoint_reuse':c['plays'],
             'private_NFL_data_used':False}
    atomic_json(root/'integration.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
