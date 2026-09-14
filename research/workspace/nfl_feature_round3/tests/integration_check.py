"""Synthetic-only local integration; not an NFL accuracy experiment."""
from pathlib import Path
import sys,json,hashlib,subprocess,os,time
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).parent/'base'))
from test_pairs import fixture
from origin_features import make_example,chronological_folds
from data_pipeline import prepare
from parent_support import atomic_json,digest
from run_round import model_input,export_report


def main():
    root=Path(sys.argv[1]);repo=root/'repo';out=root/'out';repo.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
    rawparts=[];yparts=[];old=[]
    for date in range(1,17):
        for p in range(4):
            raw,keys=fixture();g=2023090001+100*date;play=100+p
            raw.game_id=g;raw.play_id=play;raw.x+=p;raw.ball_land_x+=p+date*.2
            keys[:,0]=g;keys[:,1]=play
            y=pd.DataFrame(keys,columns=['game_id','play_id','nfl_id','frame_id'])
            y['x']=np.where(y.nfl_id==1,36,34)+p+3*y.frame_id/10+.035*(y.frame_id/10)**2*(9+date*.2)
            y['y']=np.where(y.nfl_id==1,20,22)-.15*(y.frame_id/10)**2
            rawparts.append(raw);yparts.append(y)
            if p<2:
                example=make_example(raw,y,0)
                for field in ('X','y','cv'): example[field]=example[field].astype(np.float32)
                old.append(example)
    rawpath=repo/'data/raw/train/input_2023_w01.csv';rawpath.parent.mkdir(parents=True,exist_ok=True)
    outpath=rawpath.with_name('output_2023_w01.csv');pd.concat(rawparts).to_csv(rawpath,index=False);pd.concat(yparts).to_csv(outpath,index=False)
    contract={'raw_files':[{'path':str(p.relative_to(repo)),'size':p.stat().st_size,'sha256':digest(p)} for p in (rawpath,outpath)]}
    parent={'data':{k:np.concatenate([z[k] for z in old]) for k in ('X','y','keys','role')}}
    parent['folds']=chronological_folds(parent['data']['keys'][:,0])
    sig='synthetic-engineering-only';events=lambda event,**kw:print(event,kw,flush=True)
    before={str(p):digest(p) for p in (rawpath,outpath)}
    a=prepare(repo,out,parent,contract,sig,smoke=True,event=events,total=64)
    b=prepare(repo,out,parent,contract,sig,smoke=False,event=events,total=64)
    assert a['plays']==32 and b['plays']==64 and b['reused_checkpoints']==32
    c=prepare(repo,out,parent,contract,sig,smoke=False,event=events,total=64)
    assert c['new_checkpoints']==0
    model_input(SimpleNamespace(out=out),sig)
    env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    script=Path(__file__).resolve().parents[1]/'tree_worker.py'
    subprocess.run([sys.executable,str(script),'fit','--out',str(out),'--fold','1'],check=True,env=env,timeout=90)
    saved={str(p):(digest(p),p.stat().st_mtime_ns) for p in (out/'fold_1').rglob('*.pkl')}
    subprocess.run([sys.executable,str(script),'replay','--out',str(out),'--fold','1'],check=True,env=env,timeout=30)
    assert all((digest(Path(p)),Path(p).stat().st_mtime_ns)==v for p,v in saved.items())
    assert all(digest(Path(p))==v for p,v in before.items())
    fit=json.loads((out/'fold_1/summary.json').read_text());replay=json.loads((out/'fold_1/replay.json').read_text())
    assert fit['new_coordinate_models']==8 and replay['new_coordinate_models']==0
    export_report(out)
    receipt={'synthetic_only':True,'prepared_plays':64,'first_fold_coordinate_models':8,'replay_new_models':0,
             'raw_files_unchanged':True,'model_hashes_and_mtimes_unchanged_on_replay':True,'feature_checkpoints_reused':64,
             'all_forward_replays_exact':True,'note':'No private NFL data. Runtime-lock setup not exercised in the unavailable AWS environment.'}
    atomic_json(root/'integration.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
