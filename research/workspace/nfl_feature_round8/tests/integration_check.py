"""Local synthetic integration. Does not claim private AWS execution."""
import sys,json,tempfile,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from fixtures import raw_play,sample
from parent_support import atomic_json,atomic_npz,digest
from sequence_data import prepare
from neural_study import protocol,train_arm,evaluate,environment
from neural_worker import profile
from sequence_model import CONFIG
from run_round import export_report

kit=Path(__file__).resolve().parents[1]
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
repo=root/'repo';parent=root/'parent';out=root/'out';out.mkdir(exist_ok=True)
rawfile=repo/'data/raw/train/input_2023_w01.csv';rawfile.parent.mkdir(parents=True,exist_ok=True)
# 32 training + 8 evaluation plays; only a synthetic CPU fixture.
rows=[];selected=[]
for i in range(40):
    game=(2023090700+(i//8)*100) if i<32 else (2023091100+((i-32)//4)*100);play=i+1
    rows.append(raw_play(game,play,n=4+(i%3)));selected.append({'game':game,'play':play,'input':'data/raw/train/input_2023_w01.csv'})
pd.concat(rows,ignore_index=True).to_csv(rawfile,index=False)
# Read the actual CSV bytes before forming the original-feature test fixtures.
raw=pd.read_csv(rawfile);parts=[];sig='synthetic-observed-sequence-v1'
for i,((g,p),one) in enumerate(raw.groupby(['game_id','play_id'],sort=True)):
    a=sample(one);x=a['base'];full=__import__('origin_features').query_features(__import__('origin_features').build_view(one,origin=20),a['keys'][:,2],a['keys'][:,3]/10)[0]
    full=full.astype(np.float32 if i<10 else np.float64)
    f=parent/f'features/plays/{int(g)}_{int(p)}.npz';atomic_npz(f,X=full,y=a['y'],keys=a['keys'])
    atomic_json(f.with_suffix('.json'),{'sha256':digest(f),'signature':'old'})
    parts.append(dict(X=full,y=a['y'],keys=a['keys'],role=a['role_query']))
data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
tr=np.flatnonzero(data['keys'][:,0]<2023091100);va=np.flatnonzero(data['keys'][:,0]>=2023091100)
ctx={'signature':sig,'data':data,'tr':tr,'va':va,'selection':{'plays':selected},'parent':parent,
     'contract':{'raw_input_files':[{'path':'data/raw/train/input_2023_w01.csv','size':rawfile.stat().st_size,'sha256':digest(rawfile)}]}}
a=SimpleNamespace(repo=repo,out=out,kit=kit)
from neural_study import event
prepare(a,ctx,smoke=True,event=event)
with patch('sequence_data.replay_parent',return_value=({'control':data['y'][va]*.5},[{'fixture':True}])):
    prep=prepare(a,ctx,event=event)
    again=prepare(a,ctx,event=event)
assert again['new_checkpoints']==0
profile(out,kit)
# Exercise interruption at two optimizer steps, then continue the same signature.
partial=train_arm(out,kit,'terminal',max_steps=2);assert partial['status']=='incomplete_checkpoint_preserved'
a1=train_arm(out,kit,'terminal');a2=train_arm(out,kit,'history');result=evaluate(out,kit)
protected=[rawfile]+list(parent.rglob('*.npz'))+list((out/'models').rglob('*.pt'))+list((out/'models').rglob('checkpoint.json'))+list((out/'plays').glob('*'))
before={str(p):digest(p) for p in protected}
cmd=[sys.executable,str(kit/'tests/replay_check.py'),str(out),str(kit)]
subprocess.run(cmd,check=True,timeout=60)
assert before=={str(p):digest(p) for p in protected}
report={'mode':'synthetic_only','plays':40,'training_rows':len(tr),'evaluation_rows':len(va),'environment':environment(),
        'new_scientific_fixture_models':2,'partial_resume_used':True,'all_completed_features_reused':True,
        'fresh_process_replay_exact':True,'protected_files_unchanged':len(protected),'metrics_not_nfl':result['metrics']}
atomic_json(root/'integration_result.json',report);print(json.dumps(report,indent=2))
