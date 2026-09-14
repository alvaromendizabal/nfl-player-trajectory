# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "boto3==1.43.89", "numpy==2.4.6", "pandas==3.0.5", "plotly==7.0.0",
#   "matplotlib==3.10.8", "filelock==3.32.5", "torch==2.8.0", "pytest==9.1.1",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
from __future__ import annotations
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from neural_study import (protocol,train_arm,evaluate,event,environment)
from sequence_model import (setup,Forecast,CONFIG,collate,ordered_batches,step)
from parent_support import atomic_json,seal_json,safe_file


def profile(out,kit):
    train,ev,m,ref,p,psig=protocol(out,kit)
    # Largest observed tensors/row counts; never rank by label or error values.
    ix=sorted(range(len(train)),key=lambda i:(len(train[i]['ids']),len(train[i]['keys'])),reverse=True)[:8]
    items=[train[i] for i in ix];target=torch.tensor(np.concatenate([x['y'] for x in items]),dtype=torch.float32)
    timings={};total_steps=len(ordered_batches(len(train),0))*CONFIG['epochs']
    for arm in ('terminal','history'):
        setup();model=Forecast();opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'],foreach=False)
        times=[]
        for i in range(8):
            started=time.monotonic();b=collate(items,arm,p['normalization'])
            step(model,opt,b,target,2*len(target));elapsed=time.monotonic()-started
            if i>0:times.append(elapsed)
        timings[arm]={'p90_step_seconds':float(np.quantile(times,.9)),'conservative_seconds':float(np.quantile(times,.9)*total_steps*1.5+60)}
    ready=all(v['conservative_seconds']<=300 for v in timings.values())
    r={'status':'training_profile_passed' if ready else 'training_profile_budget_stop','protocol_signature':psig,
       'timings':timings,'profile_plays':len(items),'disposable_optimizer_steps':16,
       'scientific_fits':0,'evaluation_scored':False,'planned_steps_per_arm':total_steps,
       'seconds_limit_per_arm':360,'budget_guard_seconds':300,
       'interpretation':'Timing guard, not an estimated improvement or promised runtime'}
    atomic_json(out/'profile.json',r);event('profile_complete',**{k:r[k] for k in ['status','planned_steps_per_arm']})
    if not ready:raise ValueError('Observed CPU throughput exceeds fixed budget; return report, no scientific fit authorized')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['runtime','profile','train','evaluate','replay'])
    for n in ['kit','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arm',choices=['terminal','history'],default='terminal');a=p.parse_args()
    setup()
    env=environment()
    if (sys.version_info[:2]!=(3,11) or np.__version__!='2.4.6' or torch.__version__!='2.8.0+cpu'):
        raise ValueError('Runtime differs from verified neural lock; do not change the notebook environment')
    if a.command=='runtime':
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.kit/'tests/model'),'-v'],cwd=a.kit,check=True,timeout=90)
        r={'status':'sequence_runtime_ready','environment':env,'local_model_tests_passed':True,'private_data_used':False}
        seal_json(a.out/'runtime_receipt.json',r)
    else:
        r=json.loads(safe_file(a.out,'runtime_receipt.json').read_text())
        if r['environment']!=env or r['status']!='sequence_runtime_ready':raise ValueError('Run runtime verification first')
        if a.command=='profile':r=profile(a.out,a.kit)
        elif a.command=='train':r=train_arm(a.out,a.kit,a.arm)
        elif a.command=='evaluate':r=evaluate(a.out,a.kit)
        else:
            rs=[train_arm(a.out,a.kit,x,replay_only=True) for x in ['terminal','history']]
            if any(x['new_optimizer_steps'] for x in rs):raise ValueError('Replay unexpectedly trained')
            evaluate(a.out,a.kit)
            r={'status':'matched_encoder_replay_exact','new_optimizer_steps':0,'models_replayed':2,'arms':rs}
            atomic_json(a.out/'replay.json',r)
    event('stage_complete',stage=a.command,status=r['status'])
if __name__=='__main__':main()
