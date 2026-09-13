"""LOCAL TEST HARNESS ONLY: replace private-data ingestion, not numerical workers.

This script never connects to AWS. The fixture creates its own control models;
those are not NFL fits or user checkpoints. Production has no fixture bypass.
"""
from __future__ import annotations
from datetime import date,timedelta
import argparse,hashlib,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

KIT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(KIT))
from parent_support import atomic_json,atomic_npz,digest,hash_json
from history_features import from_arrays
import worker
import run_round
from history_study import signature,indices


def fixture_context(root, tree_kit):
    sys.path.insert(0,str(tree_kit))
    # The checked original fitter is exercised, with unchanged scientific settings.
    from tree_core import SETTINGS,versions,fit_axis
    rng=np.random.default_rng(803);keys=[];roles=[];rows=[];targets=[];games=[]
    for day in range(16):
        gid=int((date(2023,9,1)+timedelta(days=day*4)).strftime('%Y%m%d'))*100
        games.append(gid)
        for play in range(4):
            for player in range(3):
                speed=2.+player*.8
                horizon=1.2 if play%2 else 1.8
                for f in range(1,int(round(horizon*10))+1):
                    t=f/10;key=[gid,100+play,10+player,f]
                    x=np.zeros(96);x[0:6]=[1,t,t*t,horizon-t,t/horizon,0]
                    # Synthetic football-like clocks plus nuisance observed fields.
                    x[6:68]=rng.normal(size=62)*t
                    x[6+player]=t;x[68:72]=rng.normal(size=4)
                    target=np.array([t*(.7+player*.2)+.03*np.sin(t*4),-t*.3*player])
                    target+=rng.normal(scale=.015,size=2)
                    keys.append(key);roles.append(player%2);rows.append(x);targets.append(target)
    d={'keys':np.asarray(keys,np.int64),'role':np.asarray(roles,np.int8),'X':np.asarray(rows),'y':np.asarray(targets)}
    folds=[{'fold':1,'train_games':games[:8],'validation_games':games[8:10]},
           {'fold':2,'train_games':games[:10],'validation_games':games[10:13]},
           {'fold':3,'train_games':games[:13],'validation_games':games[13:]}]
    rawhash=hashlib.sha256(b''.join(a.tobytes() for a in d.values())).hexdigest()
    parent_exp='synthetic-parent-experiment'
    conf=SimpleNamespace(matrices=lambda d:{'control':d['X'][:,:72]},
                         training_mask=lambda m,tr:np.ptp(m[tr],axis=0)>1e-10)
    def fi(data,f):return (np.flatnonzero(np.isin(data['keys'][:,0],f['train_games'])),np.flatnonzero(np.isin(data['keys'][:,0],f['validation_games'])))
    conf.fold_indices=fi
    from origin_features import rmse
    def slices(y,preds,k,role):
        records=[]
        for arm,p in preds.items():
            for label,mask in [('first_second',k[:,3]<=10),('after_first_second',k[:,3]>10)]+[(f'role_{i}',role==i) for i in range(4)]:
                if mask.any():records.append({'arm':arm,'slice':label,'rows':int(mask.sum()),'sse':float(np.square(y[mask]-p[mask]).sum()),'rmse':rmse(y[mask],p[mask])})
        return records
    conf.slice_statistics=slices
    selection={'plays':[{'game':g,'play':100+p} for g in games for p in range(4)]}
    ctx={'kit':KIT,'data':d,'metadata':from_arrays(d['keys'],d['role'],d['X']),
         'source':{'sha256':rawhash,'folds':folds},'r5_experiment':parent_exp,
         'r5_out':root/'parents','confirmation':conf,'selection':selection,
         'contract7':{'study':'SYNTHETIC-order-history'},'populations':[]}
    for f in (1,2,3):
        tr,va=fi(d,folds[f-1]);ctx['populations'].append({'fold':f,'training_rows':len(tr),'evaluation_rows':len(va),'training_games':len(folds[f-1]['train_games']),'evaluation_games':len(folds[f-1]['validation_games'])})
    for f in (2,3):
        if (root/f'parents/fold_{f}/summary.json').exists():continue
        tr,va=fi(d,folds[f-1]);m=d['X'][:,:72];keep=conf.training_mask(m,tr)
        parts=[]
        for axis in range(2):
            s=hash_json({'study':parent_exp,'fold':f,'arm':'control','axis':axis,'keep':keep.tolist()})
            p,_=fit_axis(root/f'parents/fold_{f}/models/control'/str(axis),np.ascontiguousarray(m[tr][:,keep]),d['y'][tr,axis],np.ascontiguousarray(m[va][:,keep]),s)
            parts.append(p)
        p=np.column_stack(parts);dest=root/f'parents/fold_{f}/predictions/control.npz'
        atomic_npz(dest,pred=p,keys=d['keys'][va],truth=d['y'][va]);atomic_json(dest.with_suffix('.json'),{'signature':parent_exp,'sha256':digest(dest)})
        atomic_json(root/f'parents/fold_{f}/summary.json',{'experiment_signature':parent_exp,'metrics':{'control':rmse(d['y'][va],p)}})
    return ctx


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','smoke','prepare','fit','replay','summarize','report']);p.add_argument('--root',type=Path,required=True);p.add_argument('--tree-kit',type=Path,required=True);p.add_argument('--fold',type=int,default=2);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);ctx=fixture_context(a.root,a.tree_kit)
    out=a.root/'results';out.mkdir(exist_ok=True)
    if a.stage=='report':
        run_round.export_report(out);return
    worker.context=lambda args:ctx # test-only substitution of private ingestion
    args=SimpleNamespace(command=a.stage,fold=a.fold,out=out)
    r=worker.execute(args)
    print(json.dumps({'fixture_only':True,'stage':a.stage,'status':r['status']}))

if __name__=='__main__':main()
