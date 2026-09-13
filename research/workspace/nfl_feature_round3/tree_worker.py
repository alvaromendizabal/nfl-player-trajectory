# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
"""Fixed-capacity tree feature-transfer experiment, not the old fitted model.

Uses the existing final fitter's exact dependency lock, but never invokes final
training. Model blobs are local, source/data-bound and hash checked before load.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn
from origin_features import ALL_NAMES, BASE_NAMES, rmse
from pair_features import NAMES
from parent_support import atomic_json, atomic_npz, _atomic, digest, hash_json, seal_json

SETTINGS={'loss':'squared_error','learning_rate':0.06,'max_iter':120,
          'max_leaf_nodes':15,'min_samples_leaf':30,'l2_regularization':1.0,
          'max_bins':127,'early_stopping':False,'warm_start':True,'random_state':20260911}
STEP=30
ARMS=('control','arrival','terminal_pairs','history_pairs')
CONTRASTS=(('arrival','control'),('terminal_pairs','arrival'),('history_pairs','terminal_pairs'))


def event(name,**kw):
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':name,**kw},allow_nan=False),flush=True)


def versions():return {'python':sys.version.split()[0],'numpy':np.__version__,'sklearn':sklearn.__version__}


def checked_blob(folder,receipt):
    name=receipt['blob']
    if Path(name).name!=name or not name.endswith('.pkl'):raise ValueError('Unsafe model checkpoint path')
    p=folder/name
    if p.is_symlink() or digest(p)!=receipt['sha256']:raise ValueError('Model checkpoint hash mismatch')
    # Only self-generated local blobs recorded under this frozen study are loaded.
    return pickle.loads(p.read_bytes())


def fit_axis(folder:Path,x,y,xe,signature,*,replay_only=False,settings=None,max_chunks=None):
    spec=dict(SETTINGS if settings is None else settings)
    folder.mkdir(parents=True,exist_ok=True)
    if any(not np.isfinite(a).all() for a in (x,y,xe)):raise ValueError('Nonfinite model arrays')
    manifest=folder/'checkpoint.json';sig=hash_json({'parent':signature,'settings':spec,'environment':versions()})
    start=0;model=None; chunks=0
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['signature']!=sig:raise ValueError('Model source/data/environment drift')
        model=checked_blob(folder,saved);start=int(saved['step'])
        if model.n_iter_!=start or start>spec['max_iter']:raise ValueError('Invalid saved boosting exposure')
        if not np.array_equal(model.predict(x[:64]),np.asarray(saved['probe'])):raise ValueError('Training probe replay differs')
    if replay_only and start!=spec['max_iter']:raise ValueError('Replay refuses to fit missing/incomplete model')
    reused=start==spec['max_iter']
    while start<spec['max_iter']:
        target=min(start+STEP,spec['max_iter'])
        if model is None:model=HistGradientBoostingRegressor(**{**spec,'max_iter':target})
        else:model.set_params(max_iter=target)
        model.fit(x,y);payload=pickle.dumps(model,protocol=5);sha=hashlib.sha256(payload).hexdigest()
        dest=folder/(sha+'.pkl')
        if not dest.exists():_atomic(dest,lambda f:f.write(payload))
        probe=model.predict(x[:64])
        receipt={'signature':sig,'step':target,'sha256':sha,'blob':dest.name,'probe':probe.tolist()}
        reread=checked_blob(folder,receipt)
        if not np.array_equal(reread.predict(x[:64]),probe):raise ValueError('Checkpoint read-back replay failed')
        atomic_json(manifest,receipt);start=target;chunks+=1
        event('boosting_checkpoint',arm=folder.parent.name,coordinate=folder.name,trees=start,total_trees=spec['max_iter'])
        if max_chunks is not None and chunks>=max_chunks and start<spec['max_iter']:
            return None,{'interrupted_for_test':True,'step':start}
    assert model is not None
    pred=model.predict(xe)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
    return pred,{'reused_complete':reused,'resumed_from_step':(json.loads(manifest.read_text())['step'] if reused else start-chunks*STEP),
                 'new_chunks':chunks,'trees':model.n_iter_,'replay_exact':True}


def paired(y,a,b,games,*,seed=20260911,repeats=10000):
    unique,inv=np.unique(games,return_inverse=True)
    if len(unique)<2:raise ValueError('At least two games required for paired interval')
    n=np.bincount(inv);sa=np.bincount(inv,weights=((y-a)**2).sum(1));sb=np.bincount(inv,weights=((y-b)**2).sum(1))
    rng=np.random.default_rng(seed);d=[]
    for _ in range((repeats+499)//500):
        ix=rng.integers(0,len(unique),(min(500,repeats-len(d)),len(unique)))
        d.extend((np.sqrt(sb[ix].sum(1)/(2*n[ix].sum(1)))-np.sqrt(sa[ix].sum(1)/(2*n[ix].sum(1)))).tolist())
    # Six planned looks: three first-fold contrasts and three pooled follow-ups.
    lo,hi=np.quantile(d,[.05/(2*6),1-.05/(2*6)])
    gain=1-rmse(y,b)/rmse(y,a)
    return {'delta_rmse':rmse(y,b)-rmse(y,a),'relative_gain':gain,'adjusted_low':float(lo),'adjusted_high':float(hi),
            'resamples':repeats,'planned_contrast_looks':6,'screen_gate':bool(gain>=.01 and hi<0)}


def feature_matrices(d):
    base=d['X']
    return {'control':base[:,:len(BASE_NAMES)],'arrival':base,
            'terminal_pairs':np.column_stack([base,d['terminal']]),
            'history_pairs':np.column_stack([base,d['history']])}


def run_fold(out:Path,fold:int,*,replay_only=False):
    start=time.monotonic(); source=json.loads((out/'model_input.json').read_text());path=out/'model_input.npz'
    if digest(path)!=source['sha256']:raise ValueError('Prepared data hash changed')
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    if sum(v.nbytes for v in d.values())>768*1024**2:raise ValueError('Data memory cap exceeded')
    f=next(s for s in source['folds'] if s['fold']==fold)
    tr=np.flatnonzero(np.isin(d['keys'][:,0],f['train_games']));va=np.flatnonzero(np.isin(d['keys'][:,0],f['validation_games']))
    if not len(tr) or not len(va) or max(f['train_games'])//100>=min(f['validation_games'])//100:raise ValueError('Invalid chronological fold')
    if fold>1:
        first=json.loads((out/'fold_1'/'summary.json').read_text())
        if first['source_signature']!=source['signature'] or not first['continue_research']:
            raise ValueError('First-fold gate did not authorize additional folds')
    spec={'source':source,'environment':versions(),'settings':SETTINGS,'arms':list(ARMS),'first_fold':1,
          'holdout_used':False,'new_kaggle_score':None}
    seal_json(out/'tree_protocol.json',spec);sig=hash_json(spec)
    matrices=feature_matrices(d);y=d['y'];preds={};fit_receipts=[];keeps={}
    for arm,m in matrices.items():
        keep=np.ptp(m[tr],axis=0)>1e-10
        if arm in ('terminal_pairs','history_pairs'):
            keep=(np.ptp(matrices['terminal_pairs'][tr],axis=0)>1e-10)|(np.ptp(matrices['history_pairs'][tr],axis=0)>1e-10)
        keeps[arm]=keep
        xx=np.ascontiguousarray(m[tr][:,keep],dtype=np.float64);xe=np.ascontiguousarray(m[va][:,keep],dtype=np.float64)
        pair=[]
        for axis in range(2):
            p,receipt=fit_axis(out/f'fold_{fold}'/arm/str(axis),xx,y[tr,axis],xe,hash_json({'study':sig,'fold':fold,'arm':arm,'axis':axis,'keep':keep.tolist()}),replay_only=replay_only)
            pair.append(p);fit_receipts.append({'arm':arm,'coordinate':axis,**receipt})
        preds[arm]=np.column_stack(pair)
        prediction=out/f'fold_{fold}'/(arm+'.npz');pr=prediction.with_suffix('.json')
        if prediction.exists():
            if not pr.exists() or json.loads(pr.read_text())['signature']!=sig or digest(prediction)!=json.loads(pr.read_text())['sha256']:raise ValueError('Prediction receipt mismatch')
            with np.load(prediction,allow_pickle=False) as old:
                if not np.array_equal(preds[arm],old['pred']) or not np.array_equal(old['keys'],d['keys'][va]):raise ValueError('Saved forecast replay differs')
        elif replay_only:raise ValueError('Missing prediction; replay cannot create a replacement')
        else:
            atomic_npz(prediction,pred=preds[arm],keys=d['keys'][va],truth=y[va])
            atomic_json(pr,{'signature':sig,'sha256':digest(prediction)})
    contrasts=[{'treatment':b,'control':a,**paired(y[va],preds[a],preds[b],d['keys'][va,0])} for b,a in CONTRASTS]
    slices=[]
    for arm,p in preds.items():
        for label,mask in [('first_second',d['keys'][va,3]<=10),('after_first_second',d['keys'][va,3]>10)]+[(f'role_{i}',d['role'][va]==i) for i in range(4)]:
            if mask.any():slices.append({'arm':arm,'slice':label,'rows':int(mask.sum()),'sse':float(((y[va][mask]-p[mask])**2).sum()),'rmse':rmse(y[va][mask],p[mask])})
    result={'status':'first_fold_complete' if fold==1 else 'followup_fold_complete','fold':fold,'source_signature':source['signature'],
            'scope':'Expanded sample, reused training-side game folds. Fixed-tree feature transfer, not production or Kaggle performance.',
            'training_rows':len(tr),'evaluation_rows':len(va),'evaluation_games':len(np.unique(d['keys'][va,0])),
            'pooled_rmse':{a:rmse(y[va],p) for a,p in preds.items()},'contrasts':contrasts,
            'continue_research':any(c['screen_gate'] for c in contrasts),'slices':slices,
            'retained_columns':{a:int(k.sum()) for a,k in keeps.items()},'fits':fit_receipts,
            'new_coordinate_models':sum(not r['reused_complete'] for r in fit_receipts),'old_models_refitted':0,
            'elapsed_seconds':round(time.monotonic()-start,3),'feature_research':'open','github_updated':False,
            'kaggle_score':None,'outer_validation_used':False,'all_model_replays_exact':True}
    result['milestone_review']={'attempted':'Four matched feature arms on one frozen chronological fold',
        'completed':result['status'],'passed':{'all_model_replays_exact':True,'gated_contrasts':[c['treatment'] for c in contrasts if c['screen_gate']]},
        'underperformed':[c['treatment'] for c in contrasts if not c['screen_gate']],
        'actual_metrics':result['pooled_rmse'],'artifacts':f'fold_{fold} model checkpoints and prediction arrays',
        'github_updated':False,'learned':'Use the paired contrasts; cross-round absolute RMSE is not a controlled comparison.',
        'next':'Return this report before considering unchanged follow-up folds.' if result['continue_research'] else 'Stop scaling this interface unchanged; review diagnostics.',
        'compute_justification':'Only a passed representation contrast earns additional temporal validation.'}
    atomic_json(out/f'fold_{fold}'/('replay.json' if replay_only else 'summary.json'),result)
    event('fold_complete',fold=fold,metrics=result['pooled_rmse'],continue_research=result['continue_research'])
    return result


def self_test(out):
    rng=np.random.default_rng(717);x=rng.normal(size=(256,8));y=x[:,0]**2+x[:,1];e=x[:97]
    spec={**SETTINGS,'max_iter':60,'min_samples_leaf':10}
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td)
        fit_axis(folder/'resume',x,y,e,'synthetic-recovery',settings=spec,max_chunks=1)
        pred,_=fit_axis(folder/'resume',x,y,e,'synthetic-recovery',settings=spec)
        clean=HistGradientBoostingRegressor(**spec).fit(x,y).predict(e)
        assert np.array_equal(clean,pred),'Checkpoint continuation differs from clean fit'
        replay,r=fit_axis(folder/'resume',x,y,e,'synthetic-recovery',settings=spec,replay_only=True)
        assert np.array_equal(pred,replay) and r['new_chunks']==0
    result={'status':'tree_runtime_ready','environment':versions(),'synthetic_recovery_exact':True,'private_data_used':False}
    atomic_json(out/'runtime.json',result);event('runtime_ready',**result)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['runtime','fit','replay']);p.add_argument('--out',type=Path,required=True);p.add_argument('--fold',type=int,default=1,choices=[1,2,3]);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.command=='runtime':self_test(a.out)
    else:run_fold(a.out,a.fold,replay_only=a.command=='replay')
if __name__=='__main__':main()
