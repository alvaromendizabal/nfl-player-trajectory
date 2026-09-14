"""Bounded, resumable, local-only feature experiment. Run with the existing venv.

No package installation, Git writes, AWS calls, or Kaggle downloads/submissions.
Run each command in foreground. Successful per-play/per-arm artifacts are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

KIT=Path(__file__).resolve().parent
DEFAULT_REPO=Path('/home/sagemaker-user/nfl-player-trajectory')
DEFAULT_OUT=Path('/home/sagemaker-user/nfl-feature-round1-results')
SEED=20260911
OFFSETS=(0,5,10,20)


def log(event,**kwargs):
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kwargs},default=str),flush=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def source_signature():
    import numpy as np,pandas as pd
    obj={'files':{p.name:digest(p) for p in [KIT/'origin_features.py', KIT/'run_round.py']},
         'contract':digest(KIT/'input_contract.json'),'python':sys.version.split()[0],
         'numpy':np.__version__,'pandas':pd.__version__}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def regular(path):
    path=Path(path)
    if not path.is_file() or path.is_symlink():raise ValueError(f'Regular file required: {path}')
    return path


def check_repo(repo):
    import numpy as np,pandas as pd
    contract=json.loads((KIT/'input_contract.json').read_text())
    if sys.version_info[:2]!=(3,11):raise ValueError('Use the existing NFL Python 3.11 venv, not system Python.')
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,text=True,capture_output=True,check=True,timeout=10)
    head=result.stdout.strip()
    if head!=contract['repo_commit']:raise ValueError('Repository revision differs from the reviewed source. Stop; do not reset or pull blindly.')
    cache=regular(repo/'artifacts/temporal/research/inner_1/samples.pkl')
    if digest(cache)!=contract['sample_sha256']:raise ValueError('Trusted sample cache hash mismatch; pickle loading blocked.')
    # This is the user's previously verified private cache, not an arbitrary pickle.
    with cache.open('rb') as f:samples=pickle.load(f)
    training=[s for s in samples if str(s['split'])=='train']
    if len(training)!=4951:raise ValueError('Frozen training play count changed.')
    # Only split and join keys are read. Cached target-derived predictors and models
    # are NEVER used. Validation samples are not transformed, screened, or scored.
    keys=sorted(set((int(s['keys'][0,0]),int(s['keys'][0,1])) for s in training))
    if len(keys)!=len(training) or len(set(k[0] for k in keys))!=94:
        raise ValueError('Training key/game population differs from frozen 4,951 plays / 94 games.')
    del samples,training
    info={'status':'preflight_passed','repo_head':head,'training_plays':len(keys),
          'training_games':94,'python':sys.version.split()[0],'numpy':np.__version__,
          'pandas':pd.__version__,'new_fits':0,'outer_validation_used':False,
          'existing_model_replay':False,'source_signature':source_signature()}
    return keys,contract,info


def choose_plays(keys,count):
    """Round-robin across games, hashed order; no outcome-dependent sampling."""
    by={}
    for k in keys:by.setdefault(k[0],[]).append(k)
    def h(k):return hashlib.sha256(f'{SEED}:{k}'.encode()).hexdigest()
    for g in by:by[g].sort(key=h)
    games=sorted(by,key=h);answer=[];round_=0
    while len(answer)<min(count,len(keys)):
        for g in games:
            if round_<len(by[g]):answer.append(by[g][round_])
            if len(answer)==min(count,len(keys)):break
        round_+=1
    return sorted(answer)


def prepare(args):
    import numpy as np,pandas as pd
    from origin_features import make_example,IneligibleOrigin,ALL_NAMES,BASE_NAMES
    started=time.monotonic()
    if shutil.disk_usage(args.out).free < 1024**3:raise ValueError('At least 1 GiB of free output storage required')
    keys,contract,pre=check_repo(args.repo)
    selected=choose_plays(keys,args.plays);needed=set(selected)
    run=args.out/args.label;run.mkdir(parents=True,exist_ok=True)
    plan={'source_signature':source_signature(),'seed':SEED,'plays':selected,'offsets':list(OFFSETS),
          'feature_names':ALL_NAMES,'control_feature_count':len(BASE_NAMES),'new_feature_count':len(ALL_NAMES)-len(BASE_NAMES),
          'scope':'subset of the frozen inner_1 TRAIN partition only','outer_validation_transformed_or_scored':False}
    ppath=run/'plan.json'
    if ppath.exists() and json.loads(ppath.read_text())!=json.loads(json.dumps(plan)):
        raise ValueError('Existing preparation plan differs; preserve it; after review, choose a separate --out directory.')
    atomic_json(ppath,plan);atomic_json(args.out/'preflight.json',pre)
    receipts={};counts={str(o):{'eligible_plays':0,'ineligible_plays':0,'rows':0,'prethrow_rows':0} for o in OFFSETS}
    expected={r['path']:r for r in contract['raw_files']}
    # Completed per-play artifacts survive timeout; reuse only exact matching hashes.
    for game,play in selected:
        receipt=run/'plays'/f'{game}_{play}.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['source_signature']!=plan['source_signature']:raise ValueError('Per-play source drift')
            for item in r['files']:
                p=regular(run/item['path'])
                if digest(p)!=item['sha256']:raise ValueError('Per-play checkpoint hash mismatch')
            receipts[(game,play)]=r;needed.discard((game,play))
    reused_plays=len(receipts)
    log('prepare_start',selected_plays=len(selected),reused_plays=reused_plays,offsets=OFFSETS)
    for week in range(1,19):
        if not needed:break
        ip=args.repo/f'data/raw/train/input_2023_w{week:02d}.csv'
        op=args.repo/f'data/raw/train/output_2023_w{week:02d}.csv'
        # SHA verify each weekly file actually scanned; don't trust mtime/size alone.
        for p in (ip,op):
            key=p.relative_to(args.repo).as_posix();meta=expected.get(key)
            if not meta or digest(regular(p))!=meta['sha256']:raise ValueError(f'Raw-file hash mismatch: {key}')
        raw=pd.read_csv(ip);labs=pd.read_csv(op)
        pairs=pd.MultiIndex.from_frame(raw[['game_id','play_id']])
        relevant=raw.loc[pairs.isin(pd.MultiIndex.from_tuples(sorted(needed)))].copy()
        del raw,pairs
        groups=labs.groupby(['game_id','play_id'],sort=False)
        for key,g in relevant.groupby(['game_id','play_id'],sort=True):
            key=tuple(map(int,key))
            if key not in needed:continue
            if key not in groups.indices:raise ValueError(f'No labels for selected training play {key}')
            y=groups.get_group(key)
            result={'source_signature':plan['source_signature'],'files':[],'ineligible':[],'offsets':{}}
            for offset in OFFSETS:
                try:ex=make_example(g,y,offset)
                except IneligibleOrigin as exc:
                    if offset==0:raise
                    result['ineligible'].append({'offset':offset,'reason':str(exc)});continue
                path=run/'plays'/f'{key[0]}_{key[1]}_o{offset}.npz'
                atomic_npz(path,**{k:(v.astype(np.float32) if k in ['X','y','cv'] else v)
                                  for k,v in ex.items() if isinstance(v,np.ndarray)})
                result['files'].append({'offset':offset,'path':path.relative_to(run).as_posix(),'sha256':digest(path)})
                result['offsets'][str(offset)]={'rows':len(ex['y']),'prethrow_rows':int(ex['prethrow'].sum()),
                                              'original_rows':int(ex['original_rows'])}
            atomic_json(run/'plays'/f'{key[0]}_{key[1]}.json',result)
            receipts[key]=result;needed.remove(key)
            if len(receipts)%8==0 or not needed:
                log('play_checkpoint',completed=len(receipts),total=len(selected),elapsed_seconds=round(time.monotonic()-started,2))
        del labs,relevant,groups
    if needed:raise ValueError(f'{len(needed)} selected training plays were not found in raw files')
    files=[]
    for key,r in sorted(receipts.items()):
        files.extend(r['files'])
        for o,m in r['offsets'].items():
            for fld in ('rows','prethrow_rows'):counts[o][fld]+=m[fld]
            counts[o]['eligible_plays']+=1
        for x in r['ineligible']:counts[str(x['offset'])]['ineligible_plays']+=1
    manifest={'source_signature':plan['source_signature'],'files':files}
    atomic_json(run/'dataset_manifest.json',manifest)
    summary={'status':'feature_dataset_ready','label':args.label,'selected_training_plays':len(selected),
             'selected_training_games':len(set(k[0] for k in selected)),'offsets':counts,
             'control_features':len(BASE_NAMES),'arrival_features':len(ALL_NAMES)-len(BASE_NAMES),
             'total_features':len(ALL_NAMES),'scientific_fits':0,'outer_validation_used':False,
             'raw_data_modified':False,'feature_research':'open','elapsed_seconds':round(time.monotonic()-started,3),
             'reused_play_checkpoints':reused_plays,
             'dataset_manifest_sha256':digest(run/'dataset_manifest.json')}
    atomic_json(run/'preparation_summary.json',summary);log('prepare_complete',**summary)


def load_dataset(run,source_check=True):
    import numpy as np
    manifest=json.loads((run/'dataset_manifest.json').read_text())
    if source_check and manifest['source_signature']!=source_signature():raise ValueError('Prepared features use a different source/environment')
    data={}
    decoded_bytes=0
    for item in manifest['files']:
        path=regular(run/item['path'])
        if digest(path)!=item['sha256']:raise ValueError('Prepared feature checkpoint changed')
        with zipfile.ZipFile(path) as archive:
            decoded_bytes += sum(info.file_size for info in archive.infolist())
        if decoded_bytes > 512*1024**2:raise ValueError('Prepared arrays exceed the 512 MiB loading budget; do not drop evaluation rows to bypass it')
        with np.load(path,allow_pickle=False) as z:
            for k in z.files:data.setdefault(k,[]).append(z[k])
    return {k:np.concatenate(v,axis=0) for k,v in data.items()}


def screen_arrays(data,run,*,repeats=2000,max_rows=10000,signature=None):
    """Three predeclared arms on expanding, date-disjoint training-side folds."""
    import numpy as np
    from origin_features import BASE_NAMES,ALL_NAMES,fit_ridge,predict_ridge,chronological_folds,paired_bootstrap,rmse
    original=data['offset']==0
    folds=chronological_folds(data['keys'][original,0])
    # Fixed budget per arm: C changes views, not labeled training-row count.
    arms=('control','arrival','arrival_origin')
    screen_dir=run/'screen';screen_dir.mkdir(parents=True,exist_ok=True)
    protocol={'signature':signature,'arms':list(arms),'seed':SEED,'max_rows_per_arm':max_rows,
              'penalty':0.01,'bootstrap':repeats,'comparison_1':'arrival minus control',
              'comparison_2':'arrival_origin minus arrival','fixed_evaluation':'all original requested rows in each validation game',
              'split_scope':'chronological partitions within the frozen inner_1 training games',
              'folds':[{'fold':f['fold'],'train_games':f['train_games'].tolist(),
                        'validation_games':f['validation_games'].tolist()} for f in folds]}
    p=screen_dir/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Screen protocol drift; do not overwrite earlier scientific evidence')
    atomic_json(p,protocol)
    output=[];comparisons=[];all_y=[];all_p={a:[] for a in arms};all_g=[];slices=[];new_fits=0;reused=0
    for fold in folds:
        fi=fold['fold'];train=np.isin(data['keys'][:,0],fold['train_games']);valid=original&np.isin(data['keys'][:,0],fold['validation_games'])
        oi=np.flatnonzero(train&original);ai=np.flatnonzero(train&~original);vi=np.flatnonzero(valid)
        n=min(max_rows,len(oi),2*len(ai))
        if n<32 or len(vi)==0:raise ValueError('Insufficient training or original evaluation rows')
        rng=np.random.default_rng(SEED+fi);oi=rng.permutation(oi);ai=rng.permutation(ai)
        train_idx={'control':oi[:n],'arrival':oi[:n],
                   'arrival_origin':np.r_[oi[:n-n//2],ai[:n//2]]}
        eval_y=data['y'][vi].astype(float);games=data['keys'][vi,0];preds={}
        for arm in arms:
            path=screen_dir/f'fold_{fi}_{arm}.npz';rec=screen_dir/f'fold_{fi}_{arm}.json'
            if path.exists() and rec.exists():
                receipt=json.loads(rec.read_text())
                if receipt['signature']!=signature or digest(path)!=receipt['sha256']:raise ValueError('Model/prediction checkpoint differs')
                with np.load(path,allow_pickle=False) as z:
                    if not np.array_equal(z['evaluation_keys'],data['keys'][vi]):raise ValueError('Resumed evaluation keys differ')
                    pred=z['pred'];preds[arm]=pred
                    if pred.shape != eval_y.shape or not np.isfinite(pred).all():raise ValueError('Invalid resumed predictions')
                    if abs(rmse(eval_y,pred)-receipt['rmse'])>1e-10:raise ValueError('Resumed metric mismatch')
                reused+=1;log('reuse_fit',fold=fi,arm=arm)
            else:
                idx=train_idx[arm];width=len(BASE_NAMES) if arm=='control' else len(ALL_NAMES)
                t=time.monotonic();model=fit_ridge(data['X'][idx,:width],data['y'][idx],penalty=.01)
                pred=predict_ridge(model,data['X'][vi,:width]);preds[arm]=pred
                if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction; no row filtering allowed')
                atomic_npz(path,**{f'model_{k}':v for k,v in model.items()},pred=pred,
                           truth=eval_y,evaluation_keys=data['keys'][vi],train_indices=idx)
                # Independent numerical read-back, not merely checking file existence.
                with np.load(path,allow_pickle=False) as z:
                    restored={k.removeprefix('model_'):z[k] for k in z.files if k.startswith('model_')}
                    replay=predict_ridge(restored,data['X'][vi,:width])
                    if not np.array_equal(pred,replay):raise ValueError('Exact fitted-model forward replay failed')
                receipt={'signature':signature,'fold':fi,'arm':arm,'sha256':digest(path),'train_rows':n,
                         'evaluation_rows':len(vi),'train_games':len(fold['train_games']),
                         'evaluation_games':len(fold['validation_games']),
                         'rmse':rmse(eval_y,pred),'input_columns':width,'retained_columns':int(model['keep'].sum()),
                         'elapsed_seconds':round(time.monotonic()-t,3),'forward_replay_exact':True}
                atomic_json(rec,receipt);new_fits+=1
                log('fit_complete',**receipt)
            output.append({k:v for k,v in receipt.items() if k not in ('sha256','signature')})
            all_p[arm].append(pred)
            for name,mask in [('first_second',data['keys'][vi,3]<=10),('after_first_second',data['keys'][vi,3]>10)]:
                if mask.any():slices.append({'fold':fi,'arm':arm,'slice':name,'rows':int(mask.sum()),'rmse':rmse(eval_y[mask],pred[mask])})
        for control,treatment in [('control','arrival'),('arrival','arrival_origin')]:
            ci=paired_bootstrap(eval_y,preds[control],preds[treatment],games,SEED+fi,repeats)
            comparisons.append({'fold':fi,'comparison':f'{treatment} minus {control}',**ci})
        all_y.append(eval_y);all_g.append(games)
        atomic_json(screen_dir/'progress.json',{'completed_folds':fi,'fits':output,'comparisons':comparisons,'slices':slices})
    yy=np.concatenate(all_y);gg=np.concatenate(all_g);pp={a:np.concatenate(v) for a,v in all_p.items()}
    # Evaluations occur in disjoint date blocks. Never average fold RMSEs as pooled RMSE.
    pooled={a:rmse(yy,p) for a,p in pp.items()};decisions=[]
    for control,treatment in [('control','arrival'),('arrival','arrival_origin')]:
        ci=paired_bootstrap(yy,pp[control],pp[treatment],gg,SEED,repeats)
        gain=(1-pooled[treatment]/pooled[control]) if pooled[control]>0 else 0.0
        fold_deltas=[r['delta_rmse'] for r in comparisons if r['comparison']==f'{treatment} minus {control}']
        pass_=gain>=.01 and ci['ci_high']<0 and sum(d<0 for d in fold_deltas)>=2
        decisions.append({'comparison':f'{treatment} minus {control}','relative_gain':gain,
                          'improving_folds':sum(d<0 for d in fold_deltas),'diagnostic_gate_passed':bool(pass_),**ci,
                          'next':'candidate for matched established-neural-model integration' if pass_ else 'do not scale this ridge interface; inspect support/underfitting before generalizing the negative result'})
    result={'status':'diagnostic_screen_complete','new_fits':new_fits,'reused_fits':reused,'total_fits':9,
            'metric':'sqrt(sum((dx)^2+(dy)^2)/(2*N))','pooled_rmse':pooled,'decisions':decisions,
            'fits':output,'comparisons':comparisons,'slices':slices,'evaluation_rows':len(yy),'evaluation_games':len(np.unique(gg)),
            'outer_validation_used':False,'kaggle_score':None,'existing_model_retrained':False,
            'scope':'small training-side ridge diagnostic; NOT a score for the saved velocity model or Kaggle',
            'feature_research':'open','comparison_1_changes_feature_width':True,
            'comparison_2_same_feature_width_and_training_row_budget':True}
    atomic_json(run/'screen_summary.json',result);log('screen_complete',pooled_rmse=pooled,new_fits=new_fits,reused_fits=reused)
    return result


def screen(args):
    run=args.out/args.label
    summary=json.loads((run/'preparation_summary.json').read_text())
    if summary['selected_training_plays']<128:raise ValueError('Screen requires the research dataset, not the 32-play engineering smoke')
    data=load_dataset(run)
    signature=hashlib.sha256((digest(run/'dataset_manifest.json')+source_signature()+'screen-v1').encode()).hexdigest()
    screen_arrays(data,run,signature=signature)


def export_report(args):
    """Only aggregate JSON: never private rows, original keys, caches or weights."""
    args.out.mkdir(parents=True,exist_ok=True)
    selected=[]
    for label in ['smoke','research']:
        for name in ['preparation_summary.json','screen_summary.json','milestone_review.json']:
            p=args.out/label/name
            if p.is_file():selected.append(p)
    for name in ['preflight.json','last_command.json','local_tests.json']:
        p=args.out/name
        if p.is_file():selected.append(p)
    path=args.out/'nfl_feature_round1_report.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected:z.write(p,p.relative_to(args.out))
        z.writestr('CONTENTS.txt','Aggregate evidence only. No raw tracking, per-row errors, cache, models or credentials.\n')
    log('report_exported',path=path,files=len(selected))


def worker(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=='preflight':
        _,_,r=check_repo(args.repo);atomic_json(args.out/'preflight.json',r);log('preflight_complete',**r)
    elif args.command=='prepare':prepare(args)
    elif args.command=='screen':screen(args)
    elif args.command=='report':export_report(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','prepare','screen','report'])
    p.add_argument('--repo',type=Path,default=DEFAULT_REPO);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--plays',type=int,default=256);p.add_argument('--label',choices=['smoke','research'],default='research')
    p.add_argument('--seconds',type=int,default=600);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not 1<=args.plays<=512 or not 1<=args.seconds<=900:p.error('Budget: 1..512 plays and 1..900 seconds')
    if args.worker:
        worker(args);return
    args.out.mkdir(parents=True,exist_ok=True)
    if args.out.resolve().is_relative_to(args.repo.resolve()):
        p.error('Keep this experiment output outside the canonical repository.')
    import fcntl
    lock=(args.out/'.round.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:p.error('Another Round 1 command is active; do not run notebook stages concurrently.')
    # A process-level watchdog also caps a stalled BLAS kernel or file operation.
    env=os.environ.copy();env.update({k:'2' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    env['PYTHONUNBUFFERED']='1'
    cmd=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker']
    started=time.monotonic();state='failed';rc=1
    proc=subprocess.Popen(cmd,env=env,start_new_session=True)
    try:
        while proc.poll() is None:
            remaining=args.seconds-(time.monotonic()-started)
            if remaining<=0:raise subprocess.TimeoutExpired(cmd,args.seconds)
            try:proc.wait(timeout=min(15,remaining))
            except subprocess.TimeoutExpired:log('heartbeat',command=args.command,elapsed_seconds=round(time.monotonic()-started,1),limit_seconds=args.seconds)
        rc=proc.returncode;state='completed' if rc==0 else 'failed'
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        state='stopped_budget' if isinstance(exc,subprocess.TimeoutExpired) else 'interrupted'
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        rc=124
    finally:
        atomic_json(args.out/'last_command.json',{'command':args.command,'status':state,'exit_code':rc,
                    'label':args.label,'hard_limit_seconds':args.seconds,'elapsed_seconds':round(time.monotonic()-started,3),
                    'cloud_resource_changes':0,'github_writes':0,'partial_checkpoints_retained':True})
    if rc:raise SystemExit(rc)

if __name__=='__main__':main()
