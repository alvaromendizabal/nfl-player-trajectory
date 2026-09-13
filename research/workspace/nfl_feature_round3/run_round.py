"""Manual, bounded Round 3. No AWS APIs, Git writes, Kaggle downloads or submissions."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

KIT=Path(__file__).resolve().parent
HOME=Path('/home/sagemaker-user')
LIMITS={'preflight':120,'runtime':180,'index':180,'smoke':180,'prepare':600,'fit':480,'replay':180,'report':60}


def log(event,**kw):print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw},allow_nan=False),flush=True)


def context(args):
    import numpy as np
    import pandas as pd
    from parent_support import digest,load_parent,hash_json,seal_json
    c=json.loads((KIT/'input_contract.json').read_text())
    env={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}
    if env!=c['parent_environment']:raise ValueError('Use the preserved NFL Python 3.11 environment; do not reinstall: '+str(env))
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=args.repo,check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    if head!=c['repo_commit']:raise ValueError('Repository source changed; do not reset it automatically')
    for name,sha in c['round2_summaries'].items():
        p=args.round2/name
        if p.is_symlink() or digest(p)!=sha:raise ValueError('Round 2 result differs from the reviewed report')
    code=['run_round.py','data_pipeline.py','pair_features.py','tree_worker.py','origin_features.py','parent_support.py','input_contract.json']
    sig=hash_json({'code':{n:digest(KIT/n) for n in code},'environment':env})
    parent=load_parent(args.parent,c)
    seal_json(args.out/'plan.json',{'signature':sig,'repo':head,'source_hashes':{n:digest(KIT/n) for n in code},
                                  'max_selected_plays':1024,'max_first_fold_coordinate_models':8,
                                  'maximum_coordinate_models_all_folds':24,'augmentation':False})
    return parent,c,sig


def runtime_command(args,action):
    from parent_support import _atomic,digest
    c=json.loads((KIT/'input_contract.json').read_text())
    source=args.repo/'scripts/fit_final.py.lock'
    if source.is_symlink():raise ValueError('Unsafe lock path')
    raw=source.read_bytes();blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if blob!=c['model_lock_git_blob']:raise ValueError('Existing tree dependency lock is not the reviewed Git blob')
    folder=args.out/'runtime';folder.mkdir(exist_ok=True)
    for n in ('tree_worker.py','parent_support.py','origin_features.py','pair_features.py'):
        dest=folder/n;payload=(KIT/n).read_bytes()
        if dest.exists() and dest.read_bytes()!=payload:raise ValueError('Runtime code changed; preserve old results')
        if not dest.exists():_atomic(dest,lambda f,p=payload:f.write(p))
    dest=folder/'tree_worker.py.lock'
    if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Runtime lock drift')
    if not dest.exists():_atomic(dest,lambda f:f.write(raw))
    uv=shutil.which('uv')
    if uv is None:
        candidate=Path.home()/'.local/bin/uv'
        if candidate.is_file():uv=str(candidate)
    if uv is None:raise FileNotFoundError('Existing uv executable not found. Do not reinstall the project; return the error report.')
    # Use the existing interpreter and its prior dependency lock in an isolated
    # uv script cache. Online installation is available ONLY on explicit runtime setup.
    cmd=[uv,'run','--frozen','--no-project','--no-python-downloads','--python',sys.executable]
    if not (args.command=='runtime' and args.online):cmd+=['--offline']
    cmd+=[str(folder/'tree_worker.py'),action,'--out',str(args.out),'--fold',str(args.fold)]
    return cmd


def preflight(args,parent,c,sig):
    from parent_support import atomic_json
    free=shutil.disk_usage(args.out).free/1024**3
    if free<3:raise ValueError('At least 3 GiB free disk is required for restartable checkpoints')
    result={'status':'round3_preflight_passed','signature':sig,'parent_models_replayed':6,'parent_refits':0,
            'round2_summaries_verified':True,'repo_head':c['repo_commit'],'free_gib':round(free,2),
            'feature_research':'open','github_updated':False,'new_kaggle_score':None}
    atomic_json(args.out/'preflight.json',result);log('preflight_complete',**result)


def model_input(args,sig):
    from data_pipeline import assemble
    from parent_support import atomic_npz,atomic_json,digest
    data,plan,datahash=assemble(args.out,sig)
    file=args.out/'model_input.npz';receipt=args.out/'model_input.json'
    if receipt.exists():
        d=json.loads(receipt.read_text())
        if d['signature']!=sig or d['data_signature']!=datahash or digest(file)!=d['sha256']:raise ValueError('Prepared model data drift')
        return
    if file.exists():raise ValueError('Orphan model input; preserve and return report rather than overwrite')
    atomic_npz(file,**data)
    atomic_json(receipt,{'signature':sig,'data_signature':datahash,'sha256':digest(file),
                        'folds':plan['folds'],'plays':len(plan['plays']),'rows':len(data['keys']),
                        'array_bytes':sum(a.nbytes for a in data.values())})


def export_report(out):
    from parent_support import _atomic
    names=['preflight.json','last_command.json','last_error.json','runtime.json','smoke.json','preparation.json','precision_audit.json','recovery.json']
    for f in (1,2,3):names += [f'fold_{f}/summary.json',f'fold_{f}/replay.json']
    dest=out/'nfl_feature_round3_report.zip'
    def write(stream):
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for n in names:
                p=out/n
                if p.is_file() and not p.is_symlink():z.write(p,n)
            z.writestr('CONTENTS.txt','Aggregates and status only. No raw tracking, target coordinates, per-row keys, feature arrays, pickled models or credentials.\n')
    _atomic(dest,write);log('report_created',path=str(dest))


def worker(args):
    if args.command=='report':export_report(args.out);return
    parent,c,sig=context(args)
    if args.command=='preflight':preflight(args,parent,c,sig)
    elif args.command=='runtime':subprocess.run(runtime_command(args,'runtime'),check=True)
    elif args.command=='index':
        from data_pipeline import make_plan
        p=make_plan(args.repo,args.out,parent,c,sig,log);log('selection_frozen',plays=len(p['plays']))
    elif args.command in ('smoke','prepare'):
        from data_pipeline import prepare
        result=prepare(args.repo,args.out,parent,c,sig,smoke=args.command=='smoke',event=log)
        log('features_complete',**result)
    elif args.command in ('fit','replay'):
        if not (args.out/'runtime.json').exists():raise ValueError('Run runtime setup/self-test before scientific fitting')
        model_input(args,sig)
        subprocess.run(runtime_command(args,args.command),check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=list(LIMITS))
    p.add_argument('--repo',type=Path,default=HOME/'nfl-player-trajectory')
    p.add_argument('--parent',type=Path,default=HOME/'nfl-feature-round1-results')
    p.add_argument('--round2',type=Path,default=HOME/'nfl-feature-round2-results')
    p.add_argument('--out',type=Path,default=HOME/'nfl-feature-round3-results')
    p.add_argument('--fold',type=int,choices=[1,2,3],default=1)
    p.add_argument('--seconds',type=int);p.add_argument('--online',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    args.seconds=args.seconds or LIMITS[args.command]
    if not 1<=args.seconds<=LIMITS[args.command]:p.error('Time cap exceeds the frozen stage limit')
    if args.online and args.command!='runtime':p.error('Only explicit runtime setup may use network package retrieval')
    for a in [args.out,*args.out.parents]:
        if a.is_symlink():p.error('Symlink output paths are not accepted')
    for protected in (args.repo,args.parent,args.round2,KIT):
        a,b=args.out.resolve(),protected.resolve()
        if a==b or a.is_relative_to(b) or b.is_relative_to(a):p.error('Outputs must be separate from code/data and old results')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    import fcntl
    from parent_support import atomic_json
    with (args.out/'.run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another Round 3 process is active; do not run concurrently')
        started=time.monotonic();env=os.environ.copy()
        env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
        logdir=args.out/'logs';logdir.mkdir(exist_ok=True)
        name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+args.command+'.log'
        logpath=logdir/name
        with logpath.open('w') as logf:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],stdout=logf,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            pos=0;last=0;code=None
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    if elapsed>=args.seconds:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        code=124;break
                    with logpath.open() as read:
                        read.seek(pos);text=read.read();pos=read.tell()
                        if text:print(text,end='',flush=True)
                    if elapsed-last>=15:log('heartbeat',stage=args.command,elapsed_seconds=round(elapsed,1),hard_cap=args.seconds);last=elapsed
                    time.sleep(.25)
            except KeyboardInterrupt:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=130
            finally:
                with logpath.open() as read:read.seek(pos);print(read.read(),end='',flush=True)
                code=child.returncode if code is None else code
                receipt={'utc':datetime.now(timezone.utc).isoformat(),'command':args.command,'fold':args.fold,'exit_code':code,'status':'completed' if code==0 else 'stopped',
                         'elapsed_seconds':round(time.monotonic()-started,3),'hard_limit_seconds':args.seconds,
                         'github_writes':0,'aws_resource_changes':0,'kaggle_submissions':0,'checkpoints_retained':True}
                atomic_json(args.out/'last_command.json',receipt)
                if code:
                    tail=logpath.read_text(errors='replace')[-6000:]
                    tail=re.sub(r'https?://\S+','<package-url-redacted>',tail)
                    tail=re.sub(r'(?i)((?:token|secret|password|access_key)[A-Za-z_]*\s*[=:]\s*)\S+',r'\1<redacted>',tail)
                    atomic_json(args.out/'last_error.json',{**receipt,'diagnostic_log':name,'diagnostic_tail':tail,'historical_error':True})
                log('stage_finished',**receipt)
        if code:raise SystemExit(code)
if __name__=='__main__':main()
