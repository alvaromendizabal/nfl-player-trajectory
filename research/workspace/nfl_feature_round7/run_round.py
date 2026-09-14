"""Manual, bounded Round 7: offline cached runtime, no Git/AWS/Kaggle writes."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

KIT=Path(__file__).resolve().parent
HOME=Path('/home/sagemaker-user')
LIMITS={'preflight':150,'smoke':180,'prepare':360,'fit':240,'replay':180,'summarize':120,'report':60}
REPORT_NAMES=('preflight.json','preflight_reuse.json','smoke.json','smoke_reuse.json',
              'preparation.json','preparation_reuse.json','summary.json','replay.json',
              'last_command.json','last_error.json','fold_2/summary.json','fold_2/replay.json',
              'fold_2/reuse.json','fold_3/summary.json','fold_3/replay.json','fold_3/reuse.json',
              'fold_3/futility.json','encoder_replay.json','fold_2/summary.receipt.json','fold_3/summary.receipt.json')


def atomic_json(path,value):
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    atomic_bytes(path,payload)


def atomic_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Refusing symlink destination')
    fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def log(event,**kw):
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':event,**kw},allow_nan=False),flush=True)


def runtime_command(args):
    c=json.loads((KIT/'input_contract.json').read_text())
    lock=args.repo/'scripts/fit_final.py.lock'
    if not lock.is_file() or lock.is_symlink():raise ValueError('Verified parent script lock missing')
    raw=lock.read_bytes()
    if hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=c['model_lock_git_blob']:
        raise ValueError('Existing tree script lock changed')
    dest=args.out/'runtime';dest.mkdir(parents=True,exist_ok=True)
    sources=[p for p in KIT.glob('*.py')]+[KIT/'input_contract.json']
    for source in sources:
        target=dest/source.name;payload=source.read_bytes()
        if target.is_symlink():raise ValueError('Symlink runtime file')
        if target.exists() and target.read_bytes()!=payload:raise ValueError('Frozen runtime source changed; preserve checkpoints')
        if not target.exists():atomic_bytes(target,payload)
    target=dest/'worker.py.lock'
    if target.exists() and target.read_bytes()!=raw:raise ValueError('Frozen runtime lock drift')
    if not target.exists():atomic_bytes(target,raw)
    uv=shutil.which('uv')
    if not uv:
        p=Path.home()/'.local/bin/uv'
        if p.is_file():uv=str(p)
    if not uv:raise FileNotFoundError('The existing uv executable was not found; return the report')
    return [uv,'run','--frozen','--offline','--no-project','--no-python-downloads',
            '--python',sys.executable,str(dest/'worker.py'),args.command,
            '--kit',str(KIT),'--round6',str(args.round6),'--round5',str(args.round5),'--round5-kit',str(args.round5_kit),'--round4',str(args.round4),'--round4-kit',str(args.round4_kit),
            '--round3',str(args.round3),'--round3-kit',str(args.round3_kit),
            '--repo',str(args.repo),'--out',str(args.out),'--fold',str(args.fold)]


def export_report(out):
    out=Path(out);dest=out/'nfl_feature_round7_report.zip'
    fd,tmp=tempfile.mkstemp(prefix='.report-',dir=out);os.close(fd)
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for name in REPORT_NAMES:
                p=out/name
                if p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)):
                    payload=p.read_bytes()
                    if len(payload)>2*1024**2:raise ValueError('Unexpectedly large aggregate report')
                    json.loads(payload);z.writestr(name,payload)
            z.writestr('CONTENTS.txt','Aggregate feature/metric reports and execution receipts only. No raw tracking, model blobs, per-row predictions, keys or targets.\n')
        if dest.is_symlink():raise ValueError('Refusing report symlink')
        os.replace(tmp,dest)
    finally:Path(tmp).unlink(missing_ok=True)
    log('report_created',path=str(dest))
    return dest


def safe_paths(args):
    for root in (args.out,args.round6,args.round5,args.round5_kit,args.round4,args.round4_kit,args.round3,args.round3_kit,args.repo,KIT):
        if any(p.is_symlink() for p in (root,*root.parents)):raise ValueError('Symlink workspace roots are not accepted')
    a=args.out.resolve()
    for protected in (args.round6,args.round5,args.round5_kit,args.round4,args.round4_kit,args.round3,args.round3_kit,args.repo,KIT):
        b=protected.resolve()
        if a==b or a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError('Round 7 results must be separate from prior results and source/data')


def stop_group(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=list(LIMITS));p.add_argument('--seconds',type=int)
    p.add_argument('--repo',type=Path,default=HOME/'nfl-player-trajectory')
    p.add_argument('--round3',type=Path,default=HOME/'nfl-feature-round3-results')
    p.add_argument('--round3-kit',type=Path,default=HOME/'nfl_feature_round3')
    p.add_argument('--round6',type=Path,default=HOME/'nfl-feature-round6-results')
    p.add_argument('--round5',type=Path,default=HOME/'nfl-feature-round5-results')
    p.add_argument('--round5-kit',type=Path,default=HOME/'nfl_feature_round5')
    p.add_argument('--round4',type=Path,default=HOME/'nfl-feature-round4-results')
    p.add_argument('--round4-kit',type=Path,default=HOME/'nfl_feature_round4')
    p.add_argument('--out',type=Path,default=HOME/'nfl-feature-round7-results')
    p.add_argument('--fold',type=int,choices=[2,3],default=2)
    args=p.parse_args();cap=args.seconds if args.seconds is not None else LIMITS[args.command]
    if not 1<=cap<=LIMITS[args.command]:p.error('Hard cap exceeds the frozen limit')
    safe_paths(args);args.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    code=0;started=time.monotonic();logpath=None
    with ExitStack() as stack:
        if (args.out/'.run.lock').is_symlink():raise ValueError('Symlink run lock')
        own=stack.enter_context((args.out/'.run.lock').open('a'))
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another Round 7 stage is running')
        if args.command!='report':
            # Lock existing parents without changing their bytes.
            for folder in (args.round3,args.round4,args.round5,args.round6):
                old_lock=folder/'.run.lock'
                if not old_lock.is_file() or old_lock.is_symlink():
                    raise SystemExit('Expected prior run lock missing; return the report')
                prior=stack.enter_context(old_lock.open('r'))
                try:fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:raise SystemExit('A prior round is active; do not run concurrently')
        try:
            if args.command=='report':
                export_report(args.out);return
            cmd=runtime_command(args)
            logdir=args.out/'logs';logdir.mkdir(exist_ok=True)
            name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+args.command+'.log'
            logpath=logdir/name
            env=os.environ.copy();env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                                            'NUMEXPR_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
            with logpath.open('w') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                pos=0;heartbeat=0
                try:
                    while child.poll() is None:
                        elapsed=time.monotonic()-started
                        if elapsed>=cap:stop_group(child);code=124;break
                        with logpath.open() as read:
                            read.seek(pos);text=read.read();pos=read.tell()
                        if text:print(text,end='',flush=True)
                        if elapsed-heartbeat>=15:
                            log('heartbeat',stage=args.command,elapsed_seconds=round(elapsed,1),hard_cap=cap);heartbeat=elapsed
                        time.sleep(.25)
                except KeyboardInterrupt:stop_group(child);code=130
                finally:
                    with logpath.open() as read:read.seek(pos);print(read.read(),end='',flush=True)
                if code==0:code=child.returncode
        except Exception as exc:
            code=2
            log('stage_error',error_type=type(exc).__name__,detail=str(exc))
            atomic_json(args.out/'last_error.json',{'command':args.command,'error_type':type(exc).__name__,'message':str(exc),'utc':datetime.now(timezone.utc).isoformat()})
        finally:
            if args.command!='report':
                receipt={'utc':datetime.now(timezone.utc).isoformat(),'command':args.command,'fold':args.fold,'exit_code':code,
                         'status':'completed' if code==0 else 'stopped','elapsed_seconds':round(time.monotonic()-started,3),
                         'hard_limit_seconds':cap,'github_writes':0,'aws_resource_changes':0,'kaggle_submissions':0,'checkpoints_retained':True}
                atomic_json(args.out/'last_command.json',receipt)
                if code and logpath is not None:
                    tail=logpath.read_text(errors='replace')[-6000:]
                    tail=re.sub(r'https?://\S+','<url-redacted>',tail)
                    tail=re.sub(r'(?i)((?:token|secret|password|access_key)[A-Za-z_]*\s*[=:]\s*)\S+',r'\1<redacted>',tail)
                    atomic_json(args.out/'last_error.json',{**receipt,'diagnostic_tail':tail,'log':logpath.name})
                log('stage_finished',**receipt)
                if code:export_report(args.out)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
