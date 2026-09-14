"""Bounded manual matched feature study. No online, submit, or cloud-write action."""
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
import time
import zipfile
from audit_io import atomic,atomic_json,digest,safe_file,read_json

KIT=Path(__file__).resolve().parent
HOME=Path('/home/sagemaker-user')
LIMITS={'preflight':90,'smoke':120,'prepare':360,'runtime':180,'profile':180,'train':360,'evaluate':180,'replay':180,'report':60}
REPORT_NAMES=('preflight.json','smoke.json','preparation.json','runtime_receipt.json','profile.json','summary.json','replay.json','last_command.json','last_error.json','models/cartesian/complete.json','models/goal/complete.json','models/cartesian/reuse.json','models/goal/reuse.json')


def safe_paths(a):
    for p in (a.out,a.parent,a.parent_kit,a.audit,a.repo,KIT):
        if not p.is_absolute() or any(x.is_symlink() for x in (p,*p.parents)):
            raise ValueError('Absolute non-symlink paths required')
    for p in (a.parent,a.parent_kit,a.audit,a.repo,KIT):
        if a.out==p or a.out.is_relative_to(p) or p.is_relative_to(a.out):
            raise ValueError('Audit output must be outside parent/source directories')


def export_report(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'nfl_feature_round10_report.zip'
    def write(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for name in REPORT_NAMES:
                p=out/name
                if p.is_file():
                    p=safe_file(out,name);raw=p.read_bytes()
                    if len(raw)>2*1024**2:raise ValueError('Aggregate report too large')
                    json.loads(raw);z.writestr(name,raw)
            z.writestr('CONTENTS.txt','Aggregate receipts only. No row predictions, keys, targets, model weights or private tensors. This is not a Kaggle score.\n')
    atomic(path,write);print(f'Report: {path}',flush=True);return path


def command(a):
    if a.stage in ('preflight','smoke','prepare'):
        worker=KIT/'worker.py';cmd=[sys.executable,str(worker)]
    else:
        c=read_json(KIT/'input_contract.json')
        lock=safe_file(a.repo,'scripts/motion_supervision.py.lock');raw=lock.read_bytes()
        if hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=c['round8']['neural_lock_git_blob']:
            raise ValueError('CPU runtime lock changed')
        runtime=a.out/'runtime';runtime.mkdir(parents=True,exist_ok=True)
        sources=list(KIT.glob('*.py'))
        for source in sources:
            target=runtime/source.name;data=source.read_bytes()
            if target.exists():
                if safe_file(runtime,target.name).read_bytes()!=data:raise ValueError('Runtime source drift')
            else:atomic(target,lambda f,data=data:f.write(data))
        lp=runtime/'worker.py.lock'
        if lp.exists():
            if safe_file(runtime,lp.name).read_bytes()!=raw:raise ValueError('Runtime lock drift')
        else:atomic(lp,lambda f:f.write(raw))
        uv=shutil.which('uv')
        if not uv and (Path.home()/'.local/bin/uv').is_file():uv=str(Path.home()/'.local/bin/uv')
        if not uv:raise FileNotFoundError('Existing uv not found; no installation is performed')
        cmd=[uv,'run','--frozen','--offline','--no-project','--no-python-downloads','--python',sys.executable,str(runtime/'worker.py')]
    cmd+=[a.stage,'--arm',a.arm,'--audit',str(a.audit),'--parent',str(a.parent),'--parent-kit',str(a.parent_kit),'--repo',str(a.repo),'--kit',str(KIT),'--out',str(a.out)]
    return cmd


def terminate(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=list(LIMITS));p.add_argument('--seconds',type=int)
    p.add_argument('--parent',type=Path,default=HOME/'nfl-feature-round8-results')
    p.add_argument('--parent-kit',type=Path,default=HOME/'nfl_feature_round8')
    p.add_argument('--repo',type=Path,default=HOME/'nfl-player-trajectory')
    p.add_argument('--out',type=Path,default=HOME/'nfl-feature-round10-results')
    p.add_argument('--audit',type=Path,default=HOME/'nfl-feature-round9-results')
    p.add_argument('--arm',choices=['cartesian','goal'],default='cartesian')
    a=p.parse_args();cap=a.seconds if a.seconds is not None else LIMITS[a.stage]
    if not 1<=cap<=LIMITS[a.stage]:p.error('Cannot increase the hard limit')
    safe_paths(a);a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with ExitStack() as stack:
        lock=a.out/'.run.lock'
        if lock.is_symlink():raise ValueError('Symlink lock')
        own=stack.enter_context(lock.open('a'))
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another Round 10 stage is active')
        if a.stage=='report':export_report(a.out);return
        old=stack.enter_context(safe_file(a.parent,'.run.lock').open('r'))
        try:fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Round 8 is active; do not run concurrently')
        old9=stack.enter_context(safe_file(a.audit,'.run.lock').open('r'))
        try:fcntl.flock(old9,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Round 9 is active; do not run concurrently')
        started=time.monotonic();code=0;logpath=None
        try:
            cmd=command(a);logs=a.out/'logs';logs.mkdir(exist_ok=True)
            logpath=logs/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+a.stage+'.log')
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with logpath.open('w') as output:
                child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                pos=0;heartbeat=0
                try:
                    while child.poll() is None:
                        elapsed=time.monotonic()-started
                        if elapsed>=cap:terminate(child);code=124;break
                        with logpath.open() as stream:stream.seek(pos);text=stream.read();pos=stream.tell()
                        if text:print(text,end='',flush=True)
                        if elapsed-heartbeat>=15:
                            print(json.dumps({'event':'heartbeat','stage':a.stage,'elapsed_seconds':round(elapsed,1),'hard_limit':cap}),flush=True);heartbeat=elapsed
                        time.sleep(.2)
                except KeyboardInterrupt:terminate(child);code=130
                finally:
                    with logpath.open() as stream:stream.seek(pos);print(stream.read(),end='',flush=True)
                if code==0:code=child.returncode
        except Exception as e:
            code=2;atomic_json(a.out/'last_error.json',{'stage':a.stage,'type':type(e).__name__,'message':str(e)})
        receipt={'stage':a.stage,'status':'completed' if code==0 else 'stopped','exit_code':code,
                 'elapsed_seconds':round(time.monotonic()-started,3),'hard_limit_seconds':cap,
                 'arm':a.arm,'scientific_fits':('see model receipts' if a.stage=='train' else 0),'optimizer_steps':('see stage receipts' if a.stage in ('profile','train') else 0),'github_writes':0,'aws_resource_changes':0,
                 'kaggle_submissions':0,'utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(a.out/'last_command.json',receipt)
        if code:
            if logpath is not None:
                text=logpath.read_text(errors='replace')[-5000:]
                text=re.sub(r'https?://\S+','<url-redacted>',text)
                text=re.sub(r'(?i)((?:secret|token|password|access_key)\w*\s*[=:]\s*)\S+',r'\1<redacted>',text)
                atomic_json(a.out/'last_error.json',{**receipt,'diagnostic_tail':text})
            export_report(a.out)
        print(json.dumps(receipt),flush=True)
        if code:raise SystemExit(code)

if __name__=='__main__':main()
