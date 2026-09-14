"""Bounded no-fit training coverage and representation audit. All stages offline."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib,json,os,re,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
from audit_io import atomic,atomic_json,digest,safe_file,read_json

KIT=Path(__file__).resolve().parent;HOME=Path('/home/sagemaker-user')
LIMITS={'preflight':90,'coverage':180,'smoke':120,'features':180,'audit':180,'replay':240,'report':60}
REPORTS=('preflight.json','coverage_summary.json','smoke.json','feature_summary.json','training_audit.json','replay.json','last_command.json','last_error.json')

def paths(a):
    allp=[a.out,a.parent,a.parent_kit,a.tensors,a.repo,KIT]
    for p in allp:
        if not p.is_absolute() or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('Absolute non-symlink paths required')
    for p in allp[1:]:
        if a.out==p or a.out.is_relative_to(p) or p.is_relative_to(a.out):raise ValueError('Output must be separate from parents and source')

def report(out):
    out.mkdir(parents=True,exist_ok=True);dest=out/'nfl_feature_round11_report.zip'
    def write(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for n in REPORTS:
                if (out/n).is_file():
                    data=safe_file(out,n).read_bytes()
                    if len(data)>2*1024**2:raise ValueError('Aggregate too large')
                    json.loads(data);z.writestr(n,data)
            z.writestr('CONTENTS.txt','Aggregate results only. No individual keys, predictions, targets, raw trajectories, model weights or candidate tensors. No new validation result.\n')
    atomic(dest,write);print('Report: '+str(dest),flush=True);return dest

def command(a):
    if a.stage not in ('audit','replay'):
        cmd=[sys.executable,str(KIT/'worker.py')]
    else:
        c=read_json(KIT/'input_contract.json');raw=safe_file(a.repo,'scripts/motion_supervision.py.lock').read_bytes()
        if hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()!=c['neural_lock_git_blob']:raise ValueError('Existing neural lock changed')
        runtime=a.out/'runtime';runtime.mkdir(exist_ok=True)
        for p in KIT.glob('*.py'):
            dest=runtime/p.name;data=p.read_bytes()
            if dest.exists():
                if safe_file(runtime,p.name).read_bytes()!=data:raise ValueError('Runtime source drift')
            else:atomic(dest,lambda f,data=data:f.write(data))
        lp=runtime/'worker.py.lock'
        if lp.exists():
            if safe_file(runtime,lp.name).read_bytes()!=raw:raise ValueError('Runtime lock drift')
        else:atomic(lp,lambda f:f.write(raw))
        uv=shutil.which('uv')
        if not uv and (Path.home()/'.local/bin/uv').is_file():uv=str(Path.home()/'.local/bin/uv')
        if not uv:raise FileNotFoundError('Existing uv missing; no installation is performed')
        cmd=[uv,'run','--frozen','--offline','--no-project','--no-python-downloads','--python',sys.executable,str(runtime/'worker.py')]
    for text in [a.stage]:cmd.append(text)
    for n,v in [('kit',KIT),('out',a.out),('parent',a.parent),('parent-kit',a.parent_kit),('tensors',a.tensors),('repo',a.repo)]:cmd+=['--'+n,str(v)]
    return cmd

def terminate(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=list(LIMITS));p.add_argument('--seconds',type=int)
    p.add_argument('--out',type=Path,default=HOME/'nfl-feature-round11-results')
    p.add_argument('--parent',type=Path,default=HOME/'nfl-feature-round10-results')
    p.add_argument('--parent-kit',type=Path,default=HOME/'nfl_feature_round10')
    p.add_argument('--tensors',type=Path,default=HOME/'nfl-feature-round8-results')
    p.add_argument('--repo',type=Path,default=HOME/'nfl-player-trajectory')
    a=p.parse_args();cap=a.seconds if a.seconds is not None else LIMITS[a.stage]
    if not 1<=cap<=LIMITS[a.stage]:p.error('Hard limit cannot be increased')
    paths(a);a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with ExitStack() as stack:
        own=a.out/'.run.lock'
        if own.is_symlink():raise ValueError('Symlink lock rejected')
        h=stack.enter_context(own.open('a'))
        try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Round 11 is already running')
        if a.stage=='report':report(a.out);return
        for root in (a.parent,a.tensors):
            h=stack.enter_context(safe_file(root,'.run.lock').open('r'))
            try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise SystemExit('A parent stage is running; stop concurrent work')
        start=time.monotonic();code=0;log=None
        try:
            cmd=command(a);d=a.out/'logs';d.mkdir(exist_ok=True)
            log=d/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+a.stage+'.log')
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            with log.open('w') as f:
                child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                pos=0;last=0
                try:
                    while child.poll() is None:
                        elapsed=time.monotonic()-start
                        if elapsed>=cap:terminate(child);code=124;break
                        with log.open() as src:src.seek(pos);text=src.read();pos=src.tell()
                        if text:print(text,end='',flush=True)
                        if elapsed-last>=15:
                            print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':'heartbeat','stage':a.stage,'elapsed_seconds':round(elapsed,1),'hard_limit_seconds':cap}),flush=True);last=elapsed
                        time.sleep(.2)
                except KeyboardInterrupt:terminate(child);code=130
                finally:
                    with log.open() as src:src.seek(pos);print(src.read(),end='',flush=True)
                if not code:code=child.returncode
        except Exception as e:
            code=2;atomic_json(a.out/'last_error.json',{'stage':a.stage,'type':type(e).__name__,'message':str(e)})
        r={'status':'completed' if code==0 else 'stopped','stage':a.stage,'exit_code':code,
           'elapsed_seconds':round(time.monotonic()-start,3),'hard_limit_seconds':cap,'new_model_fits':0,'new_optimizer_steps':0,
           'github_writes':0,'aws_resource_changes':0,'kaggle_submissions':0,'utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(a.out/'last_command.json',r)
        if code:
            if log:
                text=re.sub(r'https?://\S+','<url>',log.read_text(errors='replace')[-5000:])
                text=re.sub(r'(?i)((?:secret|token|password|access_key)\w*\s*[=:]\s*)\S+',r'\1<redacted>',text)
                atomic_json(a.out/'last_error.json',{**r,'diagnostic_tail':text})
            report(a.out)
        print(json.dumps(r),flush=True)
        if code:raise SystemExit(code)
if __name__=='__main__':main()
