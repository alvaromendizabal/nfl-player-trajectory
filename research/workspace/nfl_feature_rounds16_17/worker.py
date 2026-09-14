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
"""Worker for the two predeclared independent feature rounds."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from audit_io import atomic_json,digest,safe_file,seal_json,read_json
import bridge

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','labels','smoke','prepare','runtime','profile','train','evaluate','replay'])
    p.add_argument('--round',dest='round_no',type=int,choices=[16,17],required=True)
    p.add_argument('--arm',choices=['mask','core','full'],default='mask')
    for n in ('kit','out','audit','audit-kit','parent','parent-kit','tensors','repo'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.stage=='preflight':r=bridge.preflight(a)
    elif a.stage=='labels':
        from readiness import labels
        r=labels(a)
    elif a.stage in ('smoke','prepare'):r=bridge.features(a,smoke=a.stage=='smoke')
    else:
        import study
        if a.stage=='runtime':
            from readiness import require_review_release
            require_review_release(a)
            v=bridge.context(a)
            if study.environment()!=v['contract']['environment']:raise ValueError('Pinned runtime mismatch')
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.kit/'tests/model'),'-v'],cwd=a.kit,timeout=90,check=True)
            r={'status':'family_runtime_ready','round':a.round_no,'signature':v['signature'],'environment':study.environment(),'scientific_models_fitted':0}
            atomic_json(a.out/'runtime_receipt.json',r)
        else:
            v=bridge.context(a);runtime=read_json(safe_file(a.out,'runtime_receipt.json'))
            if runtime['signature']!=v['signature'] or runtime['environment']!=study.environment():raise ValueError('Current runtime check required')
            if a.stage=='profile':r=study.profile(a)
            elif a.stage=='train':r=study.train_arm(a,a.arm)
            else:
                before=None
                if a.stage=='replay':
                    before={str(f):digest(f) for sub in ('features','predictions','models') for f in (a.out/sub).rglob('*') if f.is_file()}
                    bridge.features(a,replay=True)
                r=study.evaluate(a,replay=a.stage=='replay')
                if before:
                    for name,sha in before.items():
                        if digest(Path(name))!=sha:raise ValueError('Replay modified completed file')
                    r['protected_result_files_unchanged']=len(before)
                    atomic_json(a.out/'replay_integrity.json',r)
            bridge.unchanged(v)
    print(json.dumps({'event':'stage_result',**r},allow_nan=False),flush=True)
    if r.get('status') in ('family_profile_budget_stop','training_label_issues_require_review'):raise SystemExit(3)
if __name__=='__main__':main()
