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
"""Stage worker: no training action exists."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from audit_io import seal_json,digest,safe_file
from core import preflight,coverage,features

def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','coverage','smoke','features','audit','replay'])
    for n in ('kit','out','parent','parent-kit','tensors','repo'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    if a.stage=='preflight':r=preflight(a)
    elif a.stage=='coverage':r=coverage(a)
    elif a.stage in ('smoke','features'):r=features(a,smoke=a.stage=='smoke')
    else:
        from diagnostics import run_audit
        if a.stage=='audit':
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.kit/'tests/model'),'-v'],cwd=a.kit,check=True,timeout=45)
        if a.stage=='replay':
            # Rebuild features and recheck key-inventory checkpoints without training.
            features(a,replay=True)
            coverage(a,replay=True)
        r=run_audit(a,replay=a.stage=='replay')
        if a.stage=='replay':
            r={'status':'training_review_replay_exact','new_optimizer_steps':0,'new_model_fits':0,
               'training_summary_sha256':digest(safe_file(a.out,'training_audit.json')),
               'feature_summary_sha256':digest(safe_file(a.out,'feature_summary.json')),
               'coverage_summary_sha256':digest(safe_file(a.out,'coverage_summary.json'))}
            seal_json(a.out/'replay.json',r)
    print(json.dumps({'event':'stage_result','stage':a.stage,**r},allow_nan=False),flush=True)
if __name__=='__main__':main()
