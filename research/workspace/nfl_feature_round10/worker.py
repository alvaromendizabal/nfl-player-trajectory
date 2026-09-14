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
"""Pinned CPU worker; invoked through the bounded launcher."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from audit_io import read_json,seal_json,safe_file,digest
from data_io import preflight,prepare,code_signature
from parent_audit import assert_unchanged


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','smoke','prepare','runtime','profile','train','evaluate','replay'])
    for n in ('kit','out','parent','parent-kit','audit','repo'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arm',choices=['cartesian','goal'],default='cartesian');a=p.parse_args()
    if a.stage=='preflight':r=preflight(a)
    elif a.stage in ('smoke','prepare'):r=prepare(a,smoke=a.stage=='smoke')
    else:
        from study import environment,profile,train_arm,evaluate
        from grouped_model import setup
        import numpy as np,torch
        setup();env=environment()
        if sys.version_info[:2]!=(3,11) or np.__version__!='2.4.6' or torch.__version__!='2.8.0+cpu':
            raise ValueError('Runtime differs from the existing pinned CPU lock; no automatic install')
        if a.stage=='runtime':
            subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.kit/'tests/model'),'-v'],
                           cwd=a.kit,check=True,timeout=100)
            r={'status':'grouped_runtime_ready','environment':env,'source':code_signature(a.kit),
               'synthetic_model_tests_passed':True,'private_data_read':False}
            seal_json(a.out/'runtime_receipt.json',r)
        else:
            runtime=read_json(safe_file(a.out,'runtime_receipt.json'))
            if runtime['environment']!=env or runtime['source']!=code_signature(a.kit):
                raise ValueError('Current model tests and runtime verification required')
            protected=read_json(safe_file(a.out,'parent_protection.json'))
            protected.update(read_json(safe_file(a.out,'tensor_protection.json')))
            assert_unchanged(protected)
            if a.stage=='profile':r=profile(a)
            elif a.stage=='train':r=train_arm(a,a.arm)
            else:r=evaluate(a,replay=a.stage=='replay')
            assert_unchanged(protected)
    print(json.dumps({'event':'stage_result','stage':a.stage,**r},allow_nan=False),flush=True)
    if r['status']=='grouped_profile_budget_stop':raise SystemExit(3)

if __name__=='__main__':main()
