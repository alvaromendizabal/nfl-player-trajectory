"""Local SYNTHETIC full-pipeline and fresh-process replay check, never NFL scoring."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fixtures import parent_fixture
from study import load_parent, screen, atomic_json, digest
from run_round import prepare_features, load_extra, export_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();root=args.root
    if args.replay:
        if not (root/'SYNTHETIC_ONLY').is_file():raise ValueError('Synthetic fixture marker required')
        contract=json.loads((root/'contract.json').read_text());parent=load_parent(root/'parent',contract)
        a=screen(parent,root/'out','attribution','synthetic-integration',replay_only=True)
        b=screen(parent,root/'out','motion','synthetic-integration',load_extra(parent,root/'out','synthetic-integration'),replay_only=True)
        assert a['new_fits_this_invocation']==b['new_fits_this_invocation']==0
        atomic_json(root/'fresh_process_replay.json',{'scope':'SYNTHETIC','new_fits':0,'candidate_models_replayed':27,
                    'parent_models_replayed':6,'all_forward_replays_exact':True})
        return
    if root.exists():raise ValueError('Choose a new synthetic validation directory; existing evidence is preserved')
    root.mkdir(parents=True);(root/'SYNTHETIC_ONLY').write_text('No NFL data.\n');started=time.monotonic()
    parentdir,repo,contract=parent_fixture(root);atomic_json(root/'contract.json',contract)
    parent=load_parent(parentdir,contract);out=root/'out';out.mkdir()
    before={p.relative_to(parentdir).as_posix():(digest(p),p.stat().st_mtime_ns) for p in parentdir.rglob('*') if p.is_file()}
    a=screen(parent,out,'attribution','synthetic-integration')
    ns=argparse.Namespace(repo=repo,parent=parentdir,out=out)
    prep=prepare_features(ns,parent,contract,'synthetic-integration',limit=256)
    extra=load_extra(parent,out,'synthetic-integration')
    b=screen(parent,out,'motion','synthetic-integration',extra)
    saved={p.relative_to(out).as_posix():(digest(p),p.stat().st_mtime_ns) for p in out.rglob('*.npz')}
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(root),'--replay'],check=True,env=env,timeout=30)
    assert saved=={p.relative_to(out).as_posix():(digest(p),p.stat().st_mtime_ns) for p in out.rglob('*.npz')}
    assert before=={p.relative_to(parentdir).as_posix():(digest(p),p.stat().st_mtime_ns) for p in parentdir.rglob('*') if p.is_file()}
    export_report(out)
    atomic_json(root/'validation.json',{'scope':'SYNTHETIC ONLY — not NFL/AWS data or accuracy',
                'status':'passed','attribution_fits':a['new_fits_this_invocation'],'motion_fits':b['new_fits_this_invocation'],
                'parent_refits':0,'fresh_process_replay':json.loads((root/'fresh_process_replay.json').read_text()),
                'parent_files_hash_and_mtime_unchanged':True,'new_checkpoint_hash_and_mtime_unchanged_on_replay':True,
                'feature_preparation_plays':prep['plays'],'elapsed_seconds':round(time.monotonic()-started,3)})
    print(json.dumps(json.loads((root/'validation.json').read_text()),indent=2))

if __name__=='__main__':main()
