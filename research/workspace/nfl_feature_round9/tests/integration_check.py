"""Synthetic end-to-end audit; no optimizer, training labels or NFL data."""
from __future__ import annotations
import hashlib,io,json,sys,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fixtures import sample
from audit_io import digest,hash_json,atomic_json,atomic
from parent_audit import verify_parent,assert_unchanged
from audit_worker import preflight,feature_smoke,run_signal_audit
from sequence_model import Forecast,setup,CONFIG
from signal_probe import weights_digest


def create(root,kit,kit8):
    parent=root/'parent';parent.mkdir(parents=True)
    norm={'mean':[0.]*72,'scale':[1.]*72}
    files=[];dsig='synthetic-data-not-NFL'
    for i in range(32):
        p=sample(i+1);p['signature']=np.array(dsig)
        # Poison object labels prove input loading never unpacks y.
        p['y']=np.array(['DO NOT READ'],object)
        path=parent/f'plays/2023090700_{i+1}.npz';path.parent.mkdir(exist_ok=True)
        with path.open('wb') as f:np.savez_compressed(f,**p)
        files.append({'path':str(path.relative_to(parent)),'sha256':digest(path),'train':True,'rows':8,'players':4})
    # Evaluation metadata exists but its file intentionally does not.
    files.append({'path':'plays/2023100700_1.npz','sha256':'0'*64,'train':False,'rows':8,'players':4})
    source={p.name:digest(p) for p in sorted(kit8.glob('*.py'))};contract_hash=digest(kit8/'input_contract.json')
    dataset={'code':hash_json(dict(source,contract=contract_hash)),'signature':dsig,'fold':2,'files':files,'training_rows':256,'evaluation_rows':8}
    atomic_json(parent/'dataset.json',dataset)
    setup();initial=Forecast();initial_hash=weights_digest(initial)
    protocol={'data_sha256':digest(parent/'dataset.json'),'signature':dsig,'config':CONFIG,'normalization':norm,
              'environment':{'synthetic':True},'parameters':14931,'fixture_only':True}
    atomic_json(parent/'neural_protocol.json',protocol);psig=hash_json(protocol)
    for arm in ('terminal','history'):
        setup();model=Forecast()
        with torch.no_grad():model.decoder[-1].weight.normal_(0,.1)
        # Different but untrained fixture weights. Not scientific checkpoints.
        if arm=='history':
            with torch.no_grad():model.pair.frame.bias.add_(.01)
        sig=hash_json({'protocol':psig,'arm':arm})
        state={'model':model.state_dict(),'signature':sig,'step':0,'initial_hash':initial_hash}
        payload=io.BytesIO();torch.save(state,payload);raw=payload.getvalue();h=hashlib.sha256(raw).hexdigest()
        folder=parent/'models'/arm;folder.mkdir(parents=True)
        blob=folder/(h+'.pt');blob.write_bytes(raw)
        atomic_json(folder/'checkpoint.json',{'signature':sig,'step':0,'initial_hash':initial_hash,'blob':blob.name,'sha256':h,'bytes':len(raw)})
    atomic_json(parent/'summary.json',{'synthetic':True,'scientific_models_trained':0})
    c={'round8_python_hashes':source,'round8_contract_hash':contract_hash,
       'report_hashes':{'summary.json':digest(parent/'summary.json')},'protocol_signature':psig,'data_signature':dsig,
       'steps':0,'initial_hash':initial_hash}
    atomic_json(root/'fixture_contract.json',c)
    return parent,c


def main():
    kit=Path(__file__).resolve().parents[1];kit8=Path(sys.argv[2]).resolve();root=Path(sys.argv[1]).resolve()
    if len(sys.argv)>3 and sys.argv[3]=='replay':
        parent=root/'parent';c=json.loads((root/'fixture_contract.json').read_text())
        v=verify_parent(parent,kit8,c);r=run_signal_audit(parent,kit,root/'result',c,v,replay=True,strict_environment=False)
        print(json.dumps(r));return
    root.mkdir(parents=True,exist_ok=True);parent,c=create(root,kit,kit8);out=root/'result';out.mkdir()
    v=preflight(parent,kit8,None,kit,out,c)
    feature_smoke(parent,kit,out,c,v)
    r=run_signal_audit(parent,kit,out,c,v,strict_environment=False)
    paths=[p for p in root.rglob('*') if p.is_file()]
    before={str(p):digest(p) for p in paths}
    # Reuse is exact and does not alter completed output files.
    again=run_signal_audit(parent,kit,out,c,v,strict_environment=False)
    assert r==again
    env=dict(__import__('os').environ);env['PYTHONPATH']=str(kit8)+':'+str(kit)
    subprocess.run([sys.executable,__file__,str(root),str(kit8),'replay'],check=True,env=env,timeout=30)
    assert all(digest(Path(p))==h for p,h in before.items())
    assert_unchanged(v['protected'])
    evidence={'status':'synthetic_integration_passed','training_plays_inspected':32,'model_fits':0,'optimizer_steps':0,
              'evaluation_file_absent_but_audit_passed':True,'object_labels_not_unpacked':True,
              'files_byte_identical_after_reuse_and_fresh_process_replay':len(before),
              'fixture_checkpoint_weights':'untrained synthetic weights, never NFL weights',
              'python':sys.version.split()[0],'numpy':np.__version__,'torch':torch.__version__}
    atomic_json(kit/'evidence/synthetic_integration.json',evidence);print(json.dumps(evidence))
if __name__=='__main__':main()
