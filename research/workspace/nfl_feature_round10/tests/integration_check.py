"""Synthetic end-to-end fixture. Never run this against private NFL sources.

Uses a separate kit copy with fixture-only contract; the delivered contract and
real AWS launcher are not weakened. Tests authenticating, preparation, training,
interruption, reuse, immutable checkpoints and fresh-process replay.
"""
from __future__ import annotations
import sys,shutil,json,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
KIT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(KIT));sys.path.insert(0,str(KIT/'tests'))
from fixtures import model_sample
from audit_io import atomic,atomic_json,digest,hash_json,checkpoint_npz,read_json
from grouped_features import make_extension
from data_io import preflight,prepare,code_signature
from study import profile,train_arm,evaluate


def fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    kit=root/'kit';shutil.copytree(KIT,kit,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    parent=root/'parent';audit=root/'audit';oldkit=root/'oldkit';repo=root/'repo';out=root/'out'
    for p in (parent,audit,oldkit,repo,out):p.mkdir()
    for d in (parent,audit):(d/'.run.lock').touch()
    (oldkit/'placeholder.py').write_text('# Synthetic source identity only\n')
    atomic_json(oldkit/'input_contract.json',{'fixture':True})
    h8={'placeholder.py':digest(oldkit/'placeholder.py')}
    c8hash=digest(oldkit/'input_contract.json');data_sig='synthetic-observed-tensors'
    records=[];train=[];ev=[];goalcounts=np.zeros(6,np.int64);groups=np.zeros(4,np.int64)
    rng=np.random.default_rng(719)
    for i in range(40):
        p=model_sample(i+1,n=4+(i%2))
        p['train']=np.array(i<32);p['signature']=np.array(data_sig)
        game=(2023091000+(i//8)*100) if i<32 else (2023101000+(i//4)*100)
        p['keys'][:,0]=game
        p['base'][:,2:]=rng.normal(size=(8,70))*.2
        p['y']=np.column_stack([.4*p['base'][:,2]+.1*np.arange(8),-.3*p['base'][:,3]])
        # Round 8 parent does not already contain goal extension fields.
        for k in ('goal','goal_valid','goal_age','groups'):p.pop(k)
        path=parent/'plays'/f'{game}_{i+1}.npz';checkpoint_npz(path,p)
        r={'path':str(path.relative_to(parent)),'sha256':digest(path),'train':i<32,'rows':8,'players':len(p['ids'])}
        records.append(r);(train if i<32 else ev).append(p)
        if i<32:
            g=make_extension(p);goalcounts+=g['goal_valid'].sum((0,1,2));groups+=g['groups'].sum((0,1,2))
    ref={'keys':np.concatenate([p['keys'] for p in ev]),'truth':np.concatenate([p['y'] for p in ev])}
    ref['pred']=np.zeros_like(ref['truth']);checkpoint_npz(parent/'reference.npz',ref)
    manifest={'code':hash_json(dict(h8,contract=c8hash)),'signature':data_sig,'fold':2,'files':records,
              'training_rows':256,'evaluation_rows':64,'reference_sha256':digest(parent/'reference.npz')}
    atomic_json(parent/'dataset.json',manifest)
    protocol={'data_sha256':digest(parent/'dataset.json'),'config':{'epochs':24,'evaluation_fold':2}}
    atomic_json(parent/'neural_protocol.json',protocol);psig=hash_json(protocol)
    atomic_json(parent/'summary.json',{'status':'matched_encoder_study_complete','synthetic':True})
    for arm in ('terminal','history'):
        f=parent/'models'/arm;f.mkdir(parents=True)
        payload=b'synthetic parent checkpoint identity; never loaded';h=hashlib.sha256(payload).hexdigest()
        (f/(h+'.pt')).write_bytes(payload)
        atomic_json(f/'checkpoint.json',{'step':0,'signature':hash_json({'protocol':psig,'arm':arm}),
                                       'blob':h+'.pt','sha256':h,'bytes':len(payload),'initial_hash':'synthetic'})
    atomic_json(audit/'signal_summary.json',{'status':'signal_audit_complete','synthetic':True})
    atomic_json(audit/'replay.json',{'status':'signal_audit_replay_exact','new_optimizer_steps':0,'summary_sha256':digest(audit/'signal_summary.json')})
    atomic_json(audit/'feature_smoke.json',{'valid_counts':goalcounts.tolist(),'group_pair_frames':groups.tolist(),'training_rows':256})
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.email=test@example.com','-c','user.name=Test','commit','--allow-empty','-qm','Synthetic fixture'],check=True)
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    c={'round8':{'round8_python_hashes':h8,'round8_contract_hash':c8hash,'report_hashes':{'summary.json':digest(parent/'summary.json')},
                 'protocol_signature':psig,'data_signature':data_sig,'steps':0,'initial_hash':'synthetic','repo_head':head},
       'round9_receipts':{n:digest(audit/n) for n in ('signal_summary.json','replay.json','feature_smoke.json')},
       'goal_frame_sha256':digest(kit/'goal_frame.py'),'repo_head':head}
    atomic_json(kit/'input_contract.json',c)
    return SimpleNamespace(kit=kit,parent=parent,parent_kit=oldkit,audit=audit,repo=repo,out=out)


def run(root):
    a=fixture(root);protected={str(p):digest(p) for d in (a.parent,a.parent_kit,a.audit) for p in d.rglob('*') if p.is_file()}
    preflight(a);prepare(a,True);prepare(a,False)
    assert profile(a)['status']=='grouped_profile_passed'
    # Abort one arm at a durable checkpoint, then complete it with identical input.
    train_arm(a,'cartesian')
    assert train_arm(a,'goal',max_steps=17)['status']=='incomplete_checkpoint_preserved'
    r=train_arm(a,'goal');assert r['new_optimizer_steps']==79
    evaluate(a)
    before={str(p):digest(p) for p in a.out.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz')}
    cmd=[sys.executable,str(KIT/'tests/integration_check.py'),'replay',str(root)]
    subprocess.run(cmd,check=True,timeout=45)
    assert all(digest(Path(p))==h for p,h in before.items())
    assert all(digest(Path(p))==h for p,h in protected.items())
    assert train_arm(a,'cartesian')['new_optimizer_steps']==0
    assert train_arm(a,'goal')['new_optimizer_steps']==0
    receipt={'status':'synthetic_end_to_end_passed','plays':40,'training_rows':256,'evaluation_rows':64,
             'steps_per_arm':96,'goal_arm_interrupted_at_step':17,'exact_replay':True,
             'protected_parent_files':len(protected),'unchanged_model_and_tensor_files':len(before),
             'new_optimizer_steps_during_replay':0,
             'scope':'Synthetic separately contracted inputs; not the AWS pinned-runtime launcher'}
    atomic_json(Path(root)/'validation.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    if sys.argv[1]=='replay':
        root=Path(sys.argv[2]);a=SimpleNamespace(kit=root/'kit',parent=root/'parent',parent_kit=root/'oldkit',audit=root/'audit',repo=root/'repo',out=root/'out')
        r=evaluate(a,True);assert r['new_optimizer_steps']==0;print(r)
    else:run(sys.argv[1])
