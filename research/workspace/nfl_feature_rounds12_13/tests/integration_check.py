"""Synthetic fixture with separate contracts; exercises the real verification chain.

Never operates on /home/sagemaker-user. A private temporary replica has synthetic
parents, its own Git commit, local environment contract, and no production data.
The pinned AWS uv runtime launcher is not replaced in the delivered source.
"""
from __future__ import annotations
import sys,shutil,json,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
KIT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(KIT));sys.path.insert(0,str(KIT/'tests'))
from fixtures import model_sample
from audit_io import atomic_json,digest,hash_json,checkpoint_npz,read_json
from new_features import build as old_features
from grouped_features import make_extension
import parent_core,bridge,study
from model import CONFIG

def setup_fixture(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    kit=root/'kit';shutil.copytree(KIT,kit,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('parent','parentkit','audit','auditkit','tensors','repo','out12','out13'):(root/name).mkdir()
    a=args(root,12);rng=np.random.default_rng(190)
    for p in (a.parent,a.audit,a.tensors):(p/'.run.lock').touch()
    for folder in (a.parent_kit,a.audit_kit):(folder/'placeholder.py').write_text('# Synthetic fixture source identity\n')
    atomic_json(a.parent_kit/'input_contract.json',{'fixture':True})
    rows=[];goals=[];feature=[];ev=[]
    for i in range(40):
        p=model_sample(i+1,n=4);p['train']=np.array(i<32);p['signature']=np.array('synthetic_observed')
        game=2023091000+(i//8)*100 if i<32 else 2023111000+(i//4)*100
        p['keys'][:,0]=game;p['base'][:,2:]=rng.normal(size=(8,70))*.2
        p['y']=np.column_stack([p['base'][:,2]*.4+np.arange(8)*.1,-p['base'][:,3]*.3])
        ext={k:p.pop(k) for k in ('goal','goal_valid','goal_age','groups')}
        path=a.tensors/'plays'/f'{game}_{i+1}.npz';checkpoint_npz(path,p)
        orig={'path':str(path.relative_to(a.tensors)),'sha256':digest(path),'train':i<32,'rows':8};rows.append(orig)
        g=a.parent/'goal'/path.name;checkpoint_npz(g,ext)
        goals.append({'path':str(g.relative_to(a.parent)),'sha256':digest(g),'parent_path':orig['path'],'parent_sha256':orig['sha256'],'train':i<32,'rows':8})
        if i<32:
            f=a.audit/'features'/path.name;checkpoint_npz(f,old_features(p));feature.append({'path':str(f.relative_to(a.audit)),'sha256':digest(f),'parent_path':orig['path'],'parent_sha256':orig['sha256']})
        else:ev.append(p)
    ref={'keys':np.concatenate([p['keys'] for p in ev]),'truth':np.concatenate([p['y'] for p in ev])};ref['pred']=np.zeros_like(ref['truth'])
    checkpoint_npz(a.tensors/'reference.npz',ref)
    atomic_json(a.tensors/'dataset.json',{'signature':'synthetic_observed','files':rows})
    m={'source':'synthetic_R10_source','signature':'synthetic_R10_data','parent':str(a.tensors),
       'parent_manifest_sha256':digest(a.tensors/'dataset.json'),'reference_sha256':digest(a.tensors/'reference.npz'),
       'files':goals,'training_rows':256,'evaluation_rows':64}
    atomic_json(a.parent/'dataset.json',m)
    proto={'data_sha256':digest(a.parent/'dataset.json'),'environment':study.environment(),'normalization':{'mean':[0]*72,'scale':[1]*72},'config':CONFIG}
    atomic_json(a.parent/'protocol.json',proto);psig=hash_json(proto)
    atomic_json(a.parent/'summary.json',{'metrics':{'cartesian':1.,'goal':1.,'preserved_tree':1.},'contrast':{'feature_gate_passed':False},'evaluation_rows':64})
    atomic_json(a.parent/'replay.json',{'status':'grouped_goal_replay_exact','new_optimizer_steps':0,'summary_sha256':digest(a.parent/'summary.json')})
    names=['summary.json','replay.json']
    for arm in ('cartesian','goal'):
        folder=a.parent/'models'/arm;folder.mkdir(parents=True)
        payload=b'Synthetic previous checkpoint; bytes authenticated, never unpickled';sha=hashlib.sha256(payload).hexdigest();(folder/(sha+'.pt')).write_bytes(payload)
        atomic_json(folder/'checkpoint.json',{'signature':hash_json({'protocol':psig,'arm':arm}),'step':0,'initial_hash':'fixture','blob':sha+'.pt','sha256':sha,'bytes':len(payload)})
        atomic_json(folder/'complete.json',{'steps':0,'initial_hash':'fixture'});names.append(f'models/{arm}/complete.json')
    subprocess.run(['git','init','-q',str(a.repo)],check=True)
    subprocess.run(['git','-C',str(a.repo),'-c','user.name=Fixture','-c','user.email=fixture@example.com','commit','--allow-empty','-qm','Synthetic input identity'],check=True)
    head=subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip()
    files=[*a.parent_kit.glob('*.py'),a.parent_kit/'input_contract.json']
    c11={'parent_source_files':{str(p.relative_to(a.parent_kit)):digest(p) for p in files},'parent_reports':{n:digest(a.parent/n) for n in names},
         'frozen_modules':{},'repo_head':head,'parent_data_signature':m['signature'],'parent_source_signature':m['source'],
         'protocol_signature':psig,'training_rows':256,'evaluation_rows':64,'environment':study.environment()}
    atomic_json(a.audit_kit/'input_contract.json',c11)
    old=SimpleNamespace(kit=a.audit_kit,parent=a.parent,parent_kit=a.parent_kit,tensors=a.tensors,repo=a.repo)
    v=parent_core.verify(old);sig=v['signature']
    atomic_json(a.audit/'feature_manifest.json',{'signature':sig,'files':feature})
    for n in ('preflight.json','coverage_summary.json','feature_summary.json','training_audit.json'):
        atomic_json(a.audit/n,{'signature':sig,'synthetic':True})
    atomic_json(a.audit/'replay.json',{'status':'training_review_replay_exact','new_optimizer_steps':0,
        'coverage_summary_sha256':digest(a.audit/'coverage_summary.json'),'feature_summary_sha256':digest(a.audit/'feature_summary.json'),
        'training_summary_sha256':digest(a.audit/'training_audit.json')})
    c=read_json(kit/'input_contract.json');c.update(repo_head=head,round11_signature=sig,
        round11_source_files={str(p.relative_to(a.audit_kit)):digest(p) for p in [*a.audit_kit.glob('*.py'),a.audit_kit/'input_contract.json']},
        round11_reports={n:digest(a.audit/n) for n in ['preflight.json','coverage_summary.json','feature_summary.json','training_audit.json','replay.json']})
    atomic_json(kit/'input_contract.json',c)
    return a

def args(root,r):
    root=Path(root)
    return SimpleNamespace(kit=root/'kit',out=root/f'out{r}',parent=root/'parent',parent_kit=root/'parentkit',audit=root/'audit',audit_kit=root/'auditkit',tensors=root/'tensors',repo=root/'repo',round_no=r)

def run(root):
    a=setup_fixture(root);before={str(p):digest(p) for d in (a.parent,a.audit,a.tensors,a.repo,a.parent_kit,a.audit_kit) for p in d.rglob('*') if p.is_file()}
    results=[]
    for r in (12,13):
        a=args(root,r);bridge.preflight(a);bridge.features(a,smoke=True);bridge.features(a)
        atomic_json(a.out/'runtime_receipt.json',{'status':'family_runtime_ready','signature':bridge.context(a)['signature'],'environment':study.environment()})
        assert study.profile(a)['status']=='family_profile_passed'
        study.train_arm(a,'mask');study.train_arm(a,'core')
        assert study.train_arm(a,'full',max_steps=17)['status']=='incomplete_checkpoint_preserved'
        final=study.train_arm(a,'full');assert final['new_optimizer_steps']==79
        study.evaluate(a)
        protected={str(p):digest(p) for sub in ('models','predictions','features') for p in (a.out/sub).rglob('*') if p.is_file()}
        subprocess.run([sys.executable,str(KIT/'tests/integration_check.py'),'replay',str(root),str(r)],check=True,timeout=45)
        assert all(digest(Path(p))==h for p,h in protected.items())
        assert all(digest(Path(p))==h for p,h in before.items())
        for arm in ('mask','core','full'):assert study.train_arm(a,arm)['new_optimizer_steps']==0
        results.append({'round':r,'models':3,'steps_per_arm':96,'training_rows':256,'evaluation_rows':64,'fresh_process_replay':True,'new_steps_during_replay':0,'unchanged_results':len(protected)})
    atomic_json(Path(root)/'validation.json',{'status':'synthetic_pair_complete','rounds':results,'protected_parent_files':len(before),'source_contract':'separate synthetic fixture; real verifier and preparation/training/evaluation/replay functions','launcher_boundary':'AWS pinned uv launcher not exercised'})
    print(json.dumps(results))

if __name__=='__main__':
    if sys.argv[1]=='replay':
        a=args(sys.argv[2],int(sys.argv[3]));bridge.features(a,replay=True);r=study.evaluate(a,replay=True);assert r['new_optimizer_steps']==0;print(r)
    else:run(sys.argv[1])
