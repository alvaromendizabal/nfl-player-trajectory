"""Reuse the authenticated Round 8 tensors; never refit or rewrite parents."""
from __future__ import annotations
import json
from pathlib import Path
import zipfile
import numpy as np
from audit_io import (digest,hash_json,safe_file,read_json,atomic_json,seal_json,
                      checkpoint_npz,INPUT_FIELDS)
from parent_audit import verify_parent, assert_unchanged
from grouped_features import make_extension
from goal_frame import NAMES, GROUPS


def code_signature(kit):
    kit=Path(kit)
    paths=sorted(list(kit.glob('*.py'))+[kit/'input_contract.json']+list((kit/'tests').rglob('*.py')))
    return hash_json({str(p.relative_to(kit)):digest(p) for p in paths})


def verify(a):
    c=read_json(a.kit/'input_contract.json')
    v=verify_parent(a.parent,a.parent_kit,c['round8'],a.repo)
    for name,h in c['round9_receipts'].items():
        p=safe_file(a.audit,name)
        if digest(p)!=h: raise ValueError('Reviewed Round 9 receipt changed: '+name)
        v['protected'][str(p)]=h
    r=read_json(a.audit/'replay.json')
    if r['status']!='signal_audit_replay_exact' or r['new_optimizer_steps']!=0:
        raise ValueError('Completed no-fit Round 9 audit required')
    if r['summary_sha256']!=digest(a.audit/'signal_summary.json'):
        raise ValueError('Round 9 summary/replay mismatch')
    if digest(a.kit/'goal_frame.py')!=c['goal_frame_sha256']:
        raise ValueError('Reviewed six-channel feature implementation changed')
    v['source_signature']=code_signature(a.kit)
    v['signature']=hash_json({'source':v['source_signature'],'parent_data':digest(a.parent/'dataset.json'),
                              'round9':c['round9_receipts']})
    return v


def read_arrays(path, fields):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or sum(i.file_size for i in z.infolist())>64*1024**2:
            raise ValueError('Oversized or duplicate-key tensor archive')
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in fields}


def preflight(a):
    v=verify(a)
    p=v['protocol']; expected=p['config']
    if expected['epochs']!=24 or expected['evaluation_fold']!=2:
        raise ValueError('Unexpected parent study')
    result={'status':'grouped_preflight_passed','signature':v['signature'],
            'parent_training_rows':v['manifest']['training_rows'],
            'parent_evaluation_rows':v['manifest']['evaluation_rows'],
            'prepared_parent_plays':len(v['manifest']['files']),
            'protected_files':len(v['protected']),'source_commit':read_json(a.kit/'input_contract.json')['repo_head'],
            'raw_csv_reads':0,'scientific_fits':0,'optimizer_steps':0,
            'old_weights_loaded_into_new_models':False,'feature_research':'open'}
    seal_json(a.out/'preflight.json',result)
    seal_json(a.out/'parent_protection.json',v['protected'])
    assert_unchanged(v['protected'])
    return result


def prepare(a, smoke=False):
    v=verify(a)
    if read_json(safe_file(a.out,'preflight.json'))['signature']!=v['signature']:
        raise ValueError('Run current preflight first')
    if not smoke:
        s=read_json(safe_file(a.out,'smoke.json'))
        if s['signature']!=v['signature'] or s['status']!='grouped_smoke_passed':
            raise ValueError('Current training smoke required')
    records=v['selected'] if smoke else v['manifest']['files']
    output=[]; reused=0; counts=np.zeros(6,np.int64); sums=np.zeros(6); groups=np.zeros(4,np.int64)
    rows=0
    for i,r in enumerate(records):
        pp=safe_file(a.parent,r['path'])
        if digest(pp)!=r['sha256']: raise ValueError('Prepared parent tensor changed')
        # Target coordinates are not unpacked during feature construction.
        p=read_arrays(pp,INPUT_FIELDS)
        if bool(p['train'])!=r['train'] or str(p['signature'])!=v['manifest']['signature']:
            raise ValueError('Tensor split/signature mismatch')
        if smoke and not bool(p['train']): raise ValueError('Smoke cannot read evaluation inputs')
        g=make_extension(p)
        path=a.out/'features'/Path(r['path']).name
        existed=path.exists()
        h=checkpoint_npz(path,g)
        reused+=int(existed); rows+=len(p['base'])
        if bool(p['train']):
            counts+=g['goal_valid'].sum((0,1,2))
            sums+=(g['goal'].astype(np.float64)**2).sum((0,1,2))
            groups+=g['groups'].sum((0,1,2))
        output.append({'path':str(path.relative_to(a.out)),'sha256':h,'parent_path':r['path'],
                       'parent_sha256':r['sha256'],'train':r['train'],'rows':r['rows']})
        if i==0 or (i+1)%64==0 or i+1==len(records):
            print(json.dumps({'event':'goal_features','completed':i+1,'total':len(records),'reused':reused}),flush=True)
    if smoke:
        old=read_json(a.audit/'feature_smoke.json')
        if not np.array_equal(counts,old['valid_counts']) or not np.array_equal(groups,old['group_pair_frames']):
            raise ValueError('Goal support differs from the reviewed 32-play audit')
        if rows!=old['training_rows']:raise ValueError('Smoke row population changed')
    else:
        manifest={'signature':v['signature'],'source':v['source_signature'],
                  'parent_manifest_sha256':digest(a.parent/'dataset.json'),'parent':str(a.parent),
                  'reference_sha256':v['manifest']['reference_sha256'],'files':output,
                  'training_rows':v['manifest']['training_rows'],'evaluation_rows':v['manifest']['evaluation_rows']}
        seal_json(a.out/'dataset.json',manifest)
        protect={str(safe_file(a.parent,r['parent_path'])):r['parent_sha256'] for r in output}
        protect[str(safe_file(a.parent,'reference.npz'))]=manifest['reference_sha256']
        seal_json(a.out/'tensor_protection.json',protect)
    result={'status':'grouped_smoke_passed' if smoke else 'grouped_features_ready',
            'signature':v['signature'],'plays':len(records),'rows':rows,
            'new_checkpoints':len(records)-reused,'reused_checkpoints':reused,
            'channels':list(NAMES),'training_valid_counts':counts.tolist(),
            'training_rms_scaled':np.sqrt(sums/np.maximum(counts,1)).tolist(),
            'group_names':list(GROUPS),'training_group_pair_frames':groups.tolist(),
            'future_targets_unpacked':False,'raw_csv_reads':0,'scientific_fits':0}
    name='smoke.json' if smoke else 'preparation.json'
    if (a.out/name).exists():name=name.replace('.json','_reuse.json')
    atomic_json(a.out/name,result)
    assert_unchanged(v['protected'])
    return result


FIELDS=INPUT_FIELDS+('keys','y','role_query')


def load_training(a):
    """Load only training tensors/labels; normalization and profiling never see evaluation arrays."""
    return load_split(a,True)


def load_split(a,training):
    m=read_json(safe_file(a.out,'dataset.json'))
    if m['source']!=code_signature(a.kit) or m['parent']!=str(a.parent):
        raise ValueError('Frozen source/location changed')
    if digest(safe_file(a.parent,'dataset.json'))!=m['parent_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    plays=[]; total=0
    for r in m['files']:
        if r['train']!=training:continue
        p=safe_file(a.parent,r['parent_path']); q=safe_file(a.out,r['path'])
        if digest(p)!=r['parent_sha256'] or digest(q)!=r['sha256']:
            raise ValueError('Tensor checksum drift')
        z=read_arrays(p,FIELDS); ext=read_arrays(q,('goal','goal_valid','goal_age','groups'))
        z.update(ext)
        if bool(z['train'])!=training or len(z['keys'])!=r['rows'] or z['keys'].shape!=(len(z['base']),4):
            raise ValueError('Split/key count drift')
        if not np.isfinite(z['y']).all():raise ValueError('Nonfinite target')
        total+=sum(t.nbytes for t in z.values())
        if total>1024**3: raise ValueError('One-GiB split memory limit')
        plays.append(z)
    if not plays:raise ValueError('Empty split')
    keys=np.concatenate([p['keys'] for p in plays])
    expected=m['training_rows' if training else 'evaluation_rows']
    if len(keys)!=expected or len(np.unique(keys,axis=0))!=len(keys):
        raise ValueError('Frozen row count or uniqueness changed')
    return plays,m


def evaluation_data(a):
    plays,m=load_split(a,False)
    p=safe_file(a.parent,'reference.npz')
    if digest(p)!=m['reference_sha256']:raise ValueError('Preserved reference changed')
    ref=read_arrays(p,('keys','truth','pred'))
    if not np.array_equal(ref['keys'],np.concatenate([p['keys'] for p in plays])) or not np.array_equal(ref['truth'],np.concatenate([p['y'] for p in plays])):
        raise ValueError('Evaluation/reference alignment differs')
    return plays,ref
