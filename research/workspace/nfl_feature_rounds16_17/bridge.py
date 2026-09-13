"""Verify the Round 11/12/13 chain. The prior feature gates remain inconclusive."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import parent_core
from audit_io import digest,hash_json,safe_file,read_json,seal_json,atomic_json,checkpoint_npz
from families import build,summarize
from motion_reference import build as build_motion

def code_signature(kit):
    paths=sorted([*kit.glob('*.py'),kit/'input_contract.json',*(kit/'tests').rglob('*.py')])
    return hash_json({str(p.relative_to(kit)):digest(p) for p in paths})

def verify(a):
    c=read_json(safe_file(a.kit,'input_contract.json'));protected={}
    for n,h in c['round11_source_files'].items():
        p=safe_file(a.audit_kit,n)
        if digest(p)!=h:raise ValueError('Round 11 source differs: '+n)
        protected[str(p)]=h
    actual=sorted([*a.audit_kit.glob('*.py'),a.audit_kit/'input_contract.json',*(a.audit_kit/'tests').rglob('*.py')])
    if {str(p.relative_to(a.audit_kit)) for p in actual}!=set(c['round11_source_files']):
        raise ValueError('Round 11 source file set differs')
    for n,h in c['round11_reports'].items():
        p=safe_file(a.audit,n)
        if digest(p)!=h:raise ValueError('Round 11 report differs: '+n)
        protected[str(p)]=h
    replay=read_json(a.audit/'replay.json')
    for field,name in [('coverage_summary_sha256','coverage_summary.json'),('feature_summary_sha256','feature_summary.json'),('training_summary_sha256','training_audit.json')]:
        if replay[field]!=digest(a.audit/name):raise ValueError('Round 11 replay checksum mismatch')
    if replay['status']!='training_review_replay_exact' or replay['new_optimizer_steps']!=0:
        raise ValueError('Completed no-fit Round 11 required')
    old=SimpleNamespace(kit=a.audit_kit,parent=a.parent,parent_kit=a.parent_kit,tensors=a.tensors,repo=a.repo)
    v=parent_core.verify(old)
    if v['signature']!=c['round11_signature']:raise ValueError('Reviewed parent identity changed')
    v['protected'].update(protected)
    fm=read_json(safe_file(a.audit,'feature_manifest.json'))
    if fm['signature']!=v['signature']:raise ValueError('Round 11 feature manifest identity changed')
    v['feature_lookup']={r['parent_path']:r for r in fm['files']}
    if len(v['feature_lookup'])!=len(fm['files']) or set(v['feature_lookup'])!={r['parent_path'] for r in v['train']}:
        raise ValueError('Round 11 feature population changed')
    v['protected'][str(a.audit/'feature_manifest.json')]=digest(a.audit/'feature_manifest.json')
    # Both uploaded completed studies are historical evidence, not acceptance.
    cnew = c
    prior_kit = a.audit_kit.parent / 'nfl_feature_rounds12_13'
    for name, expected in cnew['prior_kit_source'].items():
        file = safe_file(prior_kit,name)
        if digest(file) != expected: raise ValueError('Earlier family source drift: '+name)
        v['protected'][str(file)] = expected
    for round_id, reports in cnew['prior_round_reports'].items():
        folder = a.audit.parent / ('nfl-feature-round'+round_id+'-results')
        for name, expected in reports.items():
            file = safe_file(folder,name)
            if digest(file) != expected: raise ValueError('Reviewed Round '+round_id+' report drift: '+name)
            v['protected'][str(file)] = expected
        rp = read_json(folder/'replay.json')
        if rp['status'] != 'family_replay_exact' or rp['new_optimizer_steps'] != 0 or rp['summary_sha256'] != digest(folder/'summary.json'):
            raise ValueError('Earlier family replay does not match summary')
    motion_root = a.audit.parent/'nfl-feature-round13-results'
    motion_manifest = read_json(safe_file(motion_root,'dataset.json'))
    if motion_manifest['signature'] != read_json(motion_root/'preflight.json')['signature'] or motion_manifest['round'] != 13:
        raise ValueError('Round 13 feature manifest identity mismatch')
    v['motion_lookup'] = {r['parent_path']:r for r in motion_manifest['files']}
    if len(v['motion_lookup']) != len(motion_manifest['files']) or set(v['motion_lookup']) != {r['parent_path'] for r in v['manifest']['files']}:
        raise ValueError('Round 13 feature population mismatch')
    v['motion_root'] = motion_root
    v['protected'][str(motion_root/'dataset.json')] = digest(motion_root/'dataset.json')
    v['signature']=hash_json({'source':code_signature(a.kit),'round':a.round_no,'round11':c['round11_signature'],
                             'parent_data':digest(a.parent/'dataset.json')})
    v['source']=code_signature(a.kit)
    return v

def context(a):
    v=verify(a)
    if read_json(safe_file(a.out,'preflight.json'))['signature']!=v['signature']:
        raise ValueError('Run current preflight first')
    return v

def unchanged(v):parent_core.unchanged(v['protected'])

def preflight(a):
    from readiness import require_tests
    require_tests(a)
    v=verify(a)
    r={'status':'family_preflight_passed','round':a.round_no,'signature':v['signature'],
       'source_commit':v['contract']['repo_head'],'training_plays':len(v['train']),
       'training_rows':sum(x['rows'] for x in v['train']),'evaluation_rows':v['contract']['evaluation_rows'],
       'evaluation_games':len(v['evaluation_games']),'new_model_fits':0,'evaluation_arrays_opened':False,
       'parent_checkpoint_check':'hash verification, not new numerical replay','feature_research':'open'}
    seal_json(a.out/'preflight.json',r);unchanged(v);return r

def observed(a,r,v,labels=False,allow_eval=False):
    if not r['train'] and labels and not allow_eval:raise ValueError('Evaluation targets blocked')
    p=safe_file(a.tensors,r['parent_path'])
    if digest(p)!=r['parent_sha256']:raise ValueError('Original tensor checksum changed')
    fields=parent_core.OBS+(('keys','y','role_query') if labels else ())
    z=parent_core.arrays(p,fields)
    if bool(z['train'])!=r['train'] or str(z['signature'])!=v['input_signature'] or len(z['base'])!=r['rows']:
        raise ValueError('Original input signature/split/row mismatch')
    v['protected'][str(p)]=r['parent_sha256']
    if labels:
        if z['keys'].shape!=(r['rows'],4) or z['y'].shape!=(r['rows'],2) or not np.isfinite(z['y']).all():
            raise ValueError('Invalid target/key arrays')
        if not np.all(z['keys'][:,:2]==parent_core.game_play(r)):raise ValueError('Play identity mismatch')
        ext=safe_file(a.parent,r['path'])
        if digest(ext)!=r['sha256']:raise ValueError('Original group/goal input changed')
        z.update(parent_core.arrays(ext,parent_core.EXT));v['protected'][str(ext)]=r['sha256']
    return z

def verify_anchor(a,r,sample,v):
    """Rebuild the motion anchor and compare exact stored arrays; no old refit."""
    rebuilt=build_motion(sample,13)
    old=v['motion_lookup'][r['parent_path']]
    path=safe_file(v['motion_root'],old['path'])
    if digest(path)!=old['sha256']:raise ValueError('Round 13 motion bytes changed')
    stored=parent_core.arrays(path,('values','valid'))
    if any(rebuilt[k].dtype!=stored[k].dtype or not np.array_equal(rebuilt[k],stored[k]) for k in ('values','valid')):
        raise ValueError('Rebuilt motion anchor differs from Round 13')
    v['protected'][str(path)]=old['sha256']
    return stored

def features(a,smoke=False,replay=False):
    v=context(a)
    if not smoke:
        s=read_json(safe_file(a.out,'smoke.json'))
        if s['signature']!=v['signature'] or s['status']!='family_smoke_passed':raise ValueError('Current training smoke required')
    records=sorted(v['train'],key=parent_core.game_play)[:32] if smoke else v['manifest']['files']
    if smoke and len(records)!=32:raise ValueError('Exactly 32 training plays required')
    counts=[];entries=[];reused=0
    for i,r in enumerate(records):
        dest=a.out/'features'/Path(r['parent_path']).name
        if replay or smoke or not dest.is_file():
            p=observed(a,r,v,labels=False);f=build(p,a.round_no);verify_anchor(a,r,p,v)
            if replay and not dest.is_file():raise ValueError('Replay refuses missing feature checkpoint')
            reused+=int(dest.is_file());h=checkpoint_npz(dest,{**f,'signature':np.array(v['signature'])})
        else:
            receipt=read_json(safe_file(dest.parent,dest.with_suffix('.json').name))
            if digest(dest)!=receipt['sha256']:raise ValueError('Feature checkpoint corrupted')
            f=parent_core.arrays(dest,('values','valid','signature'))
            if str(f['signature'])!=v['signature']:raise ValueError('Feature source signature changed')
            h=receipt['sha256'];reused+=1
            # Verify parents even when the completed numerical construction is reused.
            src=safe_file(a.tensors,r['parent_path'])
            if digest(src)!=r['parent_sha256']:raise ValueError('Parent changed during cache reuse')
            v['protected'][str(src)]=r['parent_sha256']
            anchor=v['motion_lookup'][r['parent_path']];ap=safe_file(v['motion_root'],anchor['path'])
            if digest(ap)!=anchor['sha256']:raise ValueError('Motion anchor drift')
            v['protected'][str(ap)]=anchor['sha256']
        if r['train']:counts.append(f)
        entries.append({'path':str(dest.relative_to(a.out)),'sha256':h,**{k:r[k] for k in ('parent_path','parent_sha256','train','rows')}})
        if i==0 or (i+1)%64==0 or i+1==len(records):
            print(json.dumps({'event':'family_features','round':a.round_no,'plays':i+1,'total':len(records),'reused':reused}),flush=True)
    report={'status':'family_smoke_passed' if smoke else 'family_features_ready','round':a.round_no,'signature':v['signature'],
        'plays':len(records),'statistics':summarize(counts,a.round_no),'candidate_channels':12,
        'statistics_scope':'training-only; denominator includes padded frame slots','target_arrays_unpacked':False,
        'raw_csv_reads':0,'new_model_fits':0,'motion_anchor_matches_round13_exactly':True}
    name='smoke.json' if smoke else 'preparation.json'
    if replay:
        if read_json(safe_file(a.out,name))!=report:raise ValueError('Feature replay summary changed')
    else:seal_json(a.out/name,report)
    if not smoke and not replay:seal_json(a.out/'dataset.json',{'signature':v['signature'],'source':v['source'],'round':a.round_no,'files':entries})
    atomic_json(a.out/'feature_execution.json',{'round':a.round_no,'reused':reused,'constructed':len(records)-reused,'replay':replay})
    unchanged(v);return report

def load_split(a,training):
    v=context(a);m=read_json(safe_file(a.out,'dataset.json'))
    if m['signature']!=v['signature'] or m['source']!=code_signature(a.kit):raise ValueError('Prepared dataset identity changed')
    lookup={r['parent_path']:r for r in m['files']};records=v['manifest']['files'];plays=[];total=0
    if set(lookup)!={r['parent_path'] for r in records}:raise ValueError('Prepared play set changed')
    for r in records:
        if r['train']!=training:continue
        p=observed(a,r,v,labels=True,allow_eval=not training);e=lookup[r['parent_path']];path=safe_file(a.out,e['path'])
        if digest(path)!=e['sha256']:raise ValueError('Prepared feature checksum changed')
        f=parent_core.arrays(path,('values','valid','signature'))
        if str(f['signature'])!=v['signature']:raise ValueError('Prepared feature lineage changed')
        anchor=verify_anchor(a,r,p,v)
        p.update(extra=f['values'],extra_valid=f['valid'],motion=anchor['values'],motion_valid=anchor['valid']);total+=sum(x.nbytes for x in p.values())
        if total>1024**3:raise MemoryError('One-GiB split memory cap reached')
        plays.append(p)
    keys=np.concatenate([p['keys'] for p in plays]);expected=v['contract']['training_rows' if training else 'evaluation_rows']
    if len(keys)!=expected or len(np.unique(keys,axis=0))!=len(keys):raise ValueError('Forecast population drift')
    unchanged(v);return plays,v

def evaluation_data(a):
    p,v=load_split(a,False);path=safe_file(a.tensors,'reference.npz')
    if digest(path)!=v['manifest']['reference_sha256']:raise ValueError('Original reference checksum changed')
    ref=parent_core.arrays(path,('keys','truth','pred'))
    if not np.array_equal(ref['keys'],np.concatenate([z['keys'] for z in p])) or not np.array_equal(ref['truth'],np.concatenate([z['y'] for z in p])):
        raise ValueError('Reference/prediction row alignment failed')
    return p,ref
