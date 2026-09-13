"""Read-only parent verification and bounded training-only feature/coverage audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
import numpy as np
from audit_io import digest, hash_json, safe_file, read_json, atomic_json, seal_json, checkpoint_npz
from new_features import build, summarize, NAMES

OBS=('ids','node','node_valid','pair','pair_valid','pair_age','role','side','node_age','query','base','train','signature')
EXT=('goal','goal_valid','goal_age','groups')

def signature(kit):
    files=sorted([*kit.glob('*.py'),kit/'input_contract.json',*(kit/'tests').rglob('*.py')])
    return hash_json({str(p.relative_to(kit)):digest(p) for p in files})

def arrays(path,fields):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or sum(q.file_size for q in z.infolist())>64*1024**2:
            raise ValueError('Unsafe tensor archive')
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in fields}

def game_play(r):
    text=Path(r['parent_path']).stem
    if not re.fullmatch(r'\d+_\d+',text):raise ValueError('Invalid game/play path')
    return tuple(map(int,text.split('_')))

def unchanged(protected):
    for name,sha in protected.items():
        p=Path(name)
        if digest(safe_file(p.parent,p.name))!=sha:raise ValueError('Protected file changed: '+p.name)

def verify(a):
    c=read_json(safe_file(a.kit,'input_contract.json'));prot={}
    def check(root,name,expected=None):
        p=safe_file(root,name);sha=digest(p)
        if expected is not None and sha!=expected:raise ValueError('Reviewed file differs: '+name)
        prot[str(p)]=sha;return p
    for n,h in c['parent_source_files'].items():check(a.parent_kit,n,h)
    actual=sorted([*a.parent_kit.glob('*.py'),a.parent_kit/'input_contract.json',*(a.parent_kit/'tests').rglob('*.py')])
    if {str(p.relative_to(a.parent_kit)) for p in actual}!=set(c['parent_source_files']):
        raise ValueError('Round 10 source set changed')
    for n,h in c['frozen_modules'].items():check(a.kit,n,h)
    head=subprocess.run(['git','-C',str(a.repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if head!=c['repo_head']:raise ValueError('Workspace commit differs; no automatic pull')
    for n,h in c['parent_reports'].items():check(a.parent,n,h)
    report=read_json(a.parent/'summary.json');replay=read_json(a.parent/'replay.json')
    if replay.get('status')!='grouped_goal_replay_exact' or replay.get('new_optimizer_steps')!=0 or replay['summary_sha256']!=digest(a.parent/'summary.json'):
        raise ValueError('Completed no-refit parent required')
    protocol=read_json(check(a.parent,'protocol.json'))
    if hash_json(protocol)!=c['protocol_signature']:raise ValueError('Parent protocol changed')
    m=read_json(check(a.parent,'dataset.json',protocol['data_sha256']))
    if m['source']!=c['parent_source_signature'] or m['signature']!=c['parent_data_signature'] or Path(m['parent'])!=a.tensors:
        raise ValueError('Parent source/tensor location changed')
    m8=read_json(check(a.tensors,'dataset.json',m['parent_manifest_sha256']))
    original={r['path']:r for r in m8['files']}
    if len(original)!=len(m8['files']):raise ValueError('Duplicate original tensor')
    if not all(type(r['train']) is bool for r in m['files']):raise ValueError('Boolean split required')
    if len({r['parent_path'] for r in m['files']})!=len(m['files']):raise ValueError('Duplicate parent record')
    for r in m['files']:
        old=original[r['parent_path']]
        if old['sha256']!=r['parent_sha256'] or old['train']!=r['train'] or old['rows']!=r['rows']:
            raise ValueError('Original tensor identity/split mismatch')
        game_play(r)
    tr=[r for r in m['files'] if r['train']];ev=[r for r in m['files'] if not r['train']]
    tg={game_play(r)[0] for r in tr};eg={game_play(r)[0] for r in ev}
    if not tr or not ev or tg&eg or max(tg)//100>=min(eg)//100:raise ValueError('Invalid chronological split')
    if sum(r['rows'] for r in tr)!=c['training_rows'] or sum(r['rows'] for r in ev)!=c['evaluation_rows']:
        raise ValueError('Row population changed')
    checkpoints={}
    for arm in ('cartesian','goal'):
        complete=read_json(a.parent/f'models/{arm}/complete.json')
        cr=read_json(check(a.parent,f'models/{arm}/checkpoint.json'))
        expected=hash_json({'protocol':c['protocol_signature'],'arm':arm})
        if cr['signature']!=expected or cr['step']!=complete['steps'] or cr['initial_hash']!=complete['initial_hash']:
            raise ValueError('Checkpoint identity/cursor mismatch')
        p=check(a.parent/f'models/{arm}',cr['blob'],cr['sha256'])
        if p.stat().st_size!=cr['bytes']:raise ValueError('Checkpoint length changed')
        checkpoints[arm]={'path':str(p),'receipt':cr,'complete':complete}
    sig=hash_json({'code':signature(a.kit),'protocol':c['protocol_signature'],'parent_dataset':digest(a.parent/'dataset.json')})
    return {'contract':c,'protected':prot,'protocol':protocol,'manifest':m,'train':tr,'evaluation_games':sorted(eg),
            'input_signature':m8['signature'],'training_games':sorted(tg),'summary':report,'checkpoints':checkpoints,'signature':sig}

def preflight(a):
    v=verify(a)
    r={'status':'training_review_preflight_passed','signature':v['signature'],
       'training_plays':len(v['train']),'training_games':len(v['training_games']),
       'training_rows':sum(t['rows'] for t in v['train']),
       'known_evaluation_rows':v['contract']['evaluation_rows'],'known_evaluation_games':len(v['evaluation_games']),
       'reported_feature_gate_passed':v['summary']['contrast']['feature_gate_passed'],
       'new_optimizer_steps':0,'evaluation_arrays_opened':False,'checkpoint_verification':'file hashes; not numerical replay',
       'source_commit':v['contract']['repo_head'],'feature_research':'open'}
    seal_json(a.out/'preflight.json',r);unchanged(v['protected']);return r

def context(a):
    v=verify(a)
    if read_json(safe_file(a.out,'preflight.json'))['signature']!=v['signature']:raise ValueError('Current preflight required')
    return v

def load_play(a,r,v,labels=False):
    if not r['train']:raise ValueError('This audit cannot load an evaluation play')
    p=safe_file(a.tensors,r['parent_path']);q=safe_file(a.parent,r['path'])
    if digest(p)!=r['parent_sha256'] or digest(q)!=r['sha256']:raise ValueError('Training tensor changed')
    fields=OBS+(('keys','y','role_query') if labels else ())
    z=arrays(p,fields)
    if not bool(z['train']) or len(z['base'])!=r['rows'] or str(z['signature'])!=v['input_signature']:raise ValueError('Training tensor split/rows mismatch')
    v['protected'][str(p)]=r['parent_sha256']
    # Goal extensions are needed only for frozen model replay, not new input construction.
    if labels:
        z.update(arrays(q,EXT));v['protected'][str(q)]=r['sha256']
        if z['keys'].shape!=(r['rows'],4) or z['y'].shape!=(r['rows'],2) or not np.isfinite(z['y']).all():
            raise ValueError('Invalid training targets/keys')
        if not np.all(z['keys'][:,:2]==game_play(r)):raise ValueError('Game/play key mismatch')
    return z

def features(a,smoke=False,replay=False):
    v=context(a)
    if not smoke:
        s=read_json(safe_file(a.out,'smoke.json'))
        if s['signature']!=v['signature'] or s['status']!='motion_receiver_smoke_passed':raise ValueError('Current smoke required')
    records=sorted(v['train'],key=game_play)[:32] if smoke else v['train']
    if smoke and len(records)!=32:raise ValueError('Need 32 training plays')
    all_arrays=[];receipts=[]
    for i,r in enumerate(records):
        out=a.out/'features'/Path(r['parent_path']).name
        p=load_play(a,r,v,labels=False);f=build(p)
        if replay and not out.is_file():raise ValueError('Replay refuses a missing feature checkpoint')
        sha=checkpoint_npz(out,f);all_arrays.append(f)
        receipts.append({'path':str(out.relative_to(a.out)),'sha256':sha,'parent_path':r['parent_path'],'parent_sha256':r['parent_sha256']})
        if i==0 or (i+1)%64==0 or i+1==len(records):print(json.dumps({'event':'observed_feature_progress','completed_plays':i+1,'total_plays':len(records)}),flush=True)
    result={'status':'motion_receiver_smoke_passed' if smoke else 'motion_receiver_features_ready','signature':v['signature'],
      'plays':len(records),'candidate_channels':12,'channel_statistics':summarize(all_arrays),
      'target_arrays_unpacked':False,'evaluation_arrays_opened':False,'optimizer_steps':0,
      'predictive_gain_measured':False,'support_denominator':'all 20 slots for all selected observed players; padding included',
      'existing_pipeline_information':'These concepts overlap prior tabular/repository features; not new raw signals.'}
    name='smoke.json' if smoke else 'feature_summary.json'
    if replay:
        if read_json(safe_file(a.out,name))!=result:raise ValueError('Feature summary replay differs')
    else:seal_json(a.out/name,result)
    if not smoke and not replay:seal_json(a.out/'feature_manifest.json',{'signature':v['signature'],'files':receipts})
    unchanged(v['protected']);return result

def coverage(a,replay=False):
    """Only identity columns are parsed. Count eligible plays in fixed training games."""
    import pandas as pd
    v=context(a);allowed=set(v['training_games']);observed={g:set() for g in allowed};selected={g:set() for g in allowed}
    for r in v['train']:
        g,p=game_play(r);selected[g].add(p)
    for i,e in enumerate(v['contract']['raw_inputs']):
        raw=safe_file(a.repo,e['path'])
        if raw.stat().st_size!=e['size'] or digest(raw)!=e['sha256']:raise ValueError('Raw input changed')
        cache=a.out/'coverage'/f'{i:02d}.json'
        if cache.exists():
            saved=read_json(cache)
            if read_json(safe_file(cache.parent,cache.name+'.receipt.json'))['sha256']!=digest(cache):raise ValueError('Coverage checksum mismatch')
            if saved['signature']!=v['signature'] or saved['raw_sha256']!=e['sha256']:raise ValueError('Coverage checkpoint drift')
        else:
            if replay:raise ValueError('Replay refuses a missing coverage checkpoint')
            pairs={g:set() for g in allowed}
            for chunk in pd.read_csv(raw,usecols=['game_id','play_id'],chunksize=100000):
                sub=chunk.loc[chunk.game_id.isin(allowed),['game_id','play_id']].drop_duplicates()
                for g,p in sub.itertuples(index=False,name=None):pairs[int(g)].add(int(p))
            saved={'signature':v['signature'],'raw_sha256':e['sha256'],'pairs':{str(g):sorted(x) for g,x in pairs.items()}}
            seal_json(cache,saved)
            seal_json(cache.with_name(cache.name+'.receipt.json'),{'sha256':digest(cache)})
        for gs,plays in saved['pairs'].items():
            g=int(gs)
            if g not in allowed:raise ValueError('Coverage cache contains non-training game')
            if observed[g]&set(plays):raise ValueError('A play appears in multiple input files')
            observed[g].update(plays)
        if digest(raw)!=e['sha256']:raise ValueError('Raw input mutated during audit')
        print(json.dumps({'event':'training_coverage_progress','input_files':i+1,'total_files':len(v['contract']['raw_inputs'])}),flush=True)
    if any(not selected[g].issubset(observed[g]) for g in allowed):raise ValueError('Selected training play absent from observed inventory')
    per=[{'game_id':g,'eligible_observed_plays':len(observed[g]),'selected_training_plays':len(selected[g]),'fraction':len(selected[g])/len(observed[g])} for g in sorted(allowed)]
    n=sum(x['eligible_observed_plays'] for x in per);k=sum(x['selected_training_plays'] for x in per)
    # Per-game identities stay in private checkpoint; return only date-free distribution.
    seal_json(a.out/'coverage_private.json',{'per_game':per,'signature':v['signature']})
    r={'status':'training_coverage_complete','signature':v['signature'],'training_games':len(allowed),
       'eligible_observed_plays':n,'selected_training_plays':k,'fraction':k/n,
       'per_game_fraction_quantiles':np.quantile([x['fraction'] for x in per],[0,.25,.5,.75,1]).tolist(),
       'input_columns_parsed':['game_id','play_id'],'raw_output_files_read':0,'new_training_selection_created':False,
       'scope':'Only existing training games counted. Input identity columns are scanned globally; no evaluation coordinates or labels are loaded.',
       'label_completeness_of_unused_plays':'not_verified','optimizer_steps':0}
    if replay and read_json(safe_file(a.out,'coverage_summary.json'))!=r:raise ValueError('Coverage summary drift')
    seal_json(a.out/'coverage_summary.json',r);unchanged(v['protected']);return r
