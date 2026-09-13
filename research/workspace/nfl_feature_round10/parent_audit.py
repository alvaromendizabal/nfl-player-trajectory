"""Authenticate the reviewed Round 8 source/receipts and select training inputs.

Never calls the prior study runner, optimizer, or evaluator. No raw CSV access.
This verifies checkpoint self-hashes against their local receipts; the uploaded
aggregate report does not independently contain the final weight-blob hashes.
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from audit_io import digest,hash_json,safe_file,read_json,read_observed


def verify_parent(parent: Path, kit8: Path, contract: dict, repo: Path | None=None) -> dict:
    protected={}
    def retain(path):
        h=digest(path); protected[str(path)]=h; return h
    actual={p.name for p in kit8.glob('*.py')}
    if actual != set(contract['round8_python_hashes']):
        raise ValueError('Round 8 Python file set changed; preserve it and stop')
    for name,h in contract['round8_python_hashes'].items():
        if retain(safe_file(kit8,name))!=h:
            raise ValueError('Round 8 source changed: '+name)
    if retain(safe_file(kit8,'input_contract.json'))!=contract['round8_contract_hash']:
        raise ValueError('Round 8 contract changed')
    if repo is not None:
        head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
        if head!=contract['repo_head']:
            raise ValueError('Workspace revision differs from reviewed source; no automatic pull')
    for name,h in contract['report_hashes'].items():
        if retain(safe_file(parent,name))!=h:
            raise ValueError('Reviewed result changed: '+name)
    protocol_path=safe_file(parent,'neural_protocol.json'); retain(protocol_path)
    p=read_json(protocol_path)
    if hash_json(p)!=contract['protocol_signature']:
        raise ValueError('Round 8 protocol does not match the uploaded report')
    manifest_path=safe_file(parent,'dataset.json')
    if retain(manifest_path)!=p['data_sha256']:
        raise ValueError('Dataset manifest differs from the sealed neural protocol')
    m=read_json(manifest_path)
    code=hash_json(dict(contract['round8_python_hashes'],contract=contract['round8_contract_hash']))
    if m['code']!=code or m['signature']!=contract['data_signature'] or m['fold']!=2:
        raise ValueError('Unexpected data/source/fold identity')
    if len({x['path'] for x in m['files']})!=len(m['files']):
        raise ValueError('Duplicate dataset entry')
    if any(type(x['train']) is not bool for x in m['files']):
        raise ValueError('Split markers must be boolean')
    def order(r):
        parts=Path(r['path']).stem.split('_')
        if len(parts)!=2 or not all(s.isdigit() for s in parts):
            raise ValueError('Expected canonical game_play filename')
        return tuple(map(int,parts))
    train=sorted((r for r in m['files'] if r['train']),key=order)
    if len(train)<32:
        raise ValueError('Fewer than 32 training plays in frozen dataset')
    chosen=train[:32]
    for r in chosen:
        if retain(safe_file(parent,r['path']))!=r['sha256']:
            raise ValueError('Training feature checksum failed')
    checkpoints={}
    for arm in ('terminal','history'):
        cp=safe_file(parent,f'models/{arm}/checkpoint.json'); retain(cp); r=read_json(cp)
        path=safe_file(parent/f'models/{arm}',r['blob'])
        if r['step']!=contract['steps'] or r['signature']!=hash_json({'protocol':contract['protocol_signature'],'arm':arm}):
            raise ValueError('Checkpoint cursor or arm identity mismatch')
        if path.stat().st_size!=r['bytes'] or retain(path)!=r['sha256']:
            raise ValueError('Checkpoint size/checksum mismatch')
        if r['initial_hash']!=contract['initial_hash']:
            raise ValueError('Checkpoint initialization mismatch')
        checkpoints[arm]={'path':str(path),'signature':r['signature'],'sha256':r['sha256']}
    return {'protocol':p,'manifest':m,'selected':chosen,'checkpoints':checkpoints,'protected':protected}


def assert_unchanged(protected: dict):
    for name,h in protected.items():
        if digest(safe_file(Path(name).parent,Path(name).name))!=h:
            raise ValueError('Protected source/artifact changed: '+Path(name).name)


def observed_plays(parent,verified):
    for record in verified['selected']:
        p=read_observed(safe_file(parent,record['path']))
        if str(p['signature'])!=verified['manifest']['signature'] or len(p['base'])!=record['rows']:
            raise ValueError('Training input lineage/row count changed')
        yield record,p
