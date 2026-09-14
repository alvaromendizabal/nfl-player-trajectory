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
"""Inspect frozen weights on training inputs only. No training command exists."""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
import numpy as np
from audit_io import (hash_json,digest,read_json,atomic_json,seal_json,checkpoint_npz,safe_file)
from parent_audit import verify_parent,assert_unchanged,observed_plays
from goal_frame import build,NAMES,GROUPS


def event(name,**fields):
    from datetime import datetime,timezone
    print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'event':name,**fields},allow_nan=False),flush=True)


def signature(kit,contract):
    return hash_json({'contract':contract,'source':{p.name:digest(p) for p in sorted(kit.glob('*.py'))}})


def preflight(parent,kit8,repo,kit,out,contract):
    v=verify_parent(parent,kit8,contract,repo)
    result={'status':'signal_preflight_passed','audit_signature':signature(kit,contract),
            'selected_training_plays':32,'selected_training_rows':sum(r['rows'] for r in v['selected']),
            'protected_files':len(v['protected']),'protected_file_fingerprint':hash_json(v['protected']),'parent_completion':'matched_encoder_study_complete',
            'old_offline_error_is_historical':True,'scientific_fits':0,'optimizer_steps':0,
            'evaluation_inputs_opened':False,'feature_research':'open','github_updated':False}
    assert_unchanged(v['protected']);seal_json(out/'preflight.json',result)
    event('preflight_complete',status=result['status'])
    return v


def feature_smoke(parent,kit,out,contract,v):
    sig=signature(kit,contract);total=np.zeros(6,dtype=np.int64);square=np.zeros(6)
    maximum=np.zeros(6);group=np.zeros(4,dtype=np.int64);rows=0
    for i,(r,p) in enumerate(observed_plays(parent,v),1):
        result=build(p);values=result['values'].astype(float);valid=result['valid']
        total+=valid.sum(axis=(0,1,2));square+=(values**2).sum(axis=(0,1,2))
        maximum=np.maximum(maximum,np.max(np.abs(values),axis=(0,1,2)))
        group+=result['groups'].sum(axis=(0,1,2));rows+=len(p['base'])
        checkpoint_npz(out/'goal_frame'/Path(r['path']).name,{**result,'audit_signature':np.array(sig)})
        if i%8==0:event('goal_frame_progress',plays=i,total=32)
    summary={'status':'goal_frame_smoke_passed','audit_signature':sig,'plays':32,'training_rows':rows,
             'names':list(NAMES),'valid_counts':total.tolist(),
             'rms_scaled_values':np.sqrt(square/np.maximum(total,1)).tolist(),'max_abs_scaled_values':maximum.tolist(),
             'group_names':list(GROUPS),'group_pair_frames':group.tolist(),
             'new_numeric_channels':6,'group_partitions':4,'model_integration':False,
             'predictive_gain_measured':False,'scientific_fits':0,'optimizer_steps':0,
             'future_coordinate_arrays_unpacked':False,'evaluation_inputs_opened':False}
    assert_unchanged(v['protected']);seal_json(out/'feature_smoke.json',summary)
    event('feature_smoke_complete',status=summary['status']);return summary


def combine(records,names):
    differences=np.zeros(len(names));valid=np.zeros(len(names));different=np.zeros(len(names))
    arms={}
    for record in records:
        ic=record['input_contrast'];differences+=ic['sum_squared_difference'];valid+=ic['valid_counts'];different+=ic['different_counts']
    for arm in ('terminal','history'):
        rs=[r['arms'][arm] for r in records];changes=[]
        for variant in sorted(rs[0]['changes']):
            n=sum(r['changes'][variant]['coordinate_count'] for r in rs)
            sq=sum(r['changes'][variant]['sum_squared_change'] for r in rs)
            changes.append({'variant':variant,'coordinate_count':n,'prediction_change_rms_yards':float(np.sqrt(sq/n)),
                            'max_absolute_change_yards':max(r['changes'][variant]['max_absolute_change'] for r in rs)})
        energy=np.array([r['gradient_energy'] for r in rs]).sum(0)
        vc=np.array([r['valid_counts'] for r in rs]).sum(0)
        channel=[]
        for c,name in enumerate(names):
            sq=sum(r['channel_zeroing'][c]['sum_squared_change'] for r in rs)
            n=sum(r['channel_zeroing'][c]['coordinate_count'] for r in rs)
            channel.append({'name':name,'prediction_change_rms_yards':float(np.sqrt(sq/n)),
                            'signed_projection_gradient_rms':float(np.sqrt(energy[c]/max(vc[c],1))),
                            'valid_elements':int(vc[c])})
        arms[arm]={'prediction_changes':changes,'channels':channel}
    return {'pair_channels':list(names),'input_rms_scaled_difference':np.sqrt(differences/np.maximum(valid,1)).tolist(),
            'different_fraction':(different/np.maximum(valid,1)).tolist(),'valid_counts':valid.astype(int).tolist(),'arms':arms}


def run_signal_audit(parent,kit,out,contract,v,*,replay=False,strict_environment=True):
    import torch
    from sequence_model import Forecast,setup,collate
    from sequence_features import terminal_view,PAIR_NAMES
    from signal_probe import probe,input_contrast,weights_digest,weight_changes
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
         'machine':platform.machine(),'threads':2,'device':'cpu'}
    if strict_environment and env!=v['protocol']['environment']:
        raise ValueError('Runtime differs from the completed Round 8 environment; do not reinstall automatically')
    sig=signature(kit,contract)
    expected=read_json(safe_file(out,'feature_smoke.json'))
    if expected['audit_signature']!=sig or expected['status']!='goal_frame_smoke_passed':
        raise ValueError('Complete goal-frame smoke first using unchanged source')
    setup();initial=Forecast()
    if weights_digest(initial)!=contract['initial_hash']:
        raise ValueError('Original architecture or seeded initialization no longer matches')
    models={};module_changes={}
    for arm in ('terminal','history'):
        ck=v['checkpoints'][arm]
        state=torch.load(ck['path'],map_location='cpu',weights_only=True)
        if state['signature']!=ck['signature'] or state['step']!=contract['steps'] or state['initial_hash']!=contract['initial_hash']:
            raise ValueError('Final weight state differs from reviewed study')
        model=Forecast();model.load_state_dict(state['model'],strict=True);model.eval()
        models[arm]=model;module_changes[arm]=weight_changes(initial,model)
        del state
    records=[]
    for i,(r,p) in enumerate(observed_plays(parent,v),1):
        path=out/'signal_checks'/Path(r['path']).with_suffix('.json').name
        if not replay and path.exists():
            record=read_json(path)
            receipt=read_json(safe_file(path.parent,path.stem+'.receipt.json'))
            if digest(path)!=receipt['sha256']:raise ValueError('Per-play audit checksum changed')
            if record['audit_signature']!=sig or record['input_sha256']!=r['sha256']:
                raise ValueError('Old per-play audit signature changed')
            reused=True
        else:
            record={'audit_signature':sig,'input_sha256':r['sha256'],
                    'input_contrast':input_contrast(p,terminal_view),
                    'arms':{arm:probe(model,p,arm,v['protocol']['normalization'],collate) for arm,model in models.items()}}
            if replay:
                if not path.exists() or read_json(path)!=record:
                    raise ValueError('Fresh-process signal replay is not exact')
            else:
                seal_json(path,record)
                seal_json(path.with_name(path.stem+'.receipt.json'),{'sha256':digest(path)})
            reused=False
        records.append(record)
        if i%8==0:event('signal_progress',plays=i,total=32,reused_last=reused,replay=replay)
    result={'status':'signal_audit_complete','audit_signature':sig,**combine(records,PAIR_NAMES),
            'module_weight_changes':module_changes,'training_plays':32,
            'training_rows':sum(x['arms']['history']['rows'] for x in records),
            'model_checkpoints_read':2,'new_scientific_models':0,'new_optimizer_steps':0,
            'evaluation_inputs_opened':False,'future_coordinate_arrays_unpacked':False,
            'new_official_rmse':None,'feature_gain_established':False,
            'interpretation':'Training-input sensitivity only. Interventions can be off-distribution. Gradients are not feature-value evidence.',
            'environment':env,'feature_research':'open','github_updated':False}
    assert_unchanged(v['protected'])
    if replay:
        if read_json(safe_file(out,'signal_summary.json'))!=result:
            raise ValueError('Signal aggregate replay differs')
        check={'status':'signal_audit_replay_exact','plays_recomputed':32,'new_optimizer_steps':0,
               'summary_sha256':digest(out/'signal_summary.json'),'protected_files_unchanged':len(v['protected'])}
        seal_json(out/'replay.json',check);return check
    seal_json(out/'signal_summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','feature-smoke','audit','replay'])
    for name in ('parent','parent-kit','repo','kit','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();c=read_json(a.kit/'input_contract.json')
    v=preflight(a.parent,a.parent_kit,a.repo,a.kit,a.out,c)
    if a.stage=='feature-smoke':r=feature_smoke(a.parent,a.kit,a.out,c,v)
    elif a.stage in ('audit','replay'):
        r=run_signal_audit(a.parent,a.kit,a.out,c,v,replay=a.stage=='replay')
    else:r=read_json(a.out/'preflight.json')
    event('stage_complete',stage=a.stage,status=r['status'])

if __name__=='__main__':main()
