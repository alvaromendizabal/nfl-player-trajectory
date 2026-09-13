"""Reviewed read-only bridge to the existing Round 5 controls and feature arrays.
No parent fitting or source mutation is invoked.
"""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from origin_features import rmse
from parent_support import digest, hash_json, safe_file, read_npz

def read(root, name):
    return json.loads(safe_file(Path(root),name).read_text())

def parent(args):
    """Verify source before importing it; never call the parent's fitting runner."""
    c = read(args.kit, 'input_contract.json')
    for name, sha in c['round5_source_hashes'].items():
        if digest(safe_file(args.round5_kit,name)) != sha:
            raise ValueError('Round 5 source changed: '+name)
    for name, sha in c['round5_report_hashes'].items():
        if digest(safe_file(args.round5,name)) != sha:
            raise ValueError('Round 5 scientific receipt changed: '+name)
    # These common modules are included verbatim, not an alternative adapter.
    for name in ('origin_features.py','parent_support.py'):
        if digest(safe_file(args.kit,name)) != c['round5_source_hashes'][name]:
            raise ValueError('Bundled parent helper differs')
    sys.path.insert(0,str(args.round5_kit))
    conf = importlib.import_module('confirmation')
    if Path(conf.__file__).resolve() != (args.round5_kit/'confirmation.py').resolve():
        raise ValueError('Unexpected cached parent module; restart kernel/process')
    ctx=conf.context(args.round5_kit,args.round4,args.round4_kit,args.round3,args.round3_kit,args.repo)
    psig=conf.signature(ctx)
    if psig != c['round5_signature']:
        raise ValueError('Round 5 source/data identity mismatch')
    r=read(args.round5,'summary.json');rr=read(args.round5,'replay.json')
    if (r['status']!='temporal_replication_complete' or r['feature_replication_gate_passed']
        or rr['status']!='temporal_replay_exact' or rr['new_coordinate_models']!=0):
        raise ValueError('Expected completed negative Round 5 study with no-refit replay')
    if len(ctx['selection']['plays']) != c['selected_plays']:
        raise ValueError('Original play selection changed')
    data,fh=conf.assemble(ctx,args.round5,psig)
    protocol=conf.protocol(ctx,fh,psig)
    if read(args.round5,'experiment_protocol.json') != protocol:
        raise ValueError('Round 5 fitted protocol drift')
    ctx.update(data=data,r5_experiment=hash_json(protocol),r5_out=args.round5,r6_kit=args.kit,
               r6_contract=c,confirmation=conf,r6_repo=args.repo,round5_summary=r)
    return ctx

def indices(ctx,fold):
    return ctx['confirmation'].fold_indices(ctx['data'],ctx['source']['folds'][fold-1])

def replay_parent(ctx,fold,arms=('control',)):
    """Read-only replay; does not rewrite any parent receipt or checkpoint."""
    from tree_core import SETTINGS, versions, checked_blob
    tr,va=indices(ctx,fold);d=ctx['data'];conf=ctx['confirmation']
    report=read(ctx['r5_out'],f'fold_{fold}/summary.json')
    if report['experiment_signature']!=ctx['r5_experiment']:
        raise ValueError('Parent fold experiment mismatch')
    predictions={};receipts=[]
    for arm in arms:
        m=conf.matrices(d)[arm];keep=conf.training_mask(m,tr)
        xt=np.ascontiguousarray(m[tr][:,keep],dtype=np.float64)
        xe=np.ascontiguousarray(m[va][:,keep],dtype=np.float64)
        parts=[]
        for axis in range(2):
            folder=ctx['r5_out']/f'fold_{fold}/models'/arm/str(axis)
            r=read(folder,'checkpoint.json')
            a_sig=hash_json({'study':ctx['r5_experiment'],'fold':fold,'arm':arm,'axis':axis,'keep':keep.tolist()})
            expected=hash_json({'parent':a_sig,'settings':SETTINGS,'environment':versions()})
            if r['signature']!=expected or r['step']!=SETTINGS['max_iter']:
                raise ValueError('Parent model signature/exposure drift')
            safe_file(folder,r['blob'])
            model=checked_blob(folder,r)
            if model.n_iter_!=SETTINGS['max_iter'] or not np.array_equal(model.predict(xt[:64]),r['probe']):
                raise ValueError('Parent training-probe replay mismatch')
            parts.append(model.predict(xe));receipts.append({'fold':fold,'arm':arm,'axis':axis,'sha256':r['sha256']})
        p=np.column_stack(parts);path=safe_file(ctx['r5_out'],f'fold_{fold}/predictions/{arm}.npz')
        r=read(path.parent,arm+'.json')
        if r['signature']!=ctx['r5_experiment'] or r['sha256']!=digest(path):
            raise ValueError('Parent forecast signature/hash mismatch')
        z=read_npz(path)
        if any(not np.array_equal(a,b) for a,b in ((p,z['pred']),(d['keys'][va],z['keys']),(d['y'][va],z['truth']))):
            raise ValueError('Parent prediction/key/target replay failed')
        if abs(rmse(d['y'][va],p)-report['metrics'][arm])>1e-12:
            raise ValueError('Parent RMSE changed')
        predictions[arm]=p
    return predictions,receipts
