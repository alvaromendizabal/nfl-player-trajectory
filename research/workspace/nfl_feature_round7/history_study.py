"""Attributable earlier-game response experiment with immutable control reuse."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from origin_features import BASE_NAMES, rmse
from parent_support import digest, hash_json, safe_file, read_npz, atomic_json, atomic_npz, seal_json
from parent_bridge import parent, read, indices, replay_parent
from history_features import (CONFIG, NAMES, SUPPORT_NAMES, from_arrays, fit_ordered,
                              transform_frozen, state_hash, Metadata, support_summary)

CODE = ('history_features.py', 'history_study.py', 'parent_bridge.py', 'origin_features.py',
        'parent_support.py', 'worker.py', 'run_round.py', 'input_contract.json')
ARMS = ('control', 'support_only', 'role_response', 'player_response')
CONTRASTS = (('support_only','control'), ('role_response','control'),
             ('role_response','support_only'), ('player_response','control'),
             ('player_response','support_only'), ('player_response','role_response'))


def context(args):
    ctx = parent(args)  # SHA-verifies the source/imports and preserved original data.
    c = read(args.kit, 'input_contract.json')
    for name, sha in c['round6_report_hashes'].items():
        if digest(safe_file(args.round6, name)) != sha:
            raise ValueError('Reviewed Round 6 report drift: ' + name)
    r = read(args.round6, 'summary.json'); replay = read(args.round6, 'replay.json')
    if (r['status'] != 'pass_axis_study_complete'
            or r['signature'] != c['round6_signature']
            or r['terminal_earned_further_testing'] or r['history_earned_further_testing']
            or replay['status'] != 'pass_axis_replay_exact'
            or replay['new_coordinate_models'] != 0):
        raise ValueError('Expected completed negative Round 6 with no-refit replay')
    ctx.update(kit=args.kit, contract7=c, metadata=from_arrays(ctx['data']['keys'],
                    ctx['data']['role'],ctx['data']['X']), round6_summary=r)
    return ctx


def signature(ctx):
    return hash_json({'study':ctx['contract7']['study'],
                      'code':{n:digest(safe_file(ctx['kit'],n)) for n in CODE},
                      'parent_experiment':ctx['r5_experiment'],
                      'parent_data':ctx['source']['sha256'], 'history_config':CONFIG})


def feature_input_hash(ctx, fold):
    tr, va = indices(ctx, fold)
    return hash_json({'source_data':ctx['source']['sha256'], 'fold':ctx['source']['folds'][fold-1],
                      'train_indices':tr.tolist(), 'evaluation_indices':va.tolist()})


def feature_paths(out, fold):
    return (out/f'features/fold_{fold}.npz', out/f'tables/fold_{fold}.json',
            out/f'features/fold_{fold}.receipt.json')


def load_features(ctx,out,sig,fold):
    p,t,rp=feature_paths(out,fold)
    rec=read(rp.parent,rp.name)
    for path in (p,t):safe_file(path.parent,path.name)
    if (rec['signature']!=sig or rec['input_signature']!=feature_input_hash(ctx,fold)
            or rec['sha256']!=digest(p) or rec['table_sha256']!=digest(t)):
        raise ValueError('Ordered history feature/table identity changed')
    z=read_npz(p);table=read(t.parent,t.name);tr,va=indices(ctx,fold)
    for label, ix in [('train',tr),('evaluation',va)]:
        a=z[label]
        if (a.shape!=(len(ix),len(NAMES)) or a.dtype!=np.float64 or not np.isfinite(a).all()
                or not np.array_equal(z[label+'_indices'],ix)
                or not np.array_equal(z[label+'_keys'],ctx['data']['keys'][ix])):
            raise ValueError('Ordered history shape/dtype/key/index changed')
    pred=transform_frozen(ctx['metadata'].take(va),table)
    if not np.array_equal(pred,z['evaluation']):
        raise ValueError('Frozen history table cannot replay evaluation features')
    return z,table,rec


def build_fold(ctx,out,sig,fold,*,replay_only=False):
    from tree_core import event
    tr,va=indices(ctx,fold);p,t,rp=feature_paths(out,fold)
    if rp.exists() and not replay_only:
        _,_,r=load_features(ctx,out,sig,fold)
        return {**r['diagnostics'], 'fold':fold, 'reused_complete':True,
                'new_feature_fit':False}
    if replay_only and not rp.is_file():
        raise ValueError('Replay cannot create a missing historical encoder')
    counter=[0]
    def progress(r):
        counter[0]+=1
        if counter[0] % 8==0 or r['completed_dates']==r['total_dates']:
            event('history_date_progress',fold=fold,**r)
    xt,xe,table,info=fit_ordered(ctx['metadata'].take(tr),ctx['data']['y'][tr],
                               ctx['metadata'].take(va),progress)
    arrays={'train':xt,'evaluation':xe,'train_indices':tr,'evaluation_indices':va,
            'train_keys':ctx['data']['keys'][tr],'evaluation_keys':ctx['data']['keys'][va]}
    if rp.exists():
        old,st,r=load_features(ctx,out,sig,fold)
        if (any(not np.array_equal(old[k],v) for k,v in arrays.items())
                or state_hash(st)!=state_hash(table)):
            raise ValueError('Exact chronological encoder reconstruction failed')
        return {**info,'fold':fold,'reused_complete':True,'new_feature_fit':False,
                'ordered_reconstruction_exact':True}
    # An uncommitted first-attempt file is usable only if its arrays match exactly.
    if p.exists():
        old=read_npz(safe_file(p.parent,p.name))
        if any(k not in old or not np.array_equal(old[k],v) for k,v in arrays.items()):
            raise ValueError('Partial feature file conflicts; preserve and inspect')
    else:atomic_npz(p,**arrays)
    seal_json(t,table)
    r={'signature':sig,'input_signature':feature_input_hash(ctx,fold),'sha256':digest(p),
       'table_sha256':digest(t),'names':list(NAMES),'diagnostics':info}
    atomic_json(rp,r);load_features(ctx,out,sig,fold)
    return {**info,'fold':fold,'reused_complete':False,'new_feature_fit':True}


def smoke(ctx,out,sig):
    from tree_core import event
    tr,_=indices(ctx,2);m=ctx['metadata'].take(tr);y=ctx['data']['y'][tr]
    plays=np.unique(m.keys[:,:2],axis=0)[:32]
    if len(plays)!=32:raise ValueError('Smoke needs 32 training plays')
    wanted=set(map(tuple,plays));ix=np.array([i for i,k in enumerate(m.keys[:,:2]) if tuple(k) in wanted])
    m=m.take(ix);y=y[ix];empty=m.take(np.array([],dtype=int))
    a,_,st,diag=fit_ordered(m,y,empty)
    dates=m.keys[:,0]//100;first=dates.min();poison=y.copy();poison[dates==first]+=12345
    b,_,_,_=fit_ordered(m,poison,empty)
    if not np.array_equal(a[dates==first],b[dates==first]):
        raise ValueError('Current-date target contamination found')
    poison=y.copy();poison[dates==dates.max()]*=-54321
    c,_,_,_=fit_ordered(m,poison,empty)
    if not np.array_equal(a,c):raise ValueError('Future/latest-date target affected earlier features')
    result={'status':'history_smoke_passed','signature':sig,'plays':32,**diag,
            'same_day_poison_test_passed':True,'future_poison_test_passed':True,
            'raw_csvs_read':0,'new_model_fits':0,'feature_research':'open',
            'table_sha256':state_hash(st)}
    atomic_json(out/('smoke_reuse.json' if (out/'smoke.json').exists() else 'smoke.json'),result)
    event('history_smoke',plays=32,status=result['status']);return result


def prepare(ctx,out,sig):
    results=[build_fold(ctx,out,sig,f) for f in (2,3)]
    report={'status':'history_features_ready','signature':sig,'folds':results,
            'new_coordinate_models':0,'raw_csvs_read':0,'validation_targets_in_encoder':False,
            'feature_research':'open'}
    name='preparation_reuse.json' if (out/'preparation.json').exists() else 'preparation.json'
    atomic_json(out/name,report);return report


def matrices(data,features):
    base=data[:,:len(BASE_NAMES)]
    return {'support_only':np.c_[base,features[:,:9]],
            'role_response':np.c_[base,features[:,:15]],
            'player_response':np.c_[base,features]}


def compare(y,a,b,games):
    unique,inv=np.unique(games,return_inverse=True)
    if len(unique)<2:raise ValueError('Multiple games required for paired intervals')
    n=np.bincount(inv);sa=np.bincount(inv,weights=np.square(y-a).sum(1));sb=np.bincount(inv,weights=np.square(y-b).sum(1))
    if sa.sum()<=0:raise ValueError('Nonpositive reference squared error')
    rng=np.random.default_rng(20260911);ds=[]
    for _ in range(20):
        ix=rng.integers(0,len(n),(500,len(n)))
        ds.extend((np.sqrt(sb[ix].sum(1)/(2*n[ix].sum(1)))-np.sqrt(sa[ix].sum(1)/(2*n[ix].sum(1)))).tolist())
    lo,hi=np.quantile(ds,[.05/36,1-.05/36]);gain=1-rmse(y,b)/rmse(y,a)
    return {'delta_rmse':rmse(y,b)-rmse(y,a),'relative_gain':gain,
            'adjusted_low':float(lo),'adjusted_high':float(hi),'resamples':10000,
            'planned_comparison_looks':18,'screen_gate':bool(gain>=.01 and hi<0),
            'scope':'Exploratory reused-game interval; not global sequential error control'}


def protocol(ctx,out,sig):
    from tree_core import SETTINGS,versions
    rec={str(f):load_features(ctx,out,sig,f)[2] for f in (2,3)}
    return {'signature':sig,'history_feature_receipts':rec,'settings':SETTINGS,
            'environment':versions(),'arms':list(ARMS),'folds':[2,3],
            'contrasts':[list(p) for p in CONTRASTS],'comparison_looks':18,
            'parent_experiment':ctx['r5_experiment'],'historical_encoder_config':CONFIG,
            'maximum_new_coordinate_models':12}


def allowed_fold3(report):
    control=report['metrics']['control']
    return not all(report['metrics'][arm]>=control*1.05 for arm in ('role_response','player_response'))


def run_fold(ctx,out,sig,fold,*,replay_only=False):
    from tree_core import fit_axis,event
    if fold not in (2,3):raise ValueError('Only the two predefined later folds')
    spec=protocol(ctx,out,sig);esig=hash_json(spec)
    if replay_only and not (out/'experiment_protocol.json').is_file():raise ValueError('Missing fitted protocol')
    seal_json(out/'experiment_protocol.json',spec)
    if fold==3:
        prev=read(out,'fold_2/summary.json');rc=read(out,'fold_2/summary.receipt.json')
        if (prev['experiment_signature']!=esig or rc['sha256']!=digest(safe_file(out,'fold_2/summary.json'))):
            raise ValueError('Complete matching fold 2 first')
        if not allowed_fold3(prev):
            if replay_only:raise ValueError('Fold 3 stopped; replay cannot fit it')
            r={'status':'stopped_for_futility','signature':sig,'fold':3,'new_coordinate_models':0,
               'reason':'Both outcome-derived treatments at least 5% worse than the preserved control'}
            atomic_json(out/'fold_3/futility.json',r);return r
    z,_,_=load_features(ctx,out,sig,fold);tr,va=indices(ctx,fold);d=ctx['data']
    trainm=matrices(d['X'][tr],z['train']);evalm=matrices(d['X'][va],z['evaluation'])
    pred,parents=replay_parent(ctx,fold);fits=[];keeps={};start=time.monotonic()
    for arm,mt in trainm.items():
        keep=np.ptp(mt,axis=0)>1e-10;keeps[arm]=int(keep.sum())
        xt=np.ascontiguousarray(mt[:,keep],dtype=np.float64)
        xe=np.ascontiguousarray(evalm[arm][:,keep],dtype=np.float64);parts=[]
        for axis in range(2):
            folder=out/f'fold_{fold}/models'/arm/str(axis)
            if replay_only and not folder.is_dir():raise ValueError('Replay refuses missing model')
            axis_sig=hash_json({'study':esig,'fold':fold,'arm':arm,'axis':axis,'keep':keep.tolist()})
            p,r=fit_axis(folder,xt,d['y'][tr,axis],xe,axis_sig,replay_only=replay_only)
            parts.append(p);fits.append({'arm':arm,'axis':axis,**r})
        pred[arm]=np.column_stack(parts)
        pp=out/f'fold_{fold}/predictions/{arm}.npz';rp=pp.with_suffix('.json')
        if pp.exists() or rp.exists():
            r=read(rp.parent,rp.name);old=read_npz(safe_file(pp.parent,pp.name))
            if r['signature']!=esig or r['sha256']!=digest(pp):raise ValueError('Forecast receipt drift')
            if any(not np.array_equal(a,b) for a,b in ((pred[arm],old['pred']),(d['keys'][va],old['keys']),(d['y'][va],old['truth']))):
                raise ValueError('Forecast/key/target replay mismatch')
        elif replay_only:raise ValueError('Replay refuses missing saved forecasts')
        else:
            atomic_npz(pp,pred=pred[arm],keys=d['keys'][va],truth=d['y'][va]);atomic_json(rp,{'signature':esig,'sha256':digest(pp)})
    contrasts=[{'treatment':b,'control':a,**compare(d['y'][va],pred[a],pred[b],d['keys'][va,0])} for b,a in CONTRASTS]
    slices=ctx['confirmation'].slice_statistics(d['y'][va],pred,d['keys'][va],d['role'][va])
    for arm,p in pred.items():
        for label,mask in [('player_cold',z['evaluation'][:,7]==1),('player_warm',z['evaluation'][:,7]==0)]:
            if mask.any():slices.append({'arm':arm,'slice':label,'rows':int(mask.sum()),'rmse':rmse(d['y'][va][mask],p[mask]),'sse':float(np.square(d['y'][va][mask]-p[mask]).sum())})
    result={'status':'history_fold_complete','signature':sig,'experiment_signature':esig,'fold':fold,
            'metrics':{a:rmse(d['y'][va],p) for a,p in pred.items()},'contrasts':contrasts,'slices':slices,
            'training_rows':len(tr),'evaluation_rows':len(va),'evaluation_games':len(np.unique(d['keys'][va,0])),
            'retained_columns':keeps,'fits':fits,'parent_replay':parents,'old_models_refitted':0,
            'new_coordinate_models':sum(not r['reused_complete'] for r in fits),
            'all_prediction_replays_exact':True,'github_updated':False,'new_kaggle_score':None,
            'feature_research':'open','elapsed_seconds':round(time.monotonic()-start,3)}
    name='replay.json' if replay_only else 'summary.json'
    if (out/f'fold_{fold}/summary.json').exists():
        old=read(out,f'fold_{fold}/summary.json')
        for k in ('signature','experiment_signature','metrics','contrasts','slices','retained_columns'):
            if old[k]!=result[k]:raise ValueError('Scientific result changed on replay')
        if not replay_only:name='reuse.json'
    atomic_json(out/f'fold_{fold}'/name,result)
    if name=='summary.json':atomic_json(out/f'fold_{fold}/summary.receipt.json',{'sha256':digest(out/f'fold_{fold}/summary.json'),'experiment_signature':esig})
    event('history_fold_complete',fold=fold,metrics=result['metrics'],new_coordinate_models=result['new_coordinate_models'])
    return result


def summarize(ctx,out,sig):
    esig=hash_json(protocol(ctx,out,sig));reports=[];packs=[]
    for fold in (2,3):
        if not (out/f'fold_{fold}/summary.json').is_file():continue
        r=read(out,f'fold_{fold}/summary.json');rc=read(out,f'fold_{fold}/summary.receipt.json')
        if (r['signature']!=sig or r['experiment_signature']!=esig
                or digest(safe_file(out,f'fold_{fold}/summary.json'))!=rc['sha256']):raise ValueError('Pooled fold identity mismatch')
        _,va=indices(ctx,fold);p,_=replay_parent(ctx,fold)
        for arm in ARMS[1:]:
            f=safe_file(out,f'fold_{fold}/predictions/{arm}.npz');rr=read(f.parent,arm+'.json');z=read_npz(f)
            if rr['signature']!=esig or rr['sha256']!=digest(f):raise ValueError('Pooled forecast hash mismatch')
            if not np.array_equal(z['keys'],ctx['data']['keys'][va]) or not np.array_equal(z['truth'],ctx['data']['y'][va]):raise ValueError('Pooled keys/targets differ')
            p[arm]=z['pred']
        reports.append(r);packs.append({'keys':ctx['data']['keys'][va],'y':ctx['data']['y'][va],**p})
    if not packs:raise ValueError('No completed fold to summarize')
    joint={k:np.concatenate([p[k] for p in packs]) for k in packs[0]}
    if len(np.unique(joint['keys'],axis=0))!=len(joint['keys']):raise ValueError('Overlapping evaluation rows')
    game_sets=[set(p['keys'][:,0]) for p in packs]
    if sum(map(len,game_sets))!=len(set.union(*game_sets)):raise ValueError('Overlapping evaluation games')
    cs=[]
    for b,a in CONTRASTS:
        comp=compare(joint['y'],joint[a],joint[b],joint['keys'][:,0])
        improves=all(r['metrics'][b]<r['metrics'][a] for r in reports)
        cs.append({'treatment':b,'control':a,**comp,'both_folds_improve':improves if len(reports)==2 else None,
                   'replication_gate':bool(len(reports)==2 and improves and comp['screen_gate'])})
    passed={(r['treatment'],r['control']):r['replication_gate'] for r in cs}
    r={'status':'history_study_complete' if len(reports)==2 else 'history_study_partial',
       'signature':sig,'folds':[r['fold'] for r in reports],'evaluation_rows':len(joint['y']),
       'evaluation_games':len(np.unique(joint['keys'][:,0])),
       'pooled_metrics':{a:rmse(joint['y'],joint[a]) for a in ARMS},'contrasts':cs,
       'fold_metrics':[{'fold':r['fold'],'metrics':r['metrics']} for r in reports],
       'role_history_earned_further_testing':all(passed[('role_response',a)] for a in ('control','support_only')),
       'player_history_earned_further_testing':all(passed[('player_response',a)] for a in ('control','support_only','role_response')),
       'new_coordinate_models_recorded':sum(r['new_coordinate_models'] for r in reports),'old_models_refitted':0,
       'feature_research':'open','github_updated':False,'new_kaggle_score':None,
       'scope':'Same selected plays and reused game splits; no independent or cross-season confirmation'}
    atomic_json(out/'summary.json',r);return r
