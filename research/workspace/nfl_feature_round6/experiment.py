"""One frozen, later-fold pass-axis feature comparison; immutable parent reuse."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from origin_features import BASE_NAMES, REQUIRED, build_view, query_features, rmse
from parent_support import digest, hash_json, safe_file, read_npz, atomic_json, atomic_npz, seal_json
from pass_features import build_features, STATIC_NAMES, HISTORY_NAMES

CODE = ('experiment.py','pass_features.py','origin_features.py','parent_support.py',
        'worker.py','run_round.py','input_contract.json')
ARMS = ('control', 'pass_axis', 'pass_axis_history')
CONTRASTS = (('pass_axis','control'), ('pass_axis_history','control'), ('pass_axis_history','pass_axis'))


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


def signature(ctx):
    return hash_json({'study':ctx['r6_contract']['study'],
        'code':{n:digest(safe_file(ctx['r6_kit'],n)) for n in CODE},
        'r5_experiment':ctx['r5_experiment'],'parent_data':ctx['source']['sha256']})


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


def checkpoint(out,item,sig,parent_sha):
    name=f"{item['game']}_{item['play']}.npz";p=out/'features'/name;rp=p.with_suffix('.json')
    if not p.exists() and not rp.exists():return None
    r=read(p.parent,rp.name)
    if r['signature']!=sig or r['parent_sha256']!=parent_sha or r['sha256']!=digest(safe_file(p.parent,p.name)):
        raise ValueError('Pass-axis feature checkpoint drift')
    z=read_npz(p)
    if (z['static'].shape!=(len(z['keys']),len(STATIC_NAMES)) or
        z['history'].shape!=(len(z['keys']),len(HISTORY_NAMES)) or
        any(not np.isfinite(z[k]).all() for k in ('static','history'))):
        raise ValueError('Invalid feature checkpoint')
    return z,{'name':name,'sha256':r['sha256'],'parent_sha256':parent_sha}


def prepare(ctx,out,sig,*,smoke=False):
    from tree_core import event
    from state_features import storage_parity
    conf=ctx['confirmation'];started=time.monotonic()
    selected=ctx['selection']['plays']
    train=set(ctx['source']['folds'][1]['train_games'])
    if smoke:selected=[p for p in selected if p['game'] in train][:32]
    if smoke and len(selected)!=32:raise ValueError('32 training plays required')
    pending=[];oldfiles={};reused=0
    for item in selected:
        name,p,h=conf.old_play(ctx,item);oldfiles[name]=(p,h)
        existing=checkpoint(out,item,sig,h)
        if existing is None or smoke:pending.append(item)
        if existing is not None:reused+=1
    specs={e['path']:e for e in ctx['contract']['raw_input_files']}
    created=0;rebuilt=0;parity_rows=0
    for name in sorted({p['input'] for p in pending}):
        spec=specs[name];path=safe_file(ctx['r6_repo'],name)
        if path.stat().st_size!=spec['size'] or digest(path)!=spec['sha256']:
            raise ValueError('Observed raw input changed')
        wanted={(p['game'],p['play']):p for p in pending if p['input']==name}
        cols=pd.read_csv(path,nrows=0).columns
        use=[c for c in cols if c in REQUIRED|{'s','dir','o','a'}]
        parts=[]
        for part in pd.read_csv(path,usecols=use,chunksize=50000):
            mask=pd.MultiIndex.from_frame(part[['game_id','play_id']]).isin(wanted)
            if mask.any():parts.append(part.loc[mask])
        if not parts:raise ValueError('Selected play absent')
        found=set()
        for key,raw in pd.concat(parts,ignore_index=True).groupby(['game_id','play_id'],sort=True):
            key=tuple(map(int,key));found.add(key);item=wanted[key]
            raw=raw.sort_values(['nfl_id','frame_id']).reset_index(drop=True)
            old,h=oldfiles[f'{key[0]}_{key[1]}.npz'];z=read_npz(old,('X','keys'))
            cutoff=int(raw.frame_id.max());view=build_view(raw,origin=cutoff,offset=0)
            rebuilt_legacy,_=query_features(view,z['keys'][:,2],z['keys'][:,3]/10)
            storage_parity(rebuilt_legacy,z['X']);parity_rows+=len(z['keys'])
            feat=build_features(raw,z['keys'],cutoff=cutoff)
            existing=checkpoint(out,item,sig,h)
            if existing is not None:
                if any(not np.array_equal(existing[0][k],feat[k]) for k in ('static','history','keys')):
                    raise ValueError('Feature raw-adapter replay differs')
            else:
                p=out/'features'/old.name
                atomic_npz(p,**feat)
                atomic_json(p.with_suffix('.json'),{'signature':sig,'sha256':digest(p),'parent_sha256':h})
                checkpoint(out,item,sig,h);created+=1
            rebuilt+=1
            if rebuilt%16==0:event('feature_progress',rebuilt_plays=rebuilt,total_to_build=len(pending),new_checkpoints=created)
        if found!=set(wanted) or digest(path)!=spec['sha256']:
            raise ValueError('Incomplete inputs or concurrent input writer')
    if rebuilt!=len(pending):raise ValueError('Incomplete preparation')
    entries=[];training_features=[]
    for item in selected:
        name,p,h=conf.old_play(ctx,item);got=checkpoint(out,item,sig,h)
        if got is None:raise ValueError('Prepared checkpoint missing')
        zz,e=got;entries.append(e)
        if item['game'] in train:
            _,ix=np.unique(zz['keys'][:,2],return_index=True)
            training_features.append(np.c_[zz['static'][ix],zz['history'][ix]])
    a=np.concatenate(training_features)
    support_indices=list(range(16,21))+[21+j*9+k for j in range(3) for k in (6,7,8)]
    result={'status':'pass_axis_smoke_passed' if smoke else 'pass_axis_features_ready',
        'signature':sig,'plays':len(selected),'new_checkpoints':created,'reused_checkpoints':reused,
        'raw_plays_rebuilt':rebuilt,'legacy_parity_rows':parity_rows,'raw_output_csvs_opened':0,
        'support_scope':'fold_2 training only, one row per player/play',
        'training_player_plays':len(a),'support_names':[(*STATIC_NAMES,*HISTORY_NAMES)[i] for i in support_indices],
        'support_fraction':a[:,support_indices].mean(0).tolist(),'new_model_fits':0,
        'feature_research':'open','elapsed_seconds':round(time.monotonic()-started,3)}
    if not smoke:
        manifest={'signature':sig,'entries':entries,'parent_data_sha256':ctx['source']['sha256']}
        seal_json(out/'feature_manifest.json',manifest)
        result['manifest_sha256']=digest(out/'feature_manifest.json')
    filename='smoke.json' if smoke else 'preparation.json'
    if (out/filename).exists():filename=filename.replace('.json','_reuse.json')
    atomic_json(out/filename,result);return result


def assemble(ctx,out,sig):
    conf=ctx['confirmation'];manifest=read(out,'feature_manifest.json');entries=[];chunks=[];keys=[]
    if manifest['signature']!=sig or manifest['parent_data_sha256']!=ctx['source']['sha256']:
        raise ValueError('Feature manifest lineage drift')
    for item in ctx['selection']['plays']:
        _,p,h=conf.old_play(ctx,item);got=checkpoint(out,item,sig,h)
        if got is None:raise ValueError('Feature preparation incomplete')
        z,e=got;entries.append(e);chunks.append(np.c_[z['static'],z['history']]);keys.append(z['keys'])
    if entries!=manifest['entries'] or not np.array_equal(np.concatenate(keys),ctx['data']['keys']):
        raise ValueError('Feature manifest or row alignment changed')
    if sum(a.nbytes for a in chunks)>128*1024**2:
        raise ValueError('Feature memory cap exceeded')
    return np.concatenate(chunks),hash_json(manifest)


def matrices(data,feat):
    base=data['X'][:,:len(BASE_NAMES)]
    return {'pass_axis':np.c_[base,feat[:,:len(STATIC_NAMES)]],
            'pass_axis_history':np.c_[base,feat]}


def compare(y,a,b,games):
    # The parent helper uses SIX looks; explicitly implement the new NINE looks.
    unique,inv=np.unique(games,return_inverse=True)
    if len(unique)<2:raise ValueError('At least two games required')
    n=np.bincount(inv);sa=np.bincount(inv,weights=np.square(y-a).sum(1));sb=np.bincount(inv,weights=np.square(y-b).sum(1))
    if sa.sum()<=0:raise ValueError('Nonpositive control error')
    rng=np.random.default_rng(20260911);ds=[]
    for _ in range(20):
        ix=rng.integers(0,len(n),(500,len(n)))
        ds.extend((np.sqrt(sb[ix].sum(1)/(2*n[ix].sum(1)))-np.sqrt(sa[ix].sum(1)/(2*n[ix].sum(1)))).tolist())
    lo,hi=np.quantile(ds,[.05/18,1-.05/18]);gain=1-rmse(y,b)/rmse(y,a)
    return {'delta_rmse':rmse(y,b)-rmse(y,a),'relative_gain':gain,'adjusted_low':float(lo),'adjusted_high':float(hi),
        'resamples':10000,'planned_comparison_looks':9,'screen_gate':bool(gain>=.01 and hi<0),
        'scope':'Exploratory reused-game interval; no independent-confirmation claim'}


def run_fold(ctx,out,sig,fold,*,replay_only=False):
    from tree_core import fit_axis,SETTINGS,versions,event
    if fold not in (2,3):raise ValueError('Only predeclared later folds 2 and 3')
    feat,fh=assemble(ctx,out,sig)
    protocol={'signature':sig,'features_hash':fh,'parent_data_sha256':ctx['source']['sha256'],
        'settings':SETTINGS,'environment':versions(),'arms':list(ARMS),'folds':[2,3],'comparison_looks':9}
    if replay_only and not (out/'experiment_protocol.json').is_file():
        raise ValueError('Replay needs the saved protocol')
    seal_json(out/'experiment_protocol.json',protocol);esig=hash_json(protocol)
    if fold==3:
        prior=read(out,'fold_2/summary.json')
        receipt=read(out,'fold_2/summary.receipt.json')
        if receipt['sha256']!=digest(safe_file(out,'fold_2/summary.json')):
            raise ValueError('Fold 2 receipt checksum drift')
        if prior['experiment_signature']!=esig:raise ValueError('Complete fold 2 first')
        losses=[c['relative_gain'] for c in prior['contrasts'] if c['control']=='control']
        if all(g<=-.05 for g in losses):
            if replay_only:raise ValueError('Fold 3 not fitted because both new arms failed cost guard')
            r={'status':'stopped_for_futility','fold':3,'signature':sig,'new_coordinate_models':0,
               'reason':'Both arms at least 5% worse on fold 2; no fold-3 fitting'}
            atomic_json(out/'fold_3/futility.json',r);return r
    tr,va=indices(ctx,fold);d=ctx['data'];conf=ctx['confirmation']
    predictions,parent_receipts=replay_parent(ctx,fold)
    fits=[];keeps={};start=time.monotonic()
    for arm,m in matrices(d,feat).items():
        keep=conf.training_mask(m,tr);keeps[arm]=int(keep.sum())
        xt=np.ascontiguousarray(m[tr][:,keep],dtype=np.float64);xe=np.ascontiguousarray(m[va][:,keep],dtype=np.float64)
        parts=[]
        for axis in range(2):
            folder=out/f'fold_{fold}/models'/arm/str(axis)
            if replay_only and not folder.is_dir():raise ValueError('Replay cannot fit a missing model')
            axis_sig=hash_json({'study':esig,'fold':fold,'arm':arm,'axis':axis,'keep':keep.tolist()})
            pred,receipt=fit_axis(folder,xt,d['y'][tr,axis],xe,axis_sig,replay_only=replay_only)
            parts.append(pred);fits.append({'arm':arm,'axis':axis,**receipt})
        predictions[arm]=np.column_stack(parts)
        pp=out/f'fold_{fold}/predictions'/f'{arm}.npz';rp=pp.with_suffix('.json')
        if pp.exists() or rp.exists():
            rec=read(rp.parent,rp.name)
            if rec['signature']!=esig or rec['sha256']!=digest(safe_file(pp.parent,pp.name)):
                raise ValueError('Prediction checksum/signature drift')
            old=read_npz(pp)
            if any(not np.array_equal(a,b) for a,b in ((predictions[arm],old['pred']),(d['keys'][va],old['keys']),(d['y'][va],old['truth']))):
                raise ValueError('Prediction/key/target replay mismatch')
        elif replay_only:raise ValueError('Replay cannot create missing saved predictions')
        else:
            atomic_npz(pp,pred=predictions[arm],keys=d['keys'][va],truth=d['y'][va]);atomic_json(rp,{'signature':esig,'sha256':digest(pp)})
    contrasts=[{'treatment':b,'control':a,**compare(d['y'][va],predictions[a],predictions[b],d['keys'][va,0])} for b,a in CONTRASTS]
    result={'status':'pass_axis_fold_complete','signature':sig,'experiment_signature':esig,'fold':fold,
        'metrics':{a:rmse(d['y'][va],p) for a,p in predictions.items()},'contrasts':contrasts,
        'training_rows':len(tr),'evaluation_rows':len(va),'evaluation_games':len(np.unique(d['keys'][va,0])),
        'slices':conf.slice_statistics(d['y'][va],predictions,d['keys'][va],d['role'][va]),
        'fits':fits,'retained_columns':keeps,'old_models_refitted':0,'parent_replay':parent_receipts,
        'new_coordinate_models':sum(not r['reused_complete'] for r in fits),'all_prediction_replays_exact':True,
        'feature_research':'open','github_updated':False,'new_kaggle_score':None,'elapsed_seconds':round(time.monotonic()-start,3)}
    name='replay.json' if replay_only else 'summary.json'
    if (out/f'fold_{fold}/summary.json').exists():
        old=read(out,f'fold_{fold}/summary.json')
        for k in ('signature','experiment_signature','metrics','contrasts','slices','retained_columns'):
            if old[k]!=result[k]:raise ValueError('Scientific result changed on replay')
        if not replay_only:name='reuse.json'
    atomic_json(out/f'fold_{fold}'/name,result)
    if name=='summary.json':atomic_json(out/f'fold_{fold}/summary.receipt.json',{'sha256':digest(out/f'fold_{fold}/summary.json'),'experiment_signature':esig})
    event('fold_complete',fold=fold,metrics=result['metrics'],new_models=result['new_coordinate_models'])
    return result


def summarize(ctx,out,sig):
    reports=[];packs=[]
    for fold in (2,3):
        path=out/f'fold_{fold}/summary.json'
        if not path.exists():continue
        report=read(path.parent,path.name);rec=read(path.parent,'summary.receipt.json')
        if digest(path)!=rec['sha256'] or report['signature']!=sig:raise ValueError('Fold summary identity changed')
        _,va=indices(ctx,fold);p,_=replay_parent(ctx,fold)
        for arm in ARMS[1:]:
            f=safe_file(out,f'fold_{fold}/predictions/{arm}.npz');r=read(f.parent,arm+'.json');z=read_npz(f)
            if r['signature']!=report['experiment_signature'] or r['sha256']!=digest(f):raise ValueError('Prediction manifest changed')
            if not np.array_equal(z['keys'],ctx['data']['keys'][va]) or not np.array_equal(z['truth'],ctx['data']['y'][va]):raise ValueError('Pooled row alignment changed')
            p[arm]=z['pred']
        reports.append(report);packs.append({'keys':ctx['data']['keys'][va],'y':ctx['data']['y'][va],**p})
    if not reports:raise ValueError('No completed fold; export failure report instead')
    joined={k:np.concatenate([p[k] for p in packs]) for k in packs[0]}
    if len(np.unique(joined['keys'],axis=0))!=len(joined['keys']):raise ValueError('Duplicate pooled evaluation rows')
    contrasts=[]
    for b,a in CONTRASTS:
        comp=compare(joined['y'],joined[a],joined[b],joined['keys'][:,0])
        directions=all(next(c for c in r['contrasts'] if (c['treatment'],c['control'])==(b,a))['relative_gain']>0 for r in reports)
        contrasts.append({'treatment':b,'control':a,**comp,'both_folds_improve':directions if len(reports)==2 else None,
                          'replication_gate':bool(len(reports)==2 and directions and comp['screen_gate'])})
    r={'status':'pass_axis_study_complete' if len(reports)==2 else 'pass_axis_study_partial',
       'signature':sig,'folds':[r['fold'] for r in reports],
       'fold_metrics':[{'fold':r['fold'],'metrics':r['metrics'],'contrasts':r['contrasts']} for r in reports],
       'pooled_metrics':{a:rmse(joined['y'],joined[a]) for a in ARMS},'contrasts':contrasts,
       'evaluation_rows':len(joined['keys']),'evaluation_games':len(np.unique(joined['keys'][:,0])),
       'new_coordinate_models_recorded':sum(r['new_coordinate_models'] for r in reports),
       'old_models_refitted':0,'feature_research':'open','github_updated':False,'new_kaggle_score':None,
       'scope':'Same selected plays, reused chronological folds; not independent confirmation or Kaggle scoring'}
    r['history_earned_further_testing']=bool(len(reports)==2 and contrasts[1]['replication_gate'] and contrasts[2]['replication_gate'])
    r['terminal_earned_further_testing']=bool(len(reports)==2 and contrasts[0]['replication_gate'])
    atomic_json(out/'summary.json',r);return r
