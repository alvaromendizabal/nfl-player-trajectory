"""Verify labels for observed plays in the SAME frozen 64 training games.

Never select plays by their errors or target values. Mixed-game organizer files
are scanned in chunks, then filtered to allowed training games immediately.
Non-training label bytes may be read from the same CSV chunks; they are never
scored, summarized, retained or used to choose samples. This is not a new split.
No filtering/removal of problematic plays is performed; issues stop readiness.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_io import digest, hash_json, safe_file, read_json, seal_json
import bridge

KEYS=['game_id','play_id','nfl_id','frame_id']


def ints(frame, names):
    out=frame.copy()
    for col in names:
        value=pd.to_numeric(out[col],errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(value).all() or not np.equal(value,np.floor(value)).all():
            raise ValueError('Missing or noninteger training key/horizon: '+col)
        out[col]=value.astype(np.int64)
    return out


def audit_week(input_path: Path, output_path: Path, allowed: set[int]):
    """Return private per-play label eligibility; no coordinates are retained."""
    expected={};input_seen=set();all_plays=set();flag_map={}
    columns=KEYS+['player_to_predict','num_frames_output']
    for chunk in pd.read_csv(input_path,usecols=columns,chunksize=100000):
        x=chunk.loc[chunk.game_id.isin(allowed)].copy()
        if x.empty:continue
        x=ints(x,KEYS)[columns]
        flags=x.player_to_predict.astype(str).str.lower()
        if not flags.isin(['true','false','1','0']).all():
            raise ValueError('Invalid training target flag')
        for row, flag in zip(x.itertuples(index=False,name=None),flags.isin(['true','1']).to_numpy()):
            key=tuple(int(v) for v in row[:4]);entity=key[:3];play=key[:2]
            if key in input_seen:raise ValueError('Duplicate observed training frame')
            input_seen.add(key);all_plays.add(play)
            if entity in flag_map and flag_map[entity]!=bool(flag):
                raise ValueError('Target flag changes within a training trajectory')
            flag_map[entity]=bool(flag)
            if flag:
                horizon=float(row[5])
                if not np.isfinite(horizon) or horizon!=int(horizon) or not 1<=horizon<=2000:
                    raise ValueError('Invalid or unexpectedly long training horizon; review, do not clip')
                if entity in expected and expected[entity]!=int(horizon):
                    raise ValueError('Horizon changes within a training trajectory')
                expected[entity]=int(horizon)
    expected_keys={(*e,f) for e,h in expected.items() for f in range(1,h+1)}
    seen=set();bad=set();duplicates=set();unexpected=set()
    for chunk in pd.read_csv(output_path,usecols=KEYS+['x','y'],chunksize=100000):
        x=chunk.loc[chunk.game_id.isin(allowed)].copy()
        if x.empty:continue
        x=ints(x,KEYS)
        coords=x[['x','y']].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
        for key,finite in zip(x[KEYS].itertuples(index=False,name=None),np.isfinite(coords).all(axis=1)):
            key=tuple(int(v) for v in key)
            if key in seen:duplicates.add(key)
            seen.add(key)
            if key not in expected_keys:unexpected.add(key)
            if not finite:bad.add(key)
    missing=expected_keys-seen
    problematic={key[:2] for group in (missing,unexpected,duplicates,bad) for key in group}
    no_targets=all_plays-{e[:2] for e in expected}
    # No-target plays are reported rather than silently folded into eligibility.
    problematic|=no_targets
    eligible=all_plays-problematic
    return {'observed_plays':[list(p) for p in sorted(all_plays)],
            'eligible_plays':[list(p) for p in sorted(eligible)],
            'counts':{'observed_plays':len(all_plays),'eligible_plays':len(eligible),
                      'expected_rows':len(expected_keys),'actual_unique_rows':len(seen),
                      'missing_rows':len(missing),'unexpected_rows':len(unexpected),
                      'duplicate_rows':len(duplicates),'nonfinite_rows':len(bad),
                      'plays_without_scored_targets':len(no_targets)}}


def audit(a):
    if a.round_no!=14:
        raise ValueError('Run shared label-readiness audit once from notebook 30 / round 14')
    v=bridge.context(a);allowed=set(v['training_games'])
    selected={bridge.parent_core.game_play(r) for r in v['train']}
    observed=set();eligible=set();records=[];totals={}
    source=v['source']
    training_fingerprint=hash_json(sorted(allowed))
    for i,record in enumerate(v['contract']['raw_inputs']):
        inp=safe_file(a.repo,record['path'])
        if digest(inp)!=record['sha256'] or inp.stat().st_size!=record['size']:
            raise ValueError('Observed organizer input hash differs')
        if not inp.name.startswith('input_'):
            raise ValueError('Unexpected organizer input naming')
        out=safe_file(inp.parent,inp.name.replace('input_','output_',1))
        output_sha=digest(out)
        identity={'source':source,'training_games_hash':training_fingerprint,
                  'input_sha256':record['sha256'],'output_sha256':output_sha}
        dest=a.out/'labels_private'/f'{i:02d}.json'
        if dest.exists():
            saved=read_json(safe_file(dest.parent,dest.name))
            receipt=read_json(safe_file(dest.parent,dest.name+'.receipt.json'))
            if receipt['sha256']!=digest(dest) or saved['identity']!=identity:
                raise ValueError('Training-label audit cache drift')
        else:
            saved={'identity':identity,**audit_week(inp,out,allowed)}
            seal_json(dest,saved)
            seal_json(dest.with_name(dest.name+'.receipt.json'),{'sha256':digest(dest)})
        week_observed={tuple(p) for p in saved['observed_plays']}
        week_eligible={tuple(p) for p in saved['eligible_plays']}
        if week_observed & observed:raise ValueError('Training play spans multiple weekly files')
        observed|=week_observed;eligible|=week_eligible
        for key,n in saved['counts'].items():totals[key]=totals.get(key,0)+n
        if digest(inp)!=record['sha256'] or digest(out)!=output_sha:
            raise ValueError('Raw file changed during label audit')
        records.append({'path':str(dest.relative_to(a.out)),'sha256':digest(dest),**identity})
        print(json.dumps({'event':'label_coverage_progress','files':i+1,
                          'total':len(v['contract']['raw_inputs']),'training_plays_checked':len(observed)}),flush=True)
    if not selected.issubset(observed):raise ValueError('Selected training play missing from organizer inputs')
    issue_fields=('missing_rows','unexpected_rows','duplicate_rows','nonfinite_rows','plays_without_scored_targets')
    passed=(eligible==observed and selected.issubset(eligible)
            and all(totals.get(key,0)==0 for key in issue_fields))
    report={'status':'training_labels_complete' if passed else 'training_label_issues_require_review',
            'signature':v['signature'],'source_signature':source,'training_games':len(allowed),
            'selected_training_plays':len(selected),'observed_training_plays':len(observed),
            'label_eligible_training_plays':len(eligible),'additional_label_eligible_plays':len(eligible-selected),
            'selected_label_eligible':len(eligible&selected),'counts':totals,
            'new_training_selection_created':False,'validation_scored':False,'new_model_fits':0,
            'label_provenance':'Existing local output CSV snapshot hashes recorded; no new remote provenance assertion',
            'scope':'Mixed-game CSV chunks read then immediately restricted to frozen training games; other outcomes discarded',
            'next_decision':'Review data-scale-only comparison before authorizing additional feature fits'}
    seal_json(a.out/'label_snapshot_private.json',{'files':records,'eligible_plays':[list(p) for p in sorted(eligible)]})
    seal_json(a.out/'label_readiness.json',report)
    bridge.unchanged(v)
    return report
