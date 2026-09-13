"""Observed-only, fixed-identity player pairs; no inferred coverage assignments.

Terminal and history views have identical dimensions and availability masks.
No label table, fitted outcome statistic, or post-throw coordinate is accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from origin_features import KEYS, canonical_xy, validate_raw, velocity

LAGS = (0, 2, 5, 10, 19)
SLOTS = ('opponent_1', 'opponent_2', 'teammate_1', 'targeted_receiver')
CHANNELS = ('dx', 'dy', 'dvx', 'dvy', 'distance', 'closing', 'lateral', 'alignment')
SCALES = np.array([20, 20, 10, 10, 20, 10, 10, 1], float)
NAMES = [f'{slot}__lag{lag:02d}__{kind}__{ch}'
         for slot in SLOTS for lag in LAGS for kind in ('value', 'valid') for ch in CHANNELS]
NAMES += [f'{slot}__joint_age_seconds' for slot in SLOTS]
ODD = np.array([i for i, n in enumerate(NAMES) if '__value__' in n and n.rsplit('__',1)[1] in ('dy','dvy','lateral')])


def _pair(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint = a['seen'] & b['seen']
    dvok = joint & a['vok'] & b['vok']
    delta, dv = b['xy']-a['xy'], b['v']-a['v']
    dist = np.linalg.norm(delta, axis=1)
    sa, sb = np.linalg.norm(a['v'], axis=1), np.linalg.norm(b['v'], axis=1)
    values = np.column_stack([delta, dv, dist,
        -(delta*dv).sum(1)/np.maximum(dist,.1),
        (delta[:,0]*dv[:,1]-delta[:,1]*dv[:,0])/np.maximum(dist,.1),
        (a['v']*b['v']).sum(1)/np.maximum(sa*sb,1e-12)])
    valid = np.column_stack([joint,joint,dvok,dvok,joint,dvok&(dist>=.1),
                            dvok&(dist>=.1),dvok&(sa>=1e-3)&(sb>=1e-3)])
    values[:,7] = np.clip(values[:,7],-1,1)
    values = np.where(valid,values/SCALES,0)
    return values,valid,joint


def pair_views(raw: pd.DataFrame, queries: np.ndarray, *, cutoff: int) -> dict[str,np.ndarray]:
    """Return two [forecast rows,324] arrays with the same observation support.

    Select up to two opposing players, one teammate, and the targeted receiver.
    Rank by freshest joint observation, then joint distance, then stable player ID.
    Freeze each selected identity across the history. Never switch its identity at
    earlier lags, bridge a gap, infer man coverage, or use a future target location.
    """
    validate_raw(raw)
    if not isinstance(cutoff,(int,np.integer)) or (raw.frame_id > cutoff).any():
        raise ValueError('Observed rows exceed the explicit prediction cutoff')
    q=np.asarray(queries)
    if q.ndim!=2 or q.shape[1]!=4 or not np.issubdtype(q.dtype,np.integer) or not len(q):
        raise ValueError('Unique integer [game,play,player,future frame] keys required')
    if len(np.unique(q,axis=0))!=len(q) or (q[:,3]<1).any():
        raise ValueError('Duplicate or invalid forecast keys')
    if not np.all(q[:,:2]==raw[KEYS[:2]].iloc[0].to_numpy(int)):
        raise ValueError('Queries belong to a different play')
    left=raw.play_direction.iloc[0]=='left'
    states={}
    for ident,g in raw.groupby('nfl_id',sort=True):
        g=g.sort_values('frame_id'); v,vok=velocity(g,left)
        take=g.frame_id.to_numpy(int)>cutoff-20
        if not take.any(): continue
        slots=g.frame_id.to_numpy(int)[take]-cutoff+19
        state={'seen':np.zeros(20,bool),'vok':np.zeros(20,bool),
               'xy':np.zeros((20,2)), 'v':np.zeros((20,2)),
               'side':g.player_side.iloc[-1], 'role':g.player_role.iloc[-1],
               'horizon':int(g.num_frames_output.iloc[-1])}
        state['seen'][slots]=True;state['vok'][slots]=vok[take]
        state['xy'][slots]=canonical_xy(g[['x','y']].to_numpy(float)[take],left)
        state['v'][slots]=v[take];states[int(ident)]=state
    if len(states)>22: raise ValueError('More than 22 observed players')
    target=[i for i,s in states.items() if s['role']=='Targeted Receiver']
    if len(target)>1: raise ValueError('Ambiguous targeted receiver')
    xs,xt,counts=[],[],[]
    for ident in np.unique(q[:,2]):
        if ident not in states or np.any(q[q[:,2]==ident,3]>states[ident]['horizon']):
            raise ValueError('Query is outside the observed player/horizon contract')
        a=states[int(ident)]; candidates=[];pairs={}
        for other,b in states.items():
            if other==ident:continue
            p=_pair(a,b); js=np.flatnonzero(p[2])
            if not len(js):continue
            end=int(js[-1]);pairs[other]=(p,end)
            candidates.append((19-end,float(np.linalg.norm(b['xy'][end]-a['xy'][end])),other))
        ordered=[c[2] for c in sorted(candidates)]
        opp=[i for i in ordered if states[i]['side']!=a['side']]
        team=[i for i in ordered if states[i]['side']==a['side']]
        selected=[opp[0] if opp else None,opp[1] if len(opp)>1 else None,
                  team[0] if team else None,target[0] if target and target[0]!=ident and target[0] in pairs else None]
        hv,tv,ages,support=[],[],[],[]
        for other in selected:
            if other is None:
                hv.extend([0.]*(len(LAGS)*len(CHANNELS)*2));tv.extend([0.]*(len(LAGS)*len(CHANNELS)*2))
                ages.append(2.);support.append(0.);continue
            (value,valid,joint),end=pairs[other]
            ages.append((19-end)/10);support.append(float(joint.mean()))
            for lag in LAGS:
                k=19-lag
                # Both arms see identical channel masks; a terminal-unavailable
                # channel cannot create a spurious treatment/control difference.
                common=valid[k]&valid[end]
                hv.extend(np.where(common,value[k],0));hv.extend(common.astype(float))
                tv.extend(np.where(common,value[end],0));tv.extend(common.astype(float))
        h=np.asarray(hv+ages,float);t=np.asarray(tv+ages,float)
        xs.append((ident,h));xt.append((ident,t));counts.append((ident,np.asarray(support)))
    hd,td,sd=dict(xs),dict(xt),dict(counts)
    h=np.stack([hd[i] for i in q[:,2]]);t=np.stack([td[i] for i in q[:,2]])
    if h.shape!=(len(q),len(NAMES)) or not np.isfinite(h).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite or invalid pair representation')
    return {'history':h,'terminal':t,'support':np.stack([sd[i] for i in q[:,2]])}
