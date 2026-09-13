"""Observed-only full-player sequences. No truth/baseline prediction is an input.

Keep all simultaneous player pairs; never call attention weights coverage labels.
Control and treatment keep identical channels, masks, roles, and missing-data ages.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from origin_features import validate_raw, canonical_xy, velocity, ROLES, flags

T=20
NODE_NAMES=('relative_x','relative_y','vx','vy','ball_dx','ball_dy',
            'orientation_x','orientation_y','ball_distance','ball_closing_speed')
PAIR_NAMES=('dx','dy','dvx','dvy','distance','closing_speed','lateral_speed','alignment')
NODE_SCALES=np.array([10,10,10,10,20,20,1,1,20,10],float)
PAIR_SCALES=np.array([20,20,10,10,20,10,10,1],float)


def terminal_view(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Last valid value of each channel, repeated only in that channel's valid slots.

    Counterfactual ablation at the final forecast origin, not causal online outputs
    at earlier slots. This is not stale filling in the treatment.
    """
    if values.shape!=valid.shape or valid.dtype!=np.bool_ or values.ndim!=4:
        raise ValueError('Expected equal [source,destination,time,channel] arrays')
    if not np.isfinite(values[valid]).all():raise ValueError('Nonfinite valid pair')
    ix=np.where(valid,np.arange(values.shape[2])[None,None,:,None],-1).max(2)
    last=np.take_along_axis(np.where(valid,values,0),np.maximum(ix,0)[:,:,None,:],axis=2)
    return np.where(valid,last,0).astype(np.float32)


def build(observed: pd.DataFrame, *, origin: int) -> dict[str,np.ndarray]:
    validate_raw(observed)
    if not isinstance(origin,(int,np.integer)) or observed.frame_id.max()>origin:
        raise ValueError('No post-origin observations are accepted')
    if observed.nfl_id.nunique()>22:raise ValueError('More than 22 players')
    left=observed.play_direction.iloc[0]=='left'
    recent=observed.loc[observed.frame_id>origin-T]
    ids=np.sort(recent.nfl_id.unique()).astype(np.int64)
    if not len(ids):raise ValueError('No recent player observations')
    n=len(ids);pos=np.zeros((n,T,2));vel=np.zeros_like(pos)
    seen=np.zeros((n,T),bool);vok=seen.copy();ook=seen.copy();ori=np.zeros_like(pos)
    ball=np.zeros_like(pos);role=np.zeros((n,4),np.float32);side=np.zeros(n,np.int64)
    for i,pid in enumerate(ids):
        g=observed.loc[observed.nfl_id==pid].sort_values('frame_id')
        vv,vo=velocity(g,left);keep=g.frame_id.to_numpy(int)>origin-T
        vv,vo=vv[keep],vo[keep];g=g.loc[keep];slots=g.frame_id.to_numpy(int)-origin+T-1
        pos[i,slots]=canonical_xy(g[['x','y']].to_numpy(float),left)
        ball[i,slots]=canonical_xy(g[['ball_land_x','ball_land_y']].to_numpy(float),left)
        vel[i,slots]=vv;vok[i,slots]=vo;seen[i,slots]=True
        if 'o' in g:
            o=g.o.to_numpy(float);ok=np.isfinite(o);a=np.deg2rad(o[ok])
            ori[i,slots[ok]]=np.c_[np.sin(a),np.cos(a)]*(-1 if left else 1);ook[i,slots[ok]]=True
        role[i,ROLES.index(g.player_role.iloc[-1])]=1
        side[i]=int(g.player_side.iloc[-1]=='Offense')
    last=np.where(seen,np.arange(T),-1).max(1)
    anchor=pos[np.arange(n),last];rel=pos-anchor[:,None];bd=ball-pos
    dist=np.linalg.norm(bd,axis=-1)
    closing=np.sum(bd*vel,axis=-1)/np.maximum(dist,.1)
    nodes=np.concatenate([rel,vel,bd,ori,dist[...,None],closing[...,None]],axis=-1)
    nv=np.repeat(seen[...,None],len(NODE_NAMES),axis=-1)
    nv[...,2:4]&=vok[...,None];nv[...,6:8]&=ook[...,None];nv[...,9]&=vok&(dist>=.1)
    pair=seen[:,None]&seen[None,:]&~np.eye(n,dtype=bool)[...,None]
    pvvel=pair&vok[:,None]&vok[None,:]
    delta=pos[None,:]-pos[:,None];dv=vel[None,:]-vel[:,None]
    dd=np.linalg.norm(delta,axis=-1);speed=np.linalg.norm(vel,axis=-1)
    dot=(delta*dv).sum(-1);cross=delta[...,0]*dv[...,1]-delta[...,1]*dv[...,0]
    align=(vel[:,None]*vel[None,:]).sum(-1)/np.maximum(speed[:,None]*speed[None,:],1e-12)
    values=np.concatenate([delta,dv,dd[...,None],(-dot/np.maximum(dd,.1))[...,None],
                           (cross/np.maximum(dd,.1))[...,None],np.clip(align,-1,1)[...,None]],axis=-1)
    valid=np.repeat(pair[...,None],len(PAIR_NAMES),axis=-1)
    valid[...,2:4]&=pvvel[...,None]
    valid[...,5:7]&=(pvvel&(dd>=.1))[...,None]
    valid[...,7]&=pvvel&(speed[:,None]>=1e-3)&(speed[None,:]>=1e-3)
    values=np.where(valid,values/PAIR_SCALES,0).astype(np.float32)
    node=np.where(nv,nodes/NODE_SCALES,0).astype(np.float32)
    vi=np.where(valid,np.arange(T)[None,None,:,None],-1).max(2)
    age=np.where(vi>=0,(T-1-vi)/10,2.0).astype(np.float32)
    ans={'ids':ids,'node':node,'node_valid':nv,'pair':values,'pair_valid':valid,
         'pair_age':age,'role':role,'side':side,'node_age':((T-1-last)/10).astype(np.float32)}
    if any(not np.isfinite(v).all() for v in ans.values()):raise ValueError('Nonfinite sequence features')
    return ans
