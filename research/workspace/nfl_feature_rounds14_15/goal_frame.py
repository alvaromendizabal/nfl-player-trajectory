"""Observed-only pair geometry in the source player's ball-goal frame.

A change of basis, not new raw information. No labels or forecast outputs are
accepted. Each time slot uses the current observed ball-relative direction;
there is no forward filling, derivative over gaps, or coverage assignment.
"""
from __future__ import annotations
import numpy as np

NAMES = ('goal_parallel_separation','goal_lateral_separation',
         'goal_parallel_relative_velocity','goal_lateral_relative_velocity',
         'peer_minus_self_goal_distance','peer_minus_self_goal_closing')
SCALES = np.array([20.,20.,10.,10.,20.,10.])
GROUPS = ('same_side','opponent','targeted_receiver','passer')
REQUIRED = ('node','node_valid','pair','pair_valid','role','side')


def build(sample: dict) -> dict[str, np.ndarray]:
    node=np.asarray(sample['node']); nv=np.asarray(sample['node_valid'])
    pair=np.asarray(sample['pair']); pv=np.asarray(sample['pair_valid'])
    role=np.asarray(sample['role']); side=np.asarray(sample['side'])
    if node.ndim != 3 or node.shape[1:] != (20,10):
        raise ValueError('Expected [players,20,10] observed node channels')
    n=len(node)
    if not 1<=n<=22 or nv.shape!=node.shape or nv.dtype!=np.bool_:
        raise ValueError('Invalid node masks or player count')
    if pair.shape!=(n,n,20,8) or pv.shape!=pair.shape or pv.dtype!=np.bool_:
        raise ValueError('Invalid pair shape/masks')
    if role.shape!=(n,4) or not np.isin(role,[0,1]).all() or not np.all(role.sum(1)==1):
        raise ValueError('Expected four organizer-role one-hot columns')
    if side.shape!=(n,) or not np.isin(side,[0,1]).all():
        raise ValueError('Invalid side')
    if not np.isfinite(node[nv]).all() or not np.isfinite(pair[pv]).all():
        raise ValueError('Nonfinite observed measurements')
    node=np.where(nv,node,0).astype(np.float64)
    pair=np.where(pv,pair,0).astype(np.float64)
    goal=node[...,4:6]*20
    distance=np.linalg.norm(goal,axis=-1)
    goal_ok=nv[...,4]&nv[...,5]
    direction_ok=goal_ok&(distance>=.1)
    unit=goal/np.maximum(distance,.1)[...,None]
    u=unit[:,None]
    dp=pair[...,:2]*20; dv=pair[...,2:4]*10
    joint=(pv[...,0]&pv[...,1])&~np.eye(n,dtype=bool)[...,None]
    pos_ok=joint&direction_ok[:,None]
    vel_ok=pos_ok&pv[...,2]&pv[...,3]
    dg=distance[None]-distance[:,None]
    speed=node[...,2:4]*10
    closing=(goal*speed).sum(-1)/np.maximum(distance,.1)
    closing_ok=direction_ok&nv[...,2]&nv[...,3]
    values=np.stack([(dp*u).sum(-1),u[...,0]*dp[...,1]-u[...,1]*dp[...,0],
                     (dv*u).sum(-1),u[...,0]*dv[...,1]-u[...,1]*dv[...,0],
                     dg,closing[None]-closing[:,None]],axis=-1)
    valid=np.stack([pos_ok,pos_ok,vel_ok,vel_ok,
                    joint&goal_ok[:,None]&goal_ok[None],
                    joint&closing_ok[:,None]&closing_ok[None]],axis=-1)
    values=np.where(valid,values/SCALES,0).astype(np.float32)
    relation=np.stack([np.broadcast_to(side[:,None]==side[None,:],(n,n)),
                      np.broadcast_to(side[:,None]!=side[None,:],(n,n)),
                      np.broadcast_to(role[None,:,0]==1,(n,n)),
                      np.broadcast_to(role[None,:,2]==1,(n,n))],axis=-1)
    # The original adapter's ROLES order is checked by the caller's source hash.
    relation=joint[...,None]&relation[:,:,None,:]
    return {'values':values,'valid':valid,'groups':relation}
