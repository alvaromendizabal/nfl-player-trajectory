"""Observed-only route shape (16) and opponent approach geometry (17).

Prepared implementations, not executed by the assistant. Each family has six
core and six extension values with channel-specific masks. No targets, fitted
statistics, future observations, labels, or post-throw tracking are read.
"""
from __future__ import annotations
import numpy as np
from new_features import build as validate_reference

ARMS=('mask','core','full')
NAMES={
 16:('forward_displacement_3','lateral_displacement_3','path_efficiency_3',
     'forward_displacement_5','lateral_displacement_5','path_efficiency_5',
     'forward_displacement_10','lateral_displacement_10','path_efficiency_10',
     'absolute_turn_sum_10','signed_turn_sum_10','step_direction_resultant_10'),
 17:('opponents_within_3y','opponents_within_6y','nearest_opponent_distance',
     'nearest_opponent_closing_speed','minimum_1s_projected_separation','selected_closest_approach_time',
     'nearest_opponent_lateral_speed','closing_opponent_fraction','projected_threats_within_1y',
     'projected_threats_within_2y','landing_corridor_opponents','mean_closing_speed_within_6y')}
SCALES={16:np.array([10,10,1,10,10,1,10,10,1,3.141592653589793,3.141592653589793,1],float),
        17:np.array([10,10,20,10,20,1,10,1,10,10,10,10],float)}
UNITS={16:('yards','yards','ratio')*3+('radians','radians','ratio'),
       17:('players','players','yards','yards/s','yards','seconds','yards/s','fraction',
           'players','players','players','yards/s')}
ODD={16:(1,4,7,10),17:(6,)}
DT=.1


def _round(r):
    if type(r) is not int or r not in NAMES:raise ValueError('Round must be 16 or 17')


def difference(x,valid):
    if x.shape!=valid.shape or valid.dtype!=np.bool_ or x.ndim<2:
        raise ValueError('Aligned values and boolean masks required')
    m=np.zeros_like(valid);m[:,1:]=valid[:,1:]&valid[:,:-1]
    z=np.where(valid,x,0).astype(float);out=np.zeros_like(z)
    out[:,1:]=np.diff(z,axis=1)/DT
    return np.where(m,out,0),m


def _route(sample):
    node=np.asarray(sample['node']);nv=np.asarray(sample['node_valid']);n,t=node.shape[:2]
    # Node x/y are relative to each player's terminal observed anchor. Differences
    # within a player's window cancel the anchor; cross-player subtraction is not used.
    xy=np.where(nv[...,:2],node[...,:2],0).astype(float)*10
    v=np.where(nv[...,2:4],node[...,2:4],0).astype(float)*10
    xyok=nv[...,:2].all(-1);vok=nv[...,2:4].all(-1)
    speed=np.linalg.norm(v,axis=-1);heading=v/np.maximum(speed,.1)[...,None]
    values=np.zeros((n,t,12),float);valid=np.zeros_like(values,bool)
    for window,start in ((3,0),(5,3),(10,6)):
        for ti in range(window-1,t):
            seg=xy[:,ti-window+1:ti+1];ok=xyok[:,ti-window+1:ti+1].all(1)
            displacement=seg[:,-1]-seg[:,0];steps=np.diff(seg,axis=1)
            lengths=np.linalg.norm(steps,axis=-1);path=lengths.sum(1)
            axis=heading[:,ti];oriented=ok&vok[:,ti]&(speed[:,ti]>=.1)
            values[:,ti,start]=(displacement*axis).sum(-1)
            values[:,ti,start+1]=axis[:,0]*displacement[:,1]-axis[:,1]*displacement[:,0]
            valid[:,ti,start:start+2]=oriented[:,None]
            good_path=ok&(path>=.1)
            values[:,ti,start+2]=np.linalg.norm(displacement,axis=-1)/np.maximum(path,.1)
            valid[:,ti,start+2]=good_path
            if window==10:
                moving=ok&(lengths>=1e-4).all(1)
                directions=steps/np.maximum(lengths,1e-4)[...,None]
                first,last=directions[:,:-1],directions[:,1:]
                cross=first[...,0]*last[...,1]-first[...,1]*last[...,0]
                dot=(first*last).sum(-1);angles=np.arctan2(cross,dot)
                values[:,ti,9]=np.abs(angles).sum(1)
                values[:,ti,10]=angles.sum(1)
                values[:,ti,11]=np.linalg.norm(directions.mean(1),axis=-1)
                valid[:,ti,9:]=moving[:,None]
    return values,valid


def _traffic(sample):
    pair=np.asarray(sample['pair']);pv=np.asarray(sample['pair_valid'])
    node=np.asarray(sample['node']);nv=np.asarray(sample['node_valid'])
    side=np.asarray(sample['side']);n,t=node.shape[:2]
    if side.shape!=(n,) or not np.isin(side,[0,1]).all():raise ValueError('One 0/1 side per player required')
    values=np.zeros((n,t,12),float);valid=np.zeros_like(values,bool)
    # Coordinates are simultaneous destination-minus-source measurements.
    delta=np.where(pv[...,:2],pair[...,:2],0).astype(float)*20
    dv=np.where(pv[...,2:4],pair[...,2:4],0).astype(float)*10
    pair_ok=pv[...,:4].all(-1)&(side[:,None]!=side[None,:])[:,:,None]
    pair_ok &= ~np.eye(n,dtype=bool)[:,:,None]
    goal=np.where(nv[...,4:6],node[...,4:6],0).astype(float)*20
    for i in range(n):
        for ti in range(t):
            use=np.flatnonzero(pair_ok[i,:,ti])
            if not len(use):continue  # unavailable is not an invented zero count
            d=delta[i,use,ti];w=dv[i,use,ti];distance=np.linalg.norm(d,axis=-1)
            radial=(d*w).sum(-1);nonzero=distance>=.1
            closing=-radial/np.maximum(distance,.1)
            lateral=(d[:,0]*w[:,1]-d[:,1]*w[:,0])/np.maximum(distance,.1)
            speed2=(w*w).sum(-1)
            # Constant-relative-velocity hypothesis, constrained to the next 1 s.
            # This clips a hypothetical time, not any observed value or target row.
            tau=np.where(speed2>=1e-8,-radial/np.maximum(speed2,1e-8),0)
            tau=np.minimum(1,np.maximum(0,tau))
            projected=np.linalg.norm(d+tau[:,None]*w,axis=-1)
            near=np.isclose(distance,distance.min(),rtol=0,atol=1e-8)
            winner=np.isclose(projected,projected.min(),rtol=0,atol=1e-8)
            values[i,ti,:6]=[(distance<=3).sum(),(distance<=6).sum(),distance.min(),
                            closing[near].mean(),projected.min(),tau[winner].mean()]
            valid[i,ti,:6]=True;valid[i,ti,3]=bool(nonzero[near].all())
            values[i,ti,6]=lateral[near].mean();valid[i,ti,6]=bool(nonzero[near].all())
            if nonzero.any():
                values[i,ti,7]=(closing[nonzero]>0).mean();valid[i,ti,7]=True
            values[i,ti,8]=(projected<=1).sum();values[i,ti,9]=(projected<=2).sum()
            valid[i,ti,8:10]=True
            g=goal[i,ti];length=float(np.linalg.norm(g))
            if nv[i,ti,4:6].all() and length>=.1:
                axis=g/length;along=d@axis;across=axis[0]*d[:,1]-axis[1]*d[:,0]
                values[i,ti,10]=((along>=0)&(along<=length)&(np.abs(across)<=2)).sum()
                valid[i,ti,10]=True
            local=(distance<=6)&nonzero
            if local.any():
                values[i,ti,11]=closing[local].mean();valid[i,ti,11]=True
    return values,valid


def build(sample,round_no):
    _round(round_no)
    # Reuse the existing input contract validator; this never reads targets.
    validate_reference(sample)
    values,valid=_route(sample) if round_no==16 else _traffic(sample)
    out=np.where(valid,values/SCALES[round_no],0).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite feature; no clipping fallback')
    return {'values':out,'valid':valid}


def view(ext,arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    x,m=np.asarray(ext['values']),np.asarray(ext['valid'])
    if x.ndim!=3 or x.shape[1:]!=(20,12) or m.shape!=x.shape or m.dtype!=np.bool_:
        raise ValueError('Expected [player,20,12] with boolean masks')
    if not np.isfinite(x[m]).all():raise ValueError('Nonfinite supported values')
    out=np.where(m,x,0).astype(np.float32)
    if arm=='mask':out[:]=0
    elif arm=='core':out[...,6:]=0
    return out,m.copy()


def summarize(arrays,round_no):
    _round(round_no);rows=[]
    for j,name in enumerate(NAMES[round_no]):
        den=sum(a['values'].shape[0]*20 for a in arrays)
        parts=[a['values'][...,j][a['valid'][...,j]].astype(float)*SCALES[round_no][j] for a in arrays]
        x=np.concatenate(parts) if parts else np.empty(0)
        rows.append({'name':name,'unit':UNITS[round_no][j],'block':'core' if j<6 else 'extension',
                     'slots':den,'valid':len(x),'support':len(x)/den if den else 0,
                     'rms':float(np.sqrt(np.mean(x*x))) if len(x) else None,
                     'p01':float(np.quantile(x,.01)) if len(x) else None,
                     'p99':float(np.quantile(x,.99)) if len(x) else None,
                     'max_abs':float(np.max(np.abs(x))) if len(x) else None})
    return rows
