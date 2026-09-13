"""Observed-only player-centric receiver and motion channels; no predictive claim.

Receiver values reuse simultaneous pair measurements. Motion differences require
adjacent 10 Hz observations. No interpolation, future targets, clipping, fitted
scaling, or validation-dependent selection occurs here.
"""
from __future__ import annotations
import numpy as np

NAMES=('receiver_dx','receiver_dy','receiver_dvx','receiver_dvy',
       'receiver_distance','receiver_closing_speed',
       'observed_ax','observed_ay','tangential_acceleration',
       'normal_acceleration','velocity_turn_rate','speed_change_rate')
SCALES=np.array([20,20,10,10,20,10,20,20,20,20,10,20],dtype=np.float64)
UNITS=('yards','yards','yards/s','yards/s','yards','yards/s',
       'yards/s^2','yards/s^2','yards/s^2','yards/s^2','rad/s','yards/s^2')
ODD=(1,3,7,9,10)
DT=0.1


def build(sample):
    """Return [player,20,12] values and channel-validity mask, from inputs only."""
    node=np.asarray(sample['node']); nv=np.asarray(sample['node_valid'])
    pair=np.asarray(sample['pair']); pv=np.asarray(sample['pair_valid'])
    role=np.asarray(sample['role'])
    if node.ndim!=3 or node.shape[1:]!=(20,10) or not 1<=len(node)<=22:
        raise ValueError('Expected 1..22 players and 20 x 10 observed channels')
    n=len(node)
    if nv.dtype!=np.bool_ or nv.shape!=node.shape or pv.dtype!=np.bool_ or pair.shape!=(n,n,20,8) or pv.shape!=pair.shape:
        raise ValueError('Invalid observed mask/tensor layout')
    if role.shape!=(n,4) or not np.isin(role,[0,1]).all() or not np.all(role.sum(1)==1):
        raise ValueError('One supplied role per player required')
    if not np.isfinite(node[nv]).all() or not np.isfinite(pair[pv]).all():
        raise ValueError('Nonfinite valid observation')
    if pv[np.arange(n),np.arange(n)].any():
        raise ValueError('Self pairs must be masked')
    receivers=np.flatnonzero(role[:,0]==1)
    if len(receivers)>1:raise ValueError('Ambiguous targeted-receiver role')
    val=np.zeros((n,20,12),dtype=np.float64); valid=np.zeros_like(val,dtype=bool)
    if len(receivers):
        r=int(receivers[0]); scales=np.array([20,20,10,10,20,10],float)
        valid[...,:6]=pv[:,r,:,:6]
        val[...,:6]=np.where(valid[...,:6],pair[:,r,:,:6],0)*scales
        valid[r,:,:6]=False;val[r,:,:6]=0 # The receiver is not its own neighbour.
    ok=nv[...,2]&nv[...,3]&nv[...,0]&nv[...,1]
    v=np.where(ok[...,None],node[...,2:4],0).astype(np.float64)*10
    adjacent=np.zeros((n,20),bool);adjacent[:,1:]=ok[:,1:]&ok[:,:-1]
    acc=np.zeros_like(v);acc[:,1:]=np.diff(v,axis=1)/DT
    speed=np.linalg.norm(v,axis=-1)
    val[...,6:8]=acc;valid[...,6:8]=adjacent[...,None]
    moving=speed>=.1
    val[...,8]=np.sum(v*acc,axis=-1)/np.maximum(speed,.1)
    val[...,9]=(v[...,0]*acc[...,1]-v[...,1]*acc[...,0])/np.maximum(speed,.1)
    valid[...,8]=adjacent&moving;valid[...,9]=adjacent&moving
    a,b=v[:,:-1],v[:,1:]
    val[:,1:,10]=np.arctan2(a[...,0]*b[...,1]-a[...,1]*b[...,0],np.sum(a*b,axis=-1))/DT
    valid[:,1:,10]=adjacent[:,1:]&moving[:,1:]&moving[:,:-1]
    val[:,1:,11]=np.diff(speed,axis=1)/DT;valid[...,11]=adjacent
    values=np.where(valid,val/SCALES,0).astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Feature overflow; do not clip silently')
    return {'values':values,'valid':valid}


def summarize(arrays):
    """Exact training-only support, physical-unit moments and percentiles."""
    counts=np.zeros(12,np.int64);denom=0;pieces=[[] for _ in NAMES]
    for a in arrays:
        x,m=a['values'],a['valid'];denom+=x.shape[0]*20
        counts+=m.sum((0,1))
        for j in range(12):
            pieces[j].append(x[...,j][m[...,j]].astype(float)*SCALES[j])
    stats=[]
    for j,name in enumerate(NAMES):
        x=np.concatenate(pieces[j]) if pieces[j] else np.empty(0)
        stats.append({'name':name,'family':'receiver' if j<6 else 'motion',
          'unit':UNITS[j],'valid':int(counts[j]),'slots':int(denom),
          'support_fraction':float(counts[j]/denom) if denom else 0.,
          'rms':float(np.sqrt(np.mean(x*x))) if len(x) else None,
          'p01':float(np.quantile(x,.01)) if len(x) else None,
          'median':float(np.median(x)) if len(x) else None,
          'p99':float(np.quantile(x,.99)) if len(x) else None,
          'max_abs':float(np.max(np.abs(x))) if len(x) else None})
    return stats
