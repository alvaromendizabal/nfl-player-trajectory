"""Two independent observed-only feature families; no target information accepted.

Each round has 12 numerical channels and 12 validity masks. The first six reproduce
Round 11 exactly. Added channels use fixed physical scales, never validation fit.
Missing observations are neither forward-filled nor bridged by time differences.
"""
from __future__ import annotations
import numpy as np
from new_features import build as core_features

ARMS=('mask','core','full')
NAMES={12:('receiver_dx','receiver_dy','receiver_dvx','receiver_dvy','receiver_distance',
    'receiver_closing_speed','receiver_lateral_speed','receiver_velocity_alignment',
    'receiver_bearing_rate','receiver_distance_change_rate','receiver_goal_parallel_sep',
    'receiver_goal_lateral_sep'),
 13:('observed_ax','observed_ay','tangential_acceleration','normal_acceleration',
    'velocity_turn_rate','speed_change_rate','ax_trailing3','ay_trailing3',
    'ax_trailing5','ay_trailing5','observed_jerk_x','observed_jerk_y')}
SCALES={12:np.array([20,20,10,10,20,10,10,1,10,10,20,20],float),
        13:np.array([20,20,20,20,10,20,20,20,20,20,100,100],float)}
UNITS={12:('yards','yards','yards/s','yards/s','yards','yards/s','yards/s','cosine',
           'rad/s','yards/s','yards','yards'),
       13:('yards/s^2',)*4+('rad/s','yards/s^2')+('yards/s^2',)*4+('yards/s^3',)*2}
ODD={12:(1,3,6,8,11),13:(1,3,4,7,9,11)}
DT=.1

def _round(r):
    if type(r) is not int or r not in NAMES: raise ValueError('Round must be 12 or 13')

def build(sample,round_no):
    _round(round_no)
    core=core_features(sample)  # Validates layout, roles, self edges and observed values.
    shape=core['values'].shape; v=np.zeros(shape,np.float64);m=np.zeros(shape,bool)
    sl=slice(0,6) if round_no==12 else slice(6,12)
    v[...,:6]=core['values'][...,sl].astype(float)*SCALES[round_no][:6]
    m[...,:6]=core['valid'][...,sl]
    if round_no==12:
        n,t,_=shape; role=np.asarray(sample['role']); rec=np.flatnonzero(role[:,0]==1)
        if len(rec):
            j=int(rec[0]); pair=np.asarray(sample['pair']);pv=np.asarray(sample['pair_valid'])
            dp=np.where(pv[:,j,:,:2],pair[:,j,:,:2],0).astype(float)*20
            dv=np.where(pv[:,j,:,2:4],pair[:,j,:,2:4],0).astype(float)*10
            dist=np.linalg.norm(dp,axis=-1);geo=pv[:,j,:,:2].all(-1)&(dist>=.1)
            motion=geo&pv[:,j,:,2:4].all(-1)
            v[...,6]=(dp[...,0]*dv[...,1]-dp[...,1]*dv[...,0])/np.maximum(dist,.1)
            m[...,6]=motion
            v[...,7]=np.where(pv[:,j,:,7],pair[:,j,:,7],0);m[...,7]=pv[:,j,:,7]
            adjacent=geo[:,1:]&geo[:,:-1]
            a,b=dp[:,:-1],dp[:,1:]
            v[:,1:,8]=np.arctan2(a[...,0]*b[...,1]-a[...,1]*b[...,0],np.sum(a*b,-1))/DT
            m[:,1:,8]=adjacent
            v[:,1:,9]=np.diff(dist,axis=1)/DT;m[:,1:,9]=adjacent
            node=np.asarray(sample['node']);nv=np.asarray(sample['node_valid'])
            goal=np.where(nv[...,4:6],node[...,4:6],0).astype(float)*20
            d=np.linalg.norm(goal,axis=-1);u=goal/np.maximum(d,.1)[...,None]
            ok=geo&nv[...,4:6].all(-1)&(d>=.1)
            v[...,10]=np.sum(dp*u,-1);v[...,11]=u[...,0]*dp[...,1]-u[...,1]*dp[...,0]
            m[...,10:12]=ok[...,None];m[j,:,:]=False;v[j,:,:]=0
    else:
        acc=core['values'][...,6:8].astype(float)*20;ok=core['valid'][...,6:8].all(-1)
        for window,start in ((3,6),(5,8)):
            for ti in range(window-1,20):
                valid=ok[:,ti-window+1:ti+1].all(1)
                v[:,ti,start:start+2]=acc[:,ti-window+1:ti+1].mean(1)
                m[:,ti,start:start+2]=valid[:,None]
        v[:,1:,10:12]=np.diff(acc,axis=1)/DT
        m[:,1:,10:12]=(ok[:,1:]&ok[:,:-1])[...,None]
    values=np.where(m,v/SCALES[round_no],0).astype(np.float32)
    # Copy the published core values exactly; avoid a new round-trip quantization.
    values[...,:6]=core['values'][...,sl];m[...,:6]=core['valid'][...,sl]
    if not np.isfinite(values).all(): raise ValueError('Feature overflow; no silent clipping')
    return {'values':values,'valid':m}

def view(ext,arm):
    if arm not in ARMS:raise ValueError('Unknown arm')
    x=np.asarray(ext['values']);m=np.asarray(ext['valid'])
    if x.ndim!=3 or x.shape[1:]!=(20,12) or m.shape!=x.shape or m.dtype!=np.bool_:
        raise ValueError('Invalid family tensor')
    if not np.isfinite(x[m]).all():raise ValueError('Nonfinite valid features')
    x=np.where(m,x,0).astype(np.float32)
    if arm=='mask':x[:]=0
    elif arm=='core':x[...,6:]=0
    return x,m.copy()

def summarize(arrays,round_no):
    _round(round_no);stats=[]
    for j,name in enumerate(NAMES[round_no]):
        den=sum(a['values'].shape[0]*20 for a in arrays)
        pieces=[a['values'][...,j][a['valid'][...,j]].astype(float)*SCALES[round_no][j] for a in arrays]
        x=np.concatenate(pieces) if pieces else np.empty(0)
        stats.append({'name':name,'unit':UNITS[round_no][j],'block':'core' if j<6 else 'extension',
            'valid':len(x),'slots':den,'support':len(x)/den if den else 0.,
            'rms':float(np.sqrt(np.mean(x*x))) if len(x) else None,
            'p01':float(np.quantile(x,.01)) if len(x) else None,'p99':float(np.quantile(x,.99)) if len(x) else None,
            'max_abs':float(np.max(np.abs(x))) if len(x) else None})
    return stats
