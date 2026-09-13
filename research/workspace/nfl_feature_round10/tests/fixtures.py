"""Analytic observed-only fixtures; no real NFL data."""
from pathlib import Path
import numpy as np

def sample(play=1,n=4):
    t=np.arange(20)*.1
    xy=np.stack([np.c_[20+2*i+3*t+.1*i*t*t, 8+2*i+.2*(i+1)*t] for i in range(n)])
    vel=np.stack([np.c_[3+.2*i*t, np.full(20,.2*(i+1))] for i in range(n)])
    goal=np.array([42.,17.])-xy;distance=np.linalg.norm(goal,axis=-1)
    node=np.concatenate([xy-xy[:,-1,None,:],vel,goal,np.zeros_like(xy),distance[...,None],
                         ((goal*vel).sum(-1)/distance)[...,None]],-1)
    node=node/np.array([10,10,10,10,20,20,1,1,20,10])
    delta=xy[None]-xy[:,None];dv=vel[None]-vel[:,None];dd=np.linalg.norm(delta,axis=-1)
    speed=np.linalg.norm(vel,axis=-1)
    values=np.concatenate([delta,dv,dd[...,None],(-(delta*dv).sum(-1)/np.maximum(dd,.1))[...,None],
                           ((delta[...,0]*dv[...,1]-delta[...,1]*dv[...,0])/np.maximum(dd,.1))[...,None],
                           ((vel[:,None]*vel[None]).sum(-1)/(speed[:,None]*speed[None]))[...,None]],-1)
    pv=np.broadcast_to((~np.eye(n,dtype=bool))[:,:,None,None],(n,n,20,8)).copy()
    values=np.where(pv,values/np.array([20,20,10,10,20,10,10,1]),0).astype(np.float32)
    base=np.zeros((8,72),np.float64);base[:,0]=np.linspace(.1,2,8);base[:,1]=play*.001
    role=np.eye(4)[np.arange(n)%4].astype(np.float32)
    return {'ids':np.arange(10,10+n,dtype=np.int64),'node':node.astype(np.float32),
            'node_valid':np.ones((n,20,10),bool),'pair':values,'pair_valid':pv,'pair_age':np.zeros((n,n,8),np.float32),
            'role':role,'side':np.arange(n,dtype=np.int64)%2,'node_age':np.zeros(n,np.float32),
            'query':np.arange(8,dtype=np.int64)%n,'base':base,'train':np.array(True),
            'signature':np.array('fixture')}

def model_sample(play=1,n=4):
    from grouped_features import make_extension
    p=sample(play,n)
    p.update(make_extension(p))
    p['keys']=np.column_stack([np.full(8,2023091000),np.full(8,play),p['ids'][p['query']],np.arange(1,9)])
    p['y']=np.column_stack([np.sin(np.arange(8)+play),np.cos(np.arange(8)+play)]).astype(np.float64)
    p['role_query']=p['query']%4
    return p
