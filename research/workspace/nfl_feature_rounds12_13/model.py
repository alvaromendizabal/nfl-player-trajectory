"""Equal-capacity masks/core/full models. Only 12 player-frame values vary by arm.

Both rounds share this encoder, the original optimizer/exposure, and the same
split. Existing group/goal availability masks are held fixed; numerical goal-pair
values remain zero in every arm (the original Cartesian setting).
"""
from __future__ import annotations
import numpy as np
import torch
from frozen_model import (Forecast as Parent,Temporal,CONFIG,setup,normalization,
                          ordered_batches,step,collate as base_collate)
from families import ARMS,view

class Forecast(Parent):
    def __init__(self):
        super().__init__()
        self.node=Temporal(22,24)

def collate(plays,arm,norm):
    if arm not in ARMS or not plays:raise ValueError('Nonempty batch and known arm required')
    base=base_collate(plays,'cartesian',norm);b,n=base['node'].shape[:2]
    extra=np.zeros((b,n,20,12),np.float32);valid=np.zeros_like(extra,bool)
    for i,p in enumerate(plays):
        x,m=view({'values':p['extra'],'valid':p['extra_valid']},arm)
        if len(x)!=len(p['ids']):raise ValueError('Extra player dimension mismatch')
        extra[i,:len(x)]=x;valid[i,:len(x)]=m
    base['node']=torch.cat((base['node'],torch.from_numpy(extra)),dim=-1)
    base['node_valid']=torch.cat((base['node_valid'],torch.from_numpy(valid)),dim=-1)
    return base

def predict(model,plays,arm,norm):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(plays),CONFIG['batch_plays']):
            out.append(model(**collate(plays[start:start+CONFIG['batch_plays']],arm,norm)).numpy().astype(np.float64))
    return np.concatenate(out)
