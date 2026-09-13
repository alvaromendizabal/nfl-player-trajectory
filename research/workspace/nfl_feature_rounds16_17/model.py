"""Shared motion anchor + one 12-channel family, three equal-capacity arms.

The Round 13 motion values and masks are present in ALL arms. New candidate
availability masks are also shared. Only the new 12 numerical values differ.
The anchor is promising exploratory evidence, not a promoted production model.
"""
from __future__ import annotations
import numpy as np
import torch
from frozen_model import (Forecast as Parent, Temporal, CONFIG, setup, normalization,
                          ordered_batches, step, collate as base_collate)
from families import ARMS, view

class Forecast(Parent):
    def __init__(self):
        super().__init__()
        self.node = Temporal(34, 24)  # Original 10 + motion anchor 12 + candidate 12.

def collate(plays, arm, norm):
    if arm not in ARMS or not plays:
        raise ValueError('Nonempty plays and a declared arm required')
    base = base_collate(plays, 'cartesian', norm)
    b, n = base['node'].shape[:2]
    values = np.zeros((b,n,20,24),np.float32)
    masks = np.zeros_like(values,bool)
    for i,p in enumerate(plays):
        x,m = view({'values':p['extra'],'valid':p['extra_valid']},arm)
        mv,mm = np.asarray(p['motion']),np.asarray(p['motion_valid'])
        if mv.shape != x.shape or mm.shape != m.shape or mm.dtype != np.bool_ or len(x)!=len(p['ids']):
            raise ValueError('Motion anchor/candidate dimension mismatch')
        if not np.isfinite(mv[mm]).all():
            raise ValueError('Invalid motion anchor')
        values[i,:len(x),:,:12] = np.where(mm,mv,0)
        masks[i,:len(x),:,:12] = mm
        values[i,:len(x),:,12:] = x
        masks[i,:len(x),:,12:] = m
    base['node'] = torch.cat((base['node'],torch.from_numpy(values)),dim=-1)
    base['node_valid'] = torch.cat((base['node_valid'],torch.from_numpy(masks)),dim=-1)
    return base

def predict(model,plays,arm,norm):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(plays),CONFIG['batch_plays']):
            batch=plays[start:start+CONFIG['batch_plays']]
            out.append(model(**collate(batch,arm,norm)).numpy().astype(np.float64))
    if not out:
        raise ValueError('No forecast rows')
    return np.concatenate(out)
