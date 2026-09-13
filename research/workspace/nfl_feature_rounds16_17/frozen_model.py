"""Fixed two-arm group-separated relationship encoder.

Architecture changes versus Round 8 are shared, not individually attributed:
role-group pooling, query-conditioned attention and normalized context. The sole
within-study difference is the six goal-frame numerical channels. The control
retains their masks, so the contrast does not measure availability effects.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from grouped_features import ARMS, pair_view

CONFIG = {'epochs':24,'batch_plays':8,'lr':0.001,'weight_decay':0.0001,
          'seed':20260912,'checkpoint_steps':25,'threads':2,
          'gradient_clip':5.0,'evaluation_fold':2}


def setup(seed: int | None = None) -> None:
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG['seed'] if seed is None else seed)


class Temporal(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.frame=nn.Linear(2*channels+1,hidden)
        self.conv=nn.Conv1d(hidden,hidden,3,padding=1)
        self.mix=nn.Linear(2*hidden,hidden)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        shape=x.shape; t=shape[-2]; c=shape[-1]
        x=x.reshape(-1,t,c); valid=valid.reshape(-1,t,c)
        seen=valid.any(-1)
        clock=torch.linspace(-0.95,0,t,device=x.device).view(1,t,1).expand(len(x),-1,-1)
        h=torch.nn.functional.silu(self.frame(torch.cat((torch.where(valid,x,0),valid.float(),clock),-1)))
        h=h*seen[...,None]
        h=torch.nn.functional.silu(self.conv(h.transpose(1,2)).transpose(1,2))*seen[...,None]
        mean=h.sum(1)/seen.sum(1).clamp_min(1)[:,None]
        ix=torch.where(seen,torch.arange(t,device=x.device),-1).max(1).values
        last=h[torch.arange(len(h),device=x.device),ix.clamp_min(0)]*(ix>=0)[:,None]
        ans=torch.nn.functional.silu(self.mix(torch.cat((mean,last),-1)))*seen.any(1)[:,None]
        return ans.reshape(*shape[:-2],-1)


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.node=Temporal(10,24); self.pair=Temporal(14,16)
        self.peer=nn.Linear(16+14+4+1,16)
        self.key=nn.Linear(16,16,bias=False)
        self.query_encoder=nn.Linear(24+4+1,16,bias=False)
        # Non-affine normalization: no arbitrary trainable branch multiplier.
        self.context_norm=nn.LayerNorm(16,elementwise_affine=False)
        self.node_norm=nn.LayerNorm(24,elementwise_affine=False)
        self.decoder=nn.Sequential(nn.Linear(72+24+4*16+4,64),nn.SiLU(),
                                   nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, node, node_valid, pair, pair_valid, pair_age, role, side,
                groups, base, batch, player, query_time, *, zero_context=False):
        n=node.shape[1]
        own=self.node(node,node_valid)
        edge=self.pair(pair,pair_valid)
        dest=role[:,None].expand(-1,n,-1,-1)
        same=(side[:,:,None]==side[:,None,:]).float()[...,None]
        h=torch.nn.functional.silu(self.peer(torch.cat((edge,pair_age/2,dest,same),-1)))
        exists=pair_valid.any(-1).any(-1)
        h=h*exists[...,None]
        requested=h[batch,player]
        q=self.query_encoder(torch.cat((own[batch,player],role[batch,player],query_time),-1))
        logits=(self.key(requested)*q[:,None]).sum(-1)/math.sqrt(16)
        # Groups overlap intentionally (a receiver is also teammate or opponent).
        valid=groups[batch,player] & exists[batch,player,:,None]
        score=logits[...,None].expand_as(valid).masked_fill(~valid,-1e9)
        weights=torch.softmax(score,dim=1)*valid
        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        contexts=torch.einsum('qng,qnd->qgd',weights,requested)
        present=valid.any(1)
        contexts=self.context_norm(contexts)*present[...,None]
        if zero_context:
            contexts=torch.zeros_like(contexts)
        data=torch.cat((base,self.node_norm(own[batch,player]),contexts.flatten(1),present.float()),-1)
        return self.decoder(data)


def normalization(plays):
    x=np.concatenate([p['base'] for p in plays]).astype(np.float64)
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-6]=1
    return {'mean':mean.tolist(),'scale':scale.tolist()}


def collate(plays, arm, norm):
    if arm not in ARMS or not plays:
        raise ValueError('Nonempty batch and known arm required')
    n=max(len(p['ids']) for p in plays); b=len(plays)
    if not 1<=n<=22:
        raise ValueError('Invalid player count')
    vals={'node':np.zeros((b,n,20,10),np.float32),'node_valid':np.zeros((b,n,20,10),bool),
          'pair':np.zeros((b,n,n,20,14),np.float32),'pair_valid':np.zeros((b,n,n,20,14),bool),
          'pair_age':np.zeros((b,n,n,14),np.float32),'role':np.zeros((b,n,4),np.float32),
          'side':np.zeros((b,n),np.int64),'groups':np.zeros((b,n,n,4),bool)}
    base=[]; batch=[]; player=[]; qt=[]
    for i,p in enumerate(plays):
        k=len(p['ids']); q=np.asarray(p['query'])
        if not np.issubdtype(q.dtype,np.integer) or (q<0).any() or (q>=k).any():
            raise ValueError('Invalid requested player slot')
        for name in ('node','node_valid','role','side'):
            vals[name][i,:k]=p[name]
        pv,mask,age=pair_view(p,arm)
        vals['pair'][i,:k,:k]=pv; vals['pair_valid'][i,:k,:k]=mask
        vals['pair_age'][i,:k,:k]=age
        vals['groups'][i,:k,:k]=np.asarray(p['groups']).any(2)
        base.append((p['base']-np.asarray(norm['mean']))/np.asarray(norm['scale']))
        batch.extend([i]*len(q)); player.extend(q.tolist())
        keys=p['keys']
        if keys.shape!=(len(q),4) or (keys[:,3]<1).any():
            raise ValueError('Invalid forecast keys')
        qt.extend((keys[:,3]/50.0).tolist()) # 10 Hz and a fixed five-second scale.
    vals.update(base=np.concatenate(base).astype(np.float32),batch=np.asarray(batch,np.int64),
                player=np.asarray(player,np.int64),query_time=np.asarray(qt,np.float32)[:,None])
    # Masked poison is removed rather than treated as a measurement.
    vals['node']=np.where(vals['node_valid'],vals['node'],0)
    if not all(np.isfinite(v).all() for v in vals.values()):
        raise ValueError('Nonfinite model inputs')
    return {k:torch.from_numpy(v) for k,v in vals.items()}


def ordered_batches(n, epoch):
    ix=np.random.default_rng(CONFIG['seed']+epoch).permutation(n)
    bs=CONFIG['batch_plays']
    return [ix[i:i+bs] for i in range(0,n,bs)]


def step(model, optimizer, batch, target, denominator):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss=(model(**batch)-target).square().sum()/denominator
    if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def predict(model, plays, arm, norm):
    model.eval(); values=[]
    with torch.no_grad():
        for i in range(0,len(plays),CONFIG['batch_plays']):
            values.append(model(**collate(plays[i:i+CONFIG['batch_plays']],arm,norm)).numpy().astype(np.float64))
    return np.concatenate(values)
