"""Fixed compact learned temporal representation; same parameters in both arms.

All convolutions operate inside the already-observed pre-throw window. Pooling
retains actual missingness. Attention is descriptive, not a coverage probability.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from sequence_features import terminal_view

CONFIG={'epochs':24,'batch_plays':8,'lr':0.001,'weight_decay':0.0001,'seed':20260912,
        'checkpoint_steps':25,'threads':2,'node_hidden':24,'pair_hidden':16,
        'decoder_hidden':64,'gradient_clip':5.0,'evaluation_fold':2}


def setup(seed=None):
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG['seed'] if seed is None else seed)


class Temporal(nn.Module):
    def __init__(self,c,h):
        super().__init__();self.frame=nn.Linear(2*c+1,h)
        self.conv=nn.Conv1d(h,h,3,padding=1)
        self.mix=nn.Linear(2*h,h)
    def forward(self,x,valid):
        shape=x.shape;T=shape[-2];c=shape[-1]
        x=x.reshape(-1,T,c);valid=valid.reshape(-1,T,c)
        present=valid.any(-1)
        clock=torch.linspace(-.95,0,T,device=x.device).view(1,T,1).expand(len(x),-1,-1)
        h=torch.nn.functional.silu(self.frame(torch.cat((torch.where(valid,x,0),valid.float(),clock),-1)))
        h=h*present[...,None]
        h=torch.nn.functional.silu(self.conv(h.transpose(1,2)).transpose(1,2))*present[...,None]
        mean=h.sum(1)/present.sum(1).clamp_min(1)[:,None]
        ix=torch.where(present,torch.arange(T,device=x.device),-1).max(1).values
        last=h[torch.arange(len(h),device=x.device),ix.clamp_min(0)]*(ix>=0)[:,None]
        ans=torch.nn.functional.silu(self.mix(torch.cat((mean,last),-1)))
        ans=ans*present.any(1)[:,None]
        return ans.reshape(*shape[:-2],-1)


class Forecast(nn.Module):
    def __init__(self):
        super().__init__();self.node=Temporal(10,24);self.pair=Temporal(8,16)
        self.peer=nn.Linear(16+8+5,16)
        self.attn=nn.Linear(16,1)
        self.decoder=nn.Sequential(nn.Linear(72+24+16,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,2))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def forward(self,node,node_valid,pair,pair_valid,pair_age,role,side,base,batch,player):
        n=node.shape[1];own=self.node(node,node_valid);edge=self.pair(pair,pair_valid)
        destrole=role[:,None].expand(-1,n,-1,-1)
        same=(side[:,:,None]==side[:,None,:]).float()[...,None]
        h=torch.nn.functional.silu(self.peer(torch.cat((edge,pair_age/2,destrole,same),-1)))
        exists=pair_valid.any(-1).any(-1)
        logits=self.attn(h).squeeze(-1).masked_fill(~exists,-1e9)
        w=torch.softmax(logits,-1)*exists
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        context=(h*w[...,None]).sum(2)
        return self.decoder(torch.cat((base,own[batch,player],context[batch,player]),-1))


def normalization(plays):
    x=np.concatenate([p['base'] for p in plays],0).astype(np.float64)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1
    return {'mean':mean.tolist(),'scale':scale.tolist()}


def collate(plays,arm,norm):
    if arm not in ('terminal','history'):raise ValueError('Unknown arm')
    n=max(len(p['ids']) for p in plays);b=len(plays)
    vals={'node':np.zeros((b,n,20,10),np.float32),'node_valid':np.zeros((b,n,20,10),bool),
          'pair':np.zeros((b,n,n,20,8),np.float32),'pair_valid':np.zeros((b,n,n,20,8),bool),
          'pair_age':np.zeros((b,n,n,8),np.float32),'role':np.zeros((b,n,4),np.float32),
          'side':np.zeros((b,n),np.int64)}
    base=[];batch=[];player=[]
    for i,p in enumerate(plays):
        k=len(p['ids'])
        for key in ('node','node_valid','role','side'):vals[key][i,:k]=p[key]
        for key in ('pair_valid','pair_age'):vals[key][i,:k,:k]=p[key]
        pair=p['pair'] if arm=='history' else terminal_view(p['pair'],p['pair_valid'])
        vals['pair'][i,:k,:k]=pair
        base.append((p['base']-np.asarray(norm['mean']))/np.asarray(norm['scale']))
        batch.extend([i]*len(p['base']));player.extend(p['query'].tolist())
    vals.update(base=np.asarray(np.concatenate(base),np.float32),batch=np.asarray(batch,np.int64),player=np.asarray(player,np.int64))
    if not all(np.isfinite(v).all() for v in vals.values()):raise ValueError('Nonfinite model input')
    return {k:torch.from_numpy(v) for k,v in vals.items()}


def ordered_batches(n,epoch,batch_size=None):
    ix=np.random.default_rng(CONFIG['seed']+epoch).permutation(n)
    bs=CONFIG['batch_plays'] if batch_size is None else batch_size
    return [ix[i:i+bs] for i in range(0,n,bs)]


def step(model,optimizer,batch,target,denominator):
    model.train();optimizer.zero_grad(set_to_none=True)
    prediction=model(**batch)
    # Constant denominator per optimizer batch. Longer trajectories are not given
    # the same total weight as short trajectories; all requested coordinates count.
    loss=(prediction-target).square().sum()/denominator
    if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),CONFIG['gradient_clip'],error_if_nonfinite=True)
    optimizer.step();return float(loss.detach())


def predict(model,plays,arm,norm):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(plays),CONFIG['batch_plays']):
            b=plays[i:i+CONFIG['batch_plays']]
            parts.append(model(**collate(b,arm,norm)).cpu().numpy().astype(np.float64))
    return np.concatenate(parts,0)
