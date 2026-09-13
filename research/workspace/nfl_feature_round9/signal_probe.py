"""Training-input counterfactual sensitivity, with zero optimizer steps.

Output-change RMS is NOT prediction error. Perturbations may be off-distribution.
Signed-output gradient projections are sensitivity probes, not feature importance
or proof that an input improves held-out performance. No targets are accepted.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch


def weights_digest(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():
        h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def reverse_valid(values,valid):
    """Reverse each channel's valid observed values without changing masks/slots."""
    out=values.clone()
    for b in range(len(values)):
        for i in range(values.shape[1]):
            for j in range(values.shape[2]):
                for c in range(values.shape[-1]):
                    ix=torch.where(valid[b,i,j,:,c])[0]
                    out[b,i,j,ix,c]=values[b,i,j,ix.flip(0),c]
    return out


def rms_sum(a,b):
    d=(a.detach().double()-b.detach().double())
    return {'sum_squared_change':float(d.square().sum()),'coordinate_count':d.numel(),
            'max_absolute_change':float(d.abs().max())}


def prediction_with_branch_removed(model,batch,branch):
    sections={'base':slice(0,72),'node':slice(72,96),'context':slice(96,112)}
    if branch not in sections:raise ValueError('Unknown decoder branch')
    def hook(module,args):
        x=args[0].clone();x[:,sections[branch]]=0
        return (x,)
    handle=model.decoder[0].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():return model(**batch)
    finally:handle.remove()


def probe(model,play,arm,norm,collate):
    """Compare supplied information using the same frozen weights for each change."""
    model.eval();before=weights_digest(model)
    batch=collate([play],arm,norm)
    # Model parameter grads are not needed. Input grads are calculated without labels.
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():
        reference=model(**batch)
        changed=collate([play],'history' if arm=='terminal' else 'terminal',norm)
        variants={'swap_pair_view':model(**changed)}
        b=dict(batch);b['pair']=reverse_valid(batch['pair'],batch['pair_valid'])
        variants['reverse_valid_pair_values']=model(**b)
        b=dict(batch);b['pair']=torch.zeros_like(batch['pair'])
        variants['zero_pair_values_keep_masks']=model(**b)
        # Zero only one observed channel at a time, retaining masks as a stress test.
        channels=[]
        for c in range(8):
            b=dict(batch);b['pair']=batch['pair'].clone();b['pair'][...,c]=0
            channels.append(rms_sum(reference,model(**b)))
    for name in ('context','node','base'):
        variants[f'zero_decoder_{name}']=prediction_with_branch_removed(model,batch,name)
    changes={name:rms_sum(reference,pred) for name,pred in variants.items()}
    # Two fixed Rademacher projections estimate local per-channel Jacobian energy.
    # A signed projection avoids cancellation from simply summing all predictions.
    gradients=np.zeros(8);counts=batch['pair_valid'].sum(dim=(0,1,2,3)).cpu().numpy()
    for seed in (1103,1109):
        b=dict(batch);b['pair']=batch['pair'].detach().clone().requires_grad_(True)
        prediction=model(**b)
        signs=torch.from_numpy(np.random.default_rng(seed).choice([-1.,1.],size=tuple(prediction.shape)).astype(np.float32))
        g=torch.autograd.grad((prediction*signs).sum(),b['pair'],allow_unused=True)[0]
        if g is not None:
            g=torch.where(batch['pair_valid'],g,0).double()
            gradients+=g.square().sum(dim=(0,1,2,3)).cpu().numpy()/2
    if weights_digest(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Signal audit changed weights or accumulated parameter gradients')
    return {'rows':len(reference),'changes':changes,'channel_zeroing':channels,
            'gradient_energy':gradients.tolist(),'valid_counts':counts.tolist()}


def input_contrast(play,terminal_view):
    h=play['pair'];v=play['pair_valid'];t=terminal_view(h,v)
    delta=np.where(v,h.astype(float)-t.astype(float),0)
    return {'sum_squared_difference':np.square(delta).sum(axis=(0,1,2)).tolist(),
            'valid_counts':v.sum(axis=(0,1,2)).tolist(),
            'different_counts':((h!=t)&v).sum(axis=(0,1,2)).tolist()}


def weight_changes(initial,model):
    result=[]
    for group in ('node','pair','peer','attn','decoder'):
        old=[];new=[]
        for name,value in model.state_dict().items():
            if name.split('.')[0]==group:
                new.append(value.detach().cpu().numpy().ravel().astype(float))
                old.append(initial.state_dict()[name].detach().cpu().numpy().ravel().astype(float))
        a=np.concatenate(old);b=np.concatenate(new)
        denom=float(np.linalg.norm(a))
        result.append({'group':group,'parameters':len(a),'initial_l2':denom,
                       'delta_l2':float(np.linalg.norm(b-a)),
                       'relative_delta_l2':float(np.linalg.norm(b-a)/denom) if denom else None})
    return result
