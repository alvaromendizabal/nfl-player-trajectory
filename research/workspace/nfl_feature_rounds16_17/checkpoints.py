"""Immutable optimizer/RNG checkpoints, adapted unchanged from Round 8."""
from __future__ import annotations
import io, json
import torch
from audit_io import digest, safe_file, atomic_json, atomic as _atomic

def save_checkpoint(folder,model,opt,stepno,signature,training_loss,initial_hash):
    folder.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':stepno,
           'torch_rng':torch.get_rng_state(),'signature':signature,'losses':training_loss,'initial_hash':initial_hash}
    buf=io.BytesIO();torch.save(state,buf);payload=buf.getvalue()
    import hashlib
    h=hashlib.sha256(payload).hexdigest();name=f'{stepno:06d}-{h}.pt';path=folder/name
    if path.exists():
        if digest(path)!=h:raise ValueError('Existing checkpoint corrupted')
    else:_atomic(path,lambda f:f.write(payload))
    if digest(path)!=h:raise ValueError('Checkpoint readback hash failed')
    atomic_json(folder/'checkpoint.json',{'signature':signature,'step':stepno,'blob':name,'sha256':h,'bytes':len(payload),'initial_hash':initial_hash})
    return h


def load_checkpoint(folder,model,opt,signature):
    r=json.loads(safe_file(folder,'checkpoint.json').read_text());p=safe_file(folder,r['blob'])
    if r['signature']!=signature or digest(p)!=r['sha256']:raise ValueError('Checkpoint identity drift')
    state=torch.load(p,map_location='cpu',weights_only=True)
    if state['signature']!=signature or state['step']!=r['step']:raise ValueError('Checkpoint cursor mismatch')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
    return state


def weight_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().numpy().tobytes())
    return h.hexdigest()


