"""Diagonal-gradient-importance regularization prototype (EWC-style).
Importance = mean squared batch gradient of old-task margin objective, normalized
by average importance. Empirical, not a Fisher-likelihood exact estimate.
Training cost of importance estimation MUST be reported separately.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .transe import TrainConfig

def estimate_diagonal_importance(triples,E,R,n_batches=20,batch_size=128,
                                 n_negatives=3,margin=1.0,seed=11):
    torch.set_num_threads(1)
    rng=torch.Generator(device='cpu').manual_seed(seed)
    pos=torch.tensor(triples,dtype=torch.long)
    ent=torch.tensor(E,dtype=torch.float32,requires_grad=True)
    rel=torch.tensor(R,dtype=torch.float32,requires_grad=True)
    weights_e=torch.zeros_like(ent);weights_r=torch.zeros_like(rel)
    facts=set(map(tuple,np.asarray(triples,dtype=np.int64).tolist()))
    for _ in range(n_batches):
        batch=pos[torch.randint(len(pos),(batch_size,),generator=rng)]
        neg=batch.repeat_interleave(n_negatives,dim=0).clone()
        side=torch.randint(0,2,(len(neg),),generator=rng)
        corr=torch.randint(len(ent),(len(neg),),generator=rng)
        neg[side==0,0]=corr[side==0]
        neg[side==1,2]=corr[side==1]
        for j in range(len(neg)):
            axis=0 if side[j]==0 else 2
            for tries in range(20):
                if tuple(neg[j].tolist()) not in facts: break
                neg[j,axis]=torch.randint(len(ent),(1,),generator=rng).item()
        def distance(t):
            return torch.linalg.vector_norm(ent[t[:,0]]+rel[t[:,1]]-ent[t[:,2]],dim=1)
        loss=F.relu(margin+distance(batch).repeat_interleave(n_negatives)-distance(neg)).mean()
        egrad,rgrad=torch.autograd.grad(loss,(ent,rel))
        weights_e+=egrad.detach().square()
        weights_r+=rgrad.detach().square()
    weights_e/=n_batches;weights_r/=n_batches
    # Regularize importance to avoid all-zero and extreme gradient estimates.
    weights_e=(weights_e/(weights_e.mean()+1e-12)).clamp(min=.01,max=50)
    weights_r=(weights_r/(weights_r.mean()+1e-12)).clamp(min=.01,max=50)
    return weights_e.numpy().astype(np.float64),weights_r.numpy().astype(np.float64)
