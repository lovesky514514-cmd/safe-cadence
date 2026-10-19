"""Certified online mini-batch SGD for the fixed-entity TransE setting.
At each step the candidate update is either certified for the specified
historic query list or rolled back. An expensive, simple research prototype.
Not a globally optimal constrained optimizer. Does not guarantee new-task fit.
"""
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .certificate_cache import CachedRankCertificate


def certified_minibatch_sgd(E0,R0,incoming,protected,history,
                            steps=60,batch_size=32,negatives=2,lr=.4,
                            k=10,seed=17,certify=True):
    torch.set_num_threads(1)
    rng=np.random.default_rng(seed)
    train=np.asarray(incoming,dtype=np.int64)
    all_seen=set(map(tuple,np.concatenate((history,incoming)).tolist()))
    E=np.asarray(E0,dtype=np.float64).copy()
    R=np.asarray(R0,dtype=np.float64).copy()
    checker=CachedRankCertificate(E,R,protected,history,k=k)
    successes=rejects=gate_calls=0
    train_t=cert_t=0.
    for step in range(steps):
        pos=train[rng.integers(0,len(train),batch_size)]
        neg=np.repeat(pos,negatives,axis=0).copy()
        for j in range(len(neg)):
            axis=0 if rng.integers(0,2)==0 else 2
            for trial in range(80):
                candidate=int(rng.integers(0,len(E)))
                neg[j,axis]=candidate
                if tuple(neg[j]) not in all_seen: break
            else:raise ValueError('Negative sample rejection exhausted')
        start=perf_counter()
        Ee=torch.tensor(E,dtype=torch.float32,requires_grad=True)
        Rr=torch.tensor(R,dtype=torch.float32,requires_grad=True)
        p=torch.tensor(pos,dtype=torch.long)
        n=torch.tensor(neg,dtype=torch.long)
        def distance(t):return torch.linalg.vector_norm(Ee[t[:,0]]+Rr[t[:,1]]-Ee[t[:,2]],dim=1)
        loss=F.relu(1.0+distance(p).repeat_interleave(negatives)-distance(n)).mean()
        ge,gr=torch.autograd.grad(loss,(Ee,Rr))
        ge=ge.numpy().astype(np.float64);gr=gr.numpy().astype(np.float64)
        train_t+=perf_counter()-start
        # Only updated rows are potentially changed; no global renormalization.
        accepted=False
        for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125):
            Enew=E-lr*alpha*ge;Rnew=R-lr*alpha*gr
            # Project entities onto unit ball without perturbing untouched embeddings.
            norms=np.linalg.norm(Enew,axis=1)
            exceed=norms>1.000000001
            if np.any(exceed):Enew[exceed]/=norms[exceed,None]
            if certify:
                t=perf_counter();verdict=checker.verify(Enew,Rnew,added_true=incoming,
                                                         stop_early=True)
                cert_t+=perf_counter()-t
                gate_calls+=1
                if not verdict.safe:continue
            E,R=Enew,Rnew;successes+=1;accepted=True
            break
        if not accepted:rejects+=1
    if certify:
        check=checker.verify_exhaustive(E,R,added_true=incoming)
        assert check.safe,'Certified SGD violated protected historic ranking!'
    return E,R,{'steps':steps,'accepted_steps':successes,'rejected_steps':rejects,
                'gate_calls':gate_calls,'train_seconds':train_t,'certificate_seconds':cert_t,
                'protected_queries':len(checker.protected),
                'index_build_seconds':checker.indexing_seconds,
                'certified':certify}
