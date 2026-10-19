"""Transactional alternative for keeping DEPLOYED snapshots ranking-safe.

Training occurs on a private model. The public pointer is atomically replaced
only after a rank-safe proposed commit. Staging may be unsafe and is never
served. This baseline directly challenges the practical necessity of a safety
gate on *every internal SGD step*.

This provides published-checkpoint protection only. No claim about safety of
internal optimizer states, which are deliberately not deployed.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .batched_certificate_v4 import BatchedRankCertificate
from .ranking import filtered_metrics


def staged_train(E,R,batches,lr=.45,anchor=0.,E0=None,R0=None):
    E=E.copy();R=R.copy()
    for pos,neg in batches:
        ee=torch.tensor(E,dtype=torch.float32,requires_grad=True)
        rr=torch.tensor(R,dtype=torch.float32,requires_grad=True)
        def dist(x):return torch.linalg.vector_norm(ee[x[:,0]]+rr[x[:,1]]-ee[x[:,2]],dim=1)
        loss=F.relu(1.+dist(pos).repeat_interleave(len(neg)//len(pos))-dist(neg)).mean()
        if anchor:
            loss+=anchor*.5*((ee-torch.tensor(E0,dtype=torch.float32)).square().sum()+
                              (rr-torch.tensor(R0,dtype=torch.float32)).square().sum())
        ge,gr=torch.autograd.grad(loss,(ee,rr))
        E-=lr*ge.detach().numpy().astype(np.float64)
        R-=lr*gr.detach().numpy().astype(np.float64)
        norms=np.linalg.norm(E,axis=1);mask=norms>1.000000001
        if np.any(mask):E[mask]/=norms[mask,None]
    return E,R


def transactional_updates(E0,R0,old,incoming,protected,batches,
                          commit_every=10,lr=.45,anchor=0.,k=10):
    torch.set_num_threads(1)
    t=perf_counter()
    checker=BatchedRankCertificate(E0,R0,protected,old,k=k,added_true=incoming)
    setup=perf_counter()-t
    committed_E=np.asarray(E0).copy();committed_R=np.asarray(R0).copy()
    committed=0;rollbacks=0;rescue=0;checks=0;compute=0.;verify=0.
    trace=[]
    for begin in range(0,len(batches),commit_every):
        chunk=batches[begin:begin+commit_every]
        t=perf_counter()
        draft_E,draft_R=staged_train(committed_E,committed_R,chunk,lr=lr,anchor=anchor,E0=E0,R0=R0)
        compute+=perf_counter()-t
        # Protect the COMMITTED model's visible sequence only.
        accepted=False
        for j,alpha in enumerate([1.,.5,.25,.125,.0625,.03125,.015625,.0078125]):
            t=perf_counter()
            trial_E=committed_E+alpha*(draft_E-committed_E)
            trial_R=committed_R+alpha*(draft_R-committed_R)
            safe=checker.verify(trial_E,trial_R,added_true=incoming).safe
            verify+=perf_counter()-t;checks+=1
            if safe:
                committed_E,committed_R=trial_E,trial_R
                accepted=True;committed+=1;rescue+=(j>0)
                break
        if not accepted:rollbacks+=1
        trace.append({'segment_end':min(begin+commit_every,len(batches)),
                      'published':accepted,'backtracked':accepted and (j>0),
                      'always_protected':checker.verify(committed_E,committed_R,added_true=incoming).safe})
    truth=np.concatenate((old,incoming))
    return committed_E,committed_R,dict(commit_every=commit_every,lr=lr,anchor=anchor,
       commits=committed,rollbacks=rollbacks,rescued= rescue,gate_calls=checks,
       method_seconds=setup+compute+verify,train_seconds=compute,gate_seconds=verify,index_seconds=setup,
       published_checkpoints_safe=all(x['always_protected'] for x in trace),
       protected_queries=len(checker.protected),trace=trace,
       new_mrr=filtered_metrics(committed_E,committed_R,incoming,truth,max_eval=100)['mrr'],
       old_mrr=filtered_metrics(committed_E,committed_R,old,truth,max_eval=100)['mrr'])
