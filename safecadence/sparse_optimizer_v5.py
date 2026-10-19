"""Explicit sparse-entity SGD for true support-by-construction overlay proposals.

Relation vectors are frozen.  Each update touches only the current positive and
negative batch endpoints, and only the cumulative changed support is included
in the certification patch relative to the original safety checkpoint.

This is an experimental restricted optimizer, NOT a replacement for full CKGE.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .sparse_overlay_v5 import SparseOverlayCertificate,SparsePatch
from .batched_certificate_v4 import BatchedRankCertificate
from .ranking import TrueFacts, exact_rank, filtered_metrics


def sparse_entity_margin_gradient(E,R,pos,neg,margin=1.):
    """Return sorted touched IDs and gradient for all active margin pairs.

    Inputs use standard TransE per-negative pairwise margin ranking loss.
    Its relation gradient is intentionally discarded; R is frozen.
    """
    pos=np.asarray(pos,dtype=np.int64);neg=np.asarray(neg,dtype=np.int64)
    if pos.ndim!=2 or neg.ndim!=2 or pos.shape[1]!=3 or neg.shape[1]!=3 or len(neg)%len(pos):
        raise ValueError('bad positive or negative batch')
    mult=len(neg)//len(pos)
    repeated=np.repeat(pos,mult,axis=0)
    touched=np.unique(np.concatenate((repeated[:,0],repeated[:,2],neg[:,0],neg[:,2])))
    gp=E[repeated[:,0]]+R[repeated[:,1]]-E[repeated[:,2]]
    gn=E[neg[:,0]]+R[neg[:,1]]-E[neg[:,2]]
    dp=np.linalg.norm(gp,axis=1)
    dn=np.linalg.norm(gn,axis=1)
    active=(margin+dp-dn)>0
    gp=np.divide(gp,dp[:,None],out=np.zeros_like(gp),where=dp[:,None]>0)
    gn=np.divide(gn,dn[:,None],out=np.zeros_like(gn),where=dn[:,None]>0)
    gp[~active]=0;gn[~active]=0
    gp/=len(neg);gn/=len(neg)
    grad=np.zeros((len(touched),E.shape[1]),dtype=np.float64)
    for ent,sign,vec in ((repeated[:,0],1,gp),(repeated[:,2],-1,gp),
                         (neg[:,0],-1,gn),(neg[:,2],1,gn)):
        np.add.at(grad,np.searchsorted(touched,ent),sign*vec)
    loss=float(np.maximum(0,margin+dp-dn).mean())
    return touched,grad,loss


@dataclass
class SparseSGDRun:
    E:np.ndarray
    R:np.ndarray
    stats:dict


def run_certified_sparse_entity(E0,R0,old,incoming,protected,batches,
                                lr=.45,k=10,shortlist_size=24,
                                engine='overlay', max_levels=8):
    """Compare an overlay engine with batched exhaustive on identical proposals.

    Engine 'overlay' uses sparse-by-construction patches; engine 'batched'
    materializes a dense trial only because its reference verifier needs it.
    No hidden offset to audit walltime is included in method_wall_seconds.
    """
    start=perf_counter()
    E0=np.asarray(E0,dtype=np.float64);R0=np.asarray(R0,dtype=np.float64)
    E=E0.copy();R=R0.copy()
    truth=np.concatenate((np.asarray(old,dtype=np.int64),np.asarray(incoming,dtype=np.int64)))
    if engine=='overlay':
        gate=SparseOverlayCertificate(E0,R0,protected,old,k=k,shortlist_size=shortlist_size,added_true=incoming)
    elif engine=='batched':
        gate=BatchedRankCertificate(E0,R0,protected,old,k=k,added_true=incoming)
    else:raise ValueError('unsupported engine')
    setup=perf_counter()-start
    support=np.empty((0,),dtype=np.int64)
    accepts=rescues=rejects=calls=0
    gradient_time=gate_time=0.
    scored=full=0
    levels=[.5**j for j in range(max_levels)]
    for pos,neg in batches:
        pos=pos.numpy() if hasattr(pos,'numpy') else pos
        neg=neg.numpy() if hasattr(neg,'numpy') else neg
        t=perf_counter();ids,grad,_=sparse_entity_margin_gradient(E,R,pos,neg)
        gradient_time+=perf_counter()-t
        accepted=False
        cumulative=np.union1d(support,ids)
        for j,alpha in enumerate(levels):
            trial=E[ids]-lr*alpha*grad
            norms=np.linalg.norm(trial,axis=1)
            normalize=norms>1.000000001
            if normalize.any():trial[normalize]/=norms[normalize,None]
            t=perf_counter()
            if engine=='overlay':
                proposal=E[cumulative].copy()
                proposal[np.searchsorted(cumulative,ids)]=trial
                decision=gate.verify_patch(SparsePatch(cumulative,proposal,R),added_true=incoming)
            else:
                dense=E.copy();dense[ids]=trial
                decision=gate.verify(dense,R,added_true=incoming)
            gate_time+=perf_counter()-t
            calls+=1;scored+=decision.inspected_candidates;full+=decision.full_candidates
            if decision.safe:
                E[ids]=trial
                support=cumulative
                accepts+=1
                rescues+=(j>0)
                accepted=True
                break
        if not accepted:rejects+=1
    method=setup+gradient_time+gate_time
    endpoint=BatchedRankCertificate(E0,R0,protected,old,k=k,added_true=incoming).verify(E,R,added_true=incoming)
    mnew=filtered_metrics(E,R,incoming,truth,max_eval=100)['mrr']
    mold=filtered_metrics(E,R,old,truth,max_eval=100)['mrr']
    return SparseSGDRun(E,R,dict(engine=engine,method_wall_seconds=method,index_build_seconds=setup,
        gradient_seconds=gradient_time,verification_seconds=gate_time,steps=len(batches),
        accepted=accepts,rescued=rescues,rejected=rejects,gate_calls=calls,
        cumulative_support=len(support),support_fraction=len(support)/len(E0),
        inspected_fraction=scored/max(1,full),endpoint_safe=endpoint.safe,
        new_mrr=mnew,old_mrr=mold,protected_query_count=len(gate.protected)))
