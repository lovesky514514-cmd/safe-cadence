"""Batched exact-candidate gate for low/medium-N TransE.

No candidate pruning: computes conservative filtered rank for ALL candidates in
vectorized chunks. Intended to remove Python per-query and truth-set rebuilding
from stepwise certification. Equivalent to brute-force checker for the tested
float64 implementation; NOT a formal directed-rounding guarantee.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .certificate_v2 import SparseRadiusCertificate,CheckResult
from .ranking import TrueFacts
from scipy.spatial.distance import cdist

class BatchedRankCertificate(SparseRadiusCertificate):
    def __init__(self,old_entities,old_relations,protected_triples,known_true,k=10,
                 added_true=None,chunk_mb=24,chunk_queries=8,**kwargs):
        super().__init__(old_entities,old_relations,protected_triples,known_true,k=k,**kwargs)
        self.p=np.asarray([tr for tr,_ in self.protected],dtype=np.int64).reshape(-1,3)
        self.sides=np.asarray([s=='tail' for _,s in self.protected],dtype=np.bool_)
        self.answers=np.where(self.sides,self.p[:,2],self.p[:,0]) if len(self.p) else np.empty(0,dtype=np.int64)
        self.chunk_mb=chunk_mb
        self.chunk_queries=chunk_queries
        self.mask_base=None;self.truth_frozen=None
        self.configure_true_facts(added_true)

    def configure_true_facts(self,added_true=None):
        true=np.asarray(added_true,dtype=np.int64).reshape(-1,3) if added_true is not None else np.empty((0,3),dtype=np.int64)
        truth=TrueFacts(true)
        for key,vs in self.truth.heads.items(): truth.heads[key].update(vs)
        for key,vs in self.truth.tails.items(): truth.tails[key].update(vs)
        self.truth_frozen=true.copy()
        n=len(self.E0); q=len(self.p)
        mask=np.ones((q,n),dtype=bool)
        for j,(tr,side) in enumerate(self.protected):
            mask[j,self.answers[j]]=False
            other=truth.ignored(tr,side)
            if other: mask[j,list(other)]=False
        self.mask_base=mask

    def verify(self,new_entities,new_relations,added_true=None,exhaustive=False,stop_early=True):
        started=perf_counter()
        E=np.asarray(new_entities,dtype=np.float64)
        R=np.asarray(new_relations,dtype=np.float64)
        if E.ndim!=2 or R.ndim!=2 or E.shape[1]!=self.E0.shape[1] or R.shape[1]!=self.R0.shape[1]:
            raise ValueError('dimension mismatch')
        if len(E)<len(self.E0) or len(R)<len(self.R0):raise ValueError('entity/relation removal unsupported')
        if not np.isfinite(E).all() or not np.isfinite(R).all():raise ValueError('nonfinite candidate')
        if added_true is not None:
            if self.truth_frozen is None or not np.array_equal(np.asarray(added_true),self.truth_frozen):
                raise ValueError('Batched certificate requires fixed added_true for the update stream')
        n=len(E); q=len(self.p)
        mask=self.mask_base
        if n>mask.shape[1]:mask=np.pad(mask,((0,0),(0,n-mask.shape[1])),constant_values=True)
        limit=max(1,min(self.chunk_queries,int(self.chunk_mb*1048576/(n*E.shape[1]*8))))
        checked=0;inspected=0;full=0;failure=None
        for start in range(0,q,limit):
            end=min(start+limit,q);p=self.p[start:end];side=self.sides[start:end]
            query=np.where(side[:,None],E[p[:,0]]+R[p[:,1]],E[p[:,2]]-R[p[:,1]])
            distances=cdist(query,E,metric="euclidean")
            ans=self.answers[start:end]
            ans_dist=distances[np.arange(end-start),ans]
            ranks=1+np.count_nonzero((distances<=ans_dist[:,None]+1e-12) & mask[start:end],axis=1)
            checked+=end-start
            inspected+=int(mask[start:end].sum())
            full+=int(mask[start:end].sum())
            bad=np.flatnonzero(ranks>self.k)
            if bad.size:
                ix=start+int(bad[0]);failure=(self.protected[ix][0],self.protected[ix][1],int(ranks[bad[0]]))
                if stop_early:break
        return CheckResult(failure is None,checked,inspected,full,0.,perf_counter()-started,failure,
                           'rank_violation' if failure else 'certified','batched_exhaustive')
