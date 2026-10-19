"""Strict TransE Top-k preservation using sparse-update-aware radius pruning.
For any unchanged entity c, d(q',c) >= d(q,c) - ||q'-q||.
If d(q,c) > d(q',a') + ||q'-q|| then c cannot outrank answer a.
Moved existing entities and newly appended entities are checked exactly.
Every accepted result is equivalent to exhaustive checking of protected queries.
No semantic correctness or unprotected-query guarantees are claimed.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from .ranking import query_and_answer, exact_rank, TrueFacts

@dataclass
class CheckResult:
    safe: bool
    checked_queries: int
    inspected_candidates: int
    full_candidates: int
    indexing_seconds: float
    verification_seconds: float
    failed_query: tuple | None = None
    reason: str = 'certified'
    method: str = 'sparse_radius'

class SparseRadiusCertificate:
    def __init__(self, old_entities, old_relations, protected_triples,
                 known_true, k=10, sides=('head','tail'), min_index_entities=320):
        self.E0=np.array(old_entities, dtype=np.float64, copy=True)
        self.R0=np.array(old_relations, dtype=np.float64, copy=True)
        if self.E0.ndim!=2 or self.R0.ndim!=2 or self.E0.shape[1]!=self.R0.shape[1]:
            raise ValueError('Entity/relation dimensions must agree')
        if not np.isfinite(self.E0).all() or not np.isfinite(self.R0).all():
            raise ValueError('Nonfinite initial embedding')
        self.truth=TrueFacts(known_true)
        self.k=int(k)
        if self.k<1: raise ValueError('k must be >=1')
        self.min_index_entities=int(min_index_entities)
        started=perf_counter()
        self.tree=cKDTree(self.E0)
        self.indexing_seconds=perf_counter()-started
        self.protected=[]; self.not_initially_protected=0
        for triple in np.asarray(protected_triples,dtype=np.int64):
            for side in sides:
                if exact_rank(self.E0,self.R0,triple,side,self.truth)<=self.k:
                    self.protected.append((tuple(map(int,triple)),side))
                else: self.not_initially_protected+=1

    def verify(self, new_entities, new_relations, added_true=None,
               exhaustive=False, stop_early=True):
        started=perf_counter()
        E1=np.asarray(new_entities,dtype=np.float64)
        R1=np.asarray(new_relations,dtype=np.float64)
        n0=len(self.E0)
        if E1.ndim!=2 or R1.ndim!=2 or E1.shape[1]!=self.E0.shape[1] or R1.shape[1]!=self.R0.shape[1]:
            raise ValueError('Dimension mismatch')
        if len(E1)<n0 or len(R1)<len(self.R0):
            raise ValueError('Entity or relation removal unsupported')
        if not np.isfinite(E1).all() or not np.isfinite(R1).all():
            raise ValueError('Nonfinite embeddings')
        if added_true is None: truth=self.truth
        else:
            truth=TrueFacts(added_true)
            for key,values in self.truth.heads.items(): truth.heads[key].update(values)
            for key,values in self.truth.tails.items(): truth.tails[key].update(values)
        drift=np.linalg.norm(E1[:n0]-self.E0,axis=1)
        changed=np.flatnonzero(drift!=0)
        moved=np.zeros(n0,dtype=bool); moved[changed]=True
        checked=inspected=full=0; failure=None
        use_tree=not exhaustive and n0>=self.min_index_entities
        for triple,side in self.protected:
            q0,answer=query_and_answer(self.E0,self.R0,triple,side)
            q1,_=query_and_answer(E1,R1,triple,side)
            da=float(np.linalg.norm(q1-E1[answer]))
            dq=float(np.linalg.norm(q1-q0))
            excluded=truth.ignored(triple,side)
            if exhaustive or not use_tree:
                ids=np.arange(len(E1),dtype=np.int64)
            else:
                slack=1e-9*(1+da+dq+float(np.linalg.norm(q0)))
                near=self.tree.query_ball_point(q0,da+dq+slack)
                stable=np.asarray([i for i in near if not moved[i]],dtype=np.int64)
                ids=np.concatenate((stable,changed,np.arange(n0,len(E1),dtype=np.int64)))
            mask=ids!=answer
            if excluded: mask &= ~np.isin(ids,list(excluded))
            ids=ids[mask]
            rank=1+int(np.count_nonzero(np.linalg.norm(E1[ids]-q1,axis=1)<=da))
            full+=len(E1)-1-len(excluded)
            inspected+=len(ids); checked+=1
            if rank>self.k:
                failure=(triple,side,rank)
                if stop_early: break
        return CheckResult(failure is None,checked,inspected,full,
                           self.indexing_seconds if use_tree else 0.,
                           perf_counter()-started,failure,
                           'rank_violation' if failure else 'certified',
                           'exhaustive' if exhaustive else ('sparse_radius' if use_tree else 'small_exact'))

    def verify_exhaustive(self,E1,R1,added_true=None,stop_early=True):
        return self.verify(E1,R1,added_true=added_true,exhaustive=True,stop_early=stop_early)
