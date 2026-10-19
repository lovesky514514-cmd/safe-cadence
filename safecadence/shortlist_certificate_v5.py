"""Conservative, bounded-memory shortlist certificate for TransE Top-k.

Per historical query, construct a shortlist of L possible rivals and store a
rigorous LOWER distance bound for all excluded original entities. Updates
certify shortlist + moved/appended entities if the excluded-unmoved tail is
ruled out by a triangle-inequality guard, and otherwise fall back to full
interval enumeration (NO unsupported safety claims).

One-time: O(Q N d) work, O(Q L + Q d) storage (+ reference embeddings).
Update: O(Q (L+M) d) in guard-success cases (M = # moved or added entities),
        O(Q N d) in guard-failure cases; NO unconditional sublinear guarantee.

Soundness with respect to exact squared Euclidean distances on STORED binary64
query and entity coordinate values follows from outward interval arithmetic.
Query arithmetic float64 h+r / t-r is part of the model's *defined* semantics.
The conservative implementation rejects when interval comparisons cannot be
proved safe, so it may issue false rejects but cannot falsely accept within
its stated floating-point assumptions.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .certificate_v2 import SparseRadiusCertificate,CheckResult
from .interval_v5 import sqdistance_bounds_fast as sqdistance_bounds,upper_distance,lower_distance
from .ranking import TrueFacts,query_and_answer

class ShortlistRankCertificate(SparseRadiusCertificate):
    def __init__(self,E0,R0,protected_triples,known_true,k=10,shortlist_size=32,
                 added_true=None,**kwargs):
        super().__init__(E0,R0,protected_triples,known_true,k=k,**kwargs)
        if shortlist_size<k:raise ValueError('shortlist_size must be >= k')
        if not self.protected:raise ValueError('Zero initially Top-k eligible queries: no meaningful safety claim')
        self.L=int(shortlist_size)
        self.added_true=np.asarray(added_true,dtype=np.int64).reshape(-1,3) if added_true is not None else np.empty((0,3),dtype=np.int64)
        self.fixed_truth=TrueFacts(np.concatenate((np.asarray(known_true,dtype=np.int64).reshape(-1,3),self.added_true)))
        started=perf_counter()
        self.rows=[]
        ids_all=np.arange(len(self.E0),dtype=np.int64)
        for triple,side in self.protected:
            q,ans=query_and_answer(self.E0,self.R0,triple,side)
            # excluded old true facts cannot outrank: remove from candidates.
            excluded=self.fixed_truth.ignored(triple,side)
            allowed=np.ones(len(self.E0),dtype=bool)
            allowed[ans]=False
            if excluded:allowed[list(excluded)]=False
            ids=ids_all[allowed]
            if not len(ids):
                self.rows.append((q.copy(),np.empty(0,dtype=np.int64),np.inf))
                continue
            # Sort is only a heuristic for choosing the shortlist, NOT part of proof.
            dist2=np.square(self.E0[ids]-q).sum(axis=1)
            n_take=min(self.L,len(ids))
            if n_take==len(ids):
                chosen=ids.copy();lower_cutoff=np.inf
            else:
                ix=np.argpartition(dist2,n_take-1)[:n_take]
                selected=np.zeros(len(ids),dtype=bool);selected[ix]=True
                chosen=ids[ix]
                remainder=ids[~selected]
                # Rigorous lower bound on EVERY excluded entity, not on a heuristic kth item.
                lo,_=sqdistance_bounds(q,self.E0[remainder])
                lower_cutoff=float(np.min(lower_distance(lo)))
            self.rows.append((q.copy(),np.sort(chosen),lower_cutoff))
        self.precompute_seconds=perf_counter()-started
        self.last_stats={}

    def verify(self,new_entities,new_relations,added_true=None,stop_early=True,exhaustive=False):
        start=perf_counter()
        E=np.asarray(new_entities,dtype=np.float64)
        R=np.asarray(new_relations,dtype=np.float64)
        if E.ndim!=2 or R.ndim!=2 or E.shape[1]!=self.E0.shape[1] or R.shape[1]!=self.R0.shape[1]:raise ValueError('dimension mismatch')
        if len(E)<len(self.E0) or len(R)<len(self.R0):raise ValueError('removal not supported')
        if not np.isfinite(E).all() or not np.isfinite(R).all():raise ValueError('nonfinite proposal')
        if added_true is not None and not np.array_equal(np.asarray(added_true,dtype=np.int64).reshape(-1,3),self.added_true):
            raise ValueError('truth protocol changed; rebuild certificate')
        changed=np.flatnonzero(np.any(E[:len(self.E0)]!=self.E0,axis=1))
        moved=np.concatenate((changed,np.arange(len(self.E0),len(E),dtype=np.int64)))
        qchecked=inspected=total=0
        fast=fall=uncertain=0;failure=None; reason='certified'
        for (tr,side),(q0,chosen,tail_lower) in zip(self.protected,self.rows):
            q1,ans=query_and_answer(E,R,tr,side)
            _,ans_hi_sq=sqdistance_bounds(q1,E[ans:ans+1])
            answer_upper=float(upper_distance(ans_hi_sq[0]))
            _,drift_hi_sq=sqdistance_bounds(q1,q0[None,:])
            drift_upper=float(upper_distance(drift_hi_sq[0]))
            # triangle inequality, all uncertainties rounded outward.
            threshold=np.nextafter(answer_upper+drift_upper,np.inf)
            guard=(not exhaustive and np.isfinite(threshold) and tail_lower>threshold)
            if guard:
                cand=np.union1d(chosen,moved)
                fast+=1
            else:
                cand=np.arange(len(E),dtype=np.int64)
                fall+=1
            mask=(cand!=ans)
            excluded=self.fixed_truth.ignored(tr,side)
            if excluded:mask &= ~np.isin(cand,list(excluded))
            cand=cand[mask]
            rival_lo,_=sqdistance_bounds(q1,E[cand])
            # A candidate definitely cannot outrank the answer only if its
            # LOWER squared distance exceeds the answer's UPPER squared distance.
            may_outrank=int(np.count_nonzero(rival_lo<=ans_hi_sq[0]))
            if may_outrank>self.k-1:
                failure=(tr,side,may_outrank+1)
                reason='not_certifiable_or_rank_violation'
                uncertain+=1
                if stop_early:
                    qchecked+=1;inspected+=len(cand);total+=max(0,len(E)-1-len(excluded))
                    break
            qchecked+=1;inspected+=len(cand);total+=max(0,len(E)-1-len(excluded))
        self.last_stats={'fast_queries':fast,'fallback_queries':fall,'uncertain_or_unsafe_queries':uncertain,
                         'changed_entities':len(moved),'shortlist':self.L}
        return CheckResult(failure is None,qchecked,inspected,total,0.,perf_counter()-start,
                           failure,reason,'interval_shortlist')

    def verify_exhaustive(self,E,R,added_true=None,stop_early=True):
        return self.verify(E,R,added_true=added_true,stop_early=stop_early,exhaustive=True)
