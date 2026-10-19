"""Conditional rank-preservation verifier for *append-only* TransE updates.

The original entity vectors and all relation vectors MUST stay bitwise fixed.
New entities are appended to E0; existing protected query embeddings do not move.
For each initially eligible query, old rivals' conservative rank is cached;
verify only the appended candidates. Correctness is conditional on IEEE-754
binary64 arithmetic as described in the package docs and the stored query
vectors' binary64 values; not a compiler-independent proof.

A failed certificate means NOT CERTIFIED, not necessarily a genuine rank loss.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from safecadence.interval_v5 import sqdistance_bounds


@dataclass(frozen=True)
class AppendDecision:
    certified: bool
    status: str
    protected_queries: int
    candidate_distances_checked: int
    possible_violations: int
    verify_seconds: float
    # Explicit scope diagnostics: a passed certificate does not cover queries
    # that were already outside Top-k before this append.
    excluded_initial: int = 0
    requested_triples: int = 0


class FrozenAppendGuard:
    def __init__(self, old_entities, frozen_relations, old_triples,
                 known_true, *, k=10, max_protected=None, chunk_size=1024):
        """old_triples: user-chosen OLD train facts; no val/test may enter here.

        known_true may include stage-appropriate facts for FILTERING ONLY;
        do not expose future snapshots' facts. Data and filtering protocol
        must remain static throughout a verifier's lifetime.
        """
        started=perf_counter()
        self.E=np.array(old_entities,dtype=np.float64,copy=True)
        self.R=np.array(frozen_relations,dtype=np.float64,copy=True)
        self.E.setflags(write=False);self.R.setflags(write=False)
        if self.E.ndim!=2 or self.R.ndim!=2 or self.R.shape[1]!=self.E.shape[1]:
            raise ValueError('invalid embedding matrices')
        if not np.isfinite(self.E).all() or not np.isfinite(self.R).all():
            raise ValueError('nonfinite frozen parameters')
        if not isinstance(k, int) or k < 1:
            raise ValueError('k must be a positive integer')
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        if max_protected is not None and (not isinstance(max_protected, int) or max_protected < 1):
            raise ValueError('max_protected must be positive if supplied')
        self.k = k
        self.chunk = chunk_size
        self.n0 = len(self.E)
        prot=np.asarray(old_triples,dtype=np.int64).reshape(-1,3)
        facts=np.asarray(known_true,dtype=np.int64).reshape(-1,3)
        self._check_ids(prot, allow_append=False)
        if len(facts) and (np.min(facts[:,1])<0 or np.max(facts[:,1])>=len(self.R) or np.min(facts[:,[0,2]])<0):
            raise ValueError('invalid known true triple')
        if max_protected is not None:
            prot = prot[:max_protected]
        self.requested_triples = len(prot)
        # Time and space constant in number of historical queries, not entities.
        from collections import defaultdict
        heads=defaultdict(set);tails=defaultdict(set)
        for h,r,t in facts:
            heads[int(r),int(t)].add(int(h))
            tails[int(h),int(r)].add(int(t))
        self.query=[]
        self.cutoffs=[]
        self.old_possible=[]
        self.new_bans=[]
        self.excluded_initial=0
        for h,r,t in prot:
            for side in ('tail','head'):
                h=int(h);r=int(r);t=int(t)
                ans=t if side=='tail' else h
                q=(self.E[h]+self.R[r]) if side=='tail' else (self.E[t]-self.R[r])
                _,a_hi=sqdistance_bounds(q,self.E[ans:ans+1])
                threshold=float(a_hi[0]);old_count=0
                truth=tails[(h,r)] if side=='tail' else heads[(r,t)]
                # Filter known correct answers other than target.
                ban_old={x for x in truth if x<self.n0 and x!=ans}
                for start in range(0,self.n0,self.chunk):
                    end=min(start+self.chunk,self.n0)
                    lo,_=sqdistance_bounds(q,self.E[start:end]);mask=lo<=threshold
                    for idx in ban_old:
                        if start<=idx<end:mask[idx-start]=False
                    if start<=ans<end:mask[ans-start]=False
                    old_count+=int(np.count_nonzero(mask))
                if old_count>=self.k:
                    self.excluded_initial+=1;continue
                self.query.append(q.copy())
                self.cutoffs.append(threshold)
                self.old_possible.append(old_count)
                self.new_bans.append(set(x for x in truth if x>=self.n0))
        self.query=np.asarray(self.query,dtype=np.float64).reshape(-1,self.E.shape[1])
        self.cutoffs=np.asarray(self.cutoffs,dtype=np.float64)
        self.old_possible=np.asarray(self.old_possible,dtype=np.int64)
        self.precompute_seconds=perf_counter()-started

    def _check_ids(self,triples,*,allow_append):
        if not len(triples):return
        if np.min(triples)<0 or np.max(triples[:,1])>=len(self.R):
            raise ValueError('out-of-range relation or negative id')
        if not allow_append and np.max(triples[:,[0,2]])>=self.n0:
            raise ValueError('protected triple references future entity')

    def verify_appended(self,new_vectors)->AppendDecision:
        started=perf_counter()
        x=np.asarray(new_vectors,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.E.shape[1] or not np.isfinite(x).all():
            raise ValueError('nonfinite or invalid new entity vectors')
        if not len(self.query):
            # An empty effective protection scope cannot establish a safety claim.
            # The caller must explicitly choose eligible old queries first.
            return AppendDecision(False, 'NOT_CERTIFIED_EMPTY_SCOPE', 0, 0, 0,
                                  perf_counter() - started,
                                  self.excluded_initial, self.requested_triples)
        checked=0;failed=0
        for i,q in enumerate(self.query):
            count=int(self.old_possible[i]);banned=self.new_bans[i]
            for start in range(0,len(x),self.chunk):
                end=min(start+self.chunk,len(x))
                low,_=sqdistance_bounds(q,x[start:end]);mask=low<=self.cutoffs[i]
                for new_id in banned:
                    idx=new_id-self.n0
                    if start<=idx<end:mask[idx-start]=False
                checked+=(end-start)
                count+=int(np.count_nonzero(mask))
                if count>=self.k:
                    failed+=1;break
        return AppendDecision(
            certified=failed == 0,
            status='CERTIFIED' if failed == 0 else 'NOT_CERTIFIED',
            protected_queries=len(self.query),
            candidate_distances_checked=checked,
            possible_violations=failed,
            verify_seconds=perf_counter() - started,
            excluded_initial=self.excluded_initial,
            requested_triples=self.requested_triples,
        )
