"""Query-centric exact Top-k certification with a cached, sorted distance index.
For stable candidate c, d(q',c)>= d(q,c)-delta_q. Query cached old
sorted distances up to d(q',a')+delta_q; evaluate this shortlist exactly.
All moved and appended entities are also evaluated exactly.

Time: build O(Q*N*(d+log N)) and memory O(Q*N), verify O(Q*(log N +
shortlist*d + moved*d)). Fast on repeated sparse updates, not guaranteed
fast on dense ones; no false accept due to excluded unscored entities.
"""
from time import perf_counter
import numpy as np
from .certificate_v2 import SparseRadiusCertificate, CheckResult
from .ranking import query_and_answer

class CachedRankCertificate(SparseRadiusCertificate):
    def __init__(self,*args,max_cache_mb=256,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached=[]; self.cache_bytes=0
        q=len(self.protected); n=len(self.E0)
        needed=16*q*n
        if needed>max_cache_mb*1024*1024:
            self.cache_enabled=False
            return
        self.cache_enabled=True
        start=perf_counter()
        for triple,side in self.protected:
            q0,_=query_and_answer(self.E0,self.R0,triple,side)
            d=np.linalg.norm(self.E0-q0,axis=1)
            ordered=np.argsort(d,kind='stable')
            self.cached.append((d[ordered],ordered))
        self.indexing_seconds+=perf_counter()-start
        self.cache_bytes=needed

    def verify(self,new_entities,new_relations,added_true=None,
               exhaustive=False,stop_early=True):
        if exhaustive or not self.cache_enabled:
            return super().verify(new_entities,new_relations,added_true,exhaustive,stop_early)
        started=perf_counter()
        E1=np.asarray(new_entities,dtype=np.float64)
        R1=np.asarray(new_relations,dtype=np.float64)
        n0=len(self.E0)
        if E1.ndim!=2 or R1.ndim!=2 or E1.shape[1]!=self.E0.shape[1] or R1.shape[1]!=self.R0.shape[1]:
            raise ValueError('Dimension mismatch')
        if len(E1)<n0 or len(R1)<len(self.R0): raise ValueError('Removal unsupported')
        if not np.isfinite(E1).all() or not np.isfinite(R1).all(): raise ValueError('Nonfinite embeddings')
        if added_true is None: truth=self.truth
        else:
            from .ranking import TrueFacts
            truth=TrueFacts(added_true)
            for key,vals in self.truth.heads.items(): truth.heads[key].update(vals)
            for key,vals in self.truth.tails.items(): truth.tails[key].update(vals)
        moved=np.zeros(n0,dtype=bool)
        changed=np.flatnonzero(np.linalg.norm(E1[:n0]-self.E0,axis=1)!=0)
        moved[changed]=True
        appended=np.arange(n0,len(E1),dtype=np.int64)
        checked=inspected=full=0;failure=None
        for (triple,side),(ds,ordered) in zip(self.protected,self.cached):
            q0,answer=query_and_answer(self.E0,self.R0,triple,side)
            q1,_=query_and_answer(E1,R1,triple,side)
            da=float(np.linalg.norm(q1-E1[answer]))
            dq=float(np.linalg.norm(q1-q0))
            slack=1e-9*(1+da+dq+float(np.linalg.norm(q0)))
            count=int(np.searchsorted(ds,da+dq+slack,side='right'))
            near=ordered[:count]
            stable=near[~moved[near]]
            ids=np.concatenate((stable,changed,appended))
            mask=ids!=answer
            excl=truth.ignored(triple,side)
            if excl: mask &= ~np.isin(ids,list(excl))
            ids=ids[mask]
            ranks=1+int(np.count_nonzero(np.linalg.norm(E1[ids]-q1,axis=1)<=da))
            checked+=1;inspected+=len(ids);full+=len(E1)-1-len(excl)
            if ranks>self.k:
                failure=(triple,side,ranks)
                if stop_early: break
        return CheckResult(failure is None,checked,inspected,full,self.indexing_seconds,
                           perf_counter()-started,failure,
                           'rank_violation' if failure else 'certified','sorted_cache')
