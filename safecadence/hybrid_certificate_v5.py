"""Dynamic safe gate: vectorized exhaustive for small/dense updates,
certified bounded-shortlist for large sparse updates. Build shortlist lazily.
No semantic prediction or unlabeled-query guarantees.
"""
import numpy as np
from .batched_certificate_v4 import BatchedRankCertificate
from .batched_shortlist_v5 import BatchedShortlistCertificate

class HybridRankCertificate:
    def __init__(self,E0,R0,protected_triples,known_true,k=10,added_true=None,
                 threshold_entities=1500,max_changed_fraction=.02,shortlist_size=32):
        self.E0=np.asarray(E0,dtype=np.float64).copy()
        self.R0=np.asarray(R0,dtype=np.float64).copy()
        self._original_protected=np.asarray(protected_triples,dtype=np.int64).copy()
        self._known_true=np.asarray(known_true,dtype=np.int64).copy()
        self._added_true=np.asarray(added_true,dtype=np.int64).reshape(-1,3) if added_true is not None else np.empty((0,3),dtype=np.int64)
        self._params=dict(k=k,added_true=self._added_true)
        self.threshold_entities=threshold_entities
        self.max_changed_fraction=max_changed_fraction
        self.L=shortlist_size
        self.exhaustive=BatchedRankCertificate(self.E0,self.R0,self._original_protected,self._known_true,**self._params)
        self.shortlist=None
        self.protected=self.exhaustive.protected
        self.not_initially_protected=self.exhaustive.not_initially_protected
        self.last_engine='batched'
        self.shortlist_construction_count=0

    def verify(self,E,R,added_true=None,stop_early=True,exhaustive=False):
        E=np.asarray(E,dtype=np.float64)
        if exhaustive or len(E)<self.threshold_entities:
            self.last_engine='batched'
            return self.exhaustive.verify(E,R,added_true=added_true,stop_early=stop_early)
        changed=int(np.count_nonzero(np.any(E[:len(self.E0)]!=self.E0,axis=1)))+len(E)-len(self.E0)
        if changed>self.max_changed_fraction*len(E):
            self.last_engine='batched'
            return self.exhaustive.verify(E,R,added_true=added_true,stop_early=stop_early)
        if self.shortlist is None:
            self.shortlist=BatchedShortlistCertificate(self.E0,self.R0,self._original_protected,
                      self._known_true,shortlist_size=self.L,**self._params)
            self.shortlist_construction_count+=1
        self.last_engine='shortlist'
        return self.shortlist.verify(E,R,added_true=added_true,stop_early=stop_early)

    def verify_exhaustive(self,E,R,added_true=None,stop_early=True):
        # Original v0.4 brute-force as an independent numerical comparator.
        return self.exhaustive.verify(E,R,added_true=added_true,stop_early=stop_early)
