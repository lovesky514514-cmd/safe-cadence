"""Proof-carrying sparse embedding patches and a candidate-sublinear rank gate.

Strict support is guaranteed by construction: the proposal contains ONLY
indexed entity overrides (and optional appended rows) and a relation matrix.
There is no caller-supplied 'these are all changed IDs' trust assumption.

For safe guard passes, update verification needs O(Q*(L+M)*d) candidate work
and O(Q*(L+M)*log M) indexed override lookup, independent of N apart from
reference storage and one-time O(Q*N*d) indexing. When margin guards fail,
it materializes a full proposal and falls back to exhaustive verification;
thus NO unconditional sublinear-complexity claim.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .batched_shortlist_v5 import BatchedShortlistCertificate
from .interval_v5 import sqdistance_pairs_fast, sqdistance_pairs_outward, upper_distance
from .certificate_v2 import CheckResult

@dataclass(frozen=True)
class SparsePatch:
    ids: np.ndarray                 # Existing row IDs, unique, sorted
    vectors: np.ndarray             # Corresponding new values, [m,d]
    relations: np.ndarray           # Entire new relation matrix
    appended: np.ndarray | None = None # New appended entities, [a,d]

class SparseOverlayCertificate(BatchedShortlistCertificate):
    def verify_patch(self,patch:SparsePatch,added_true=None,stop_early=True):
        started=perf_counter()
        ids=np.asarray(patch.ids,dtype=np.int64)
        vectors=np.asarray(patch.vectors,dtype=np.float64)
        R=np.asarray(patch.relations,dtype=np.float64)
        extra=np.asarray(patch.appended,dtype=np.float64) if patch.appended is not None else np.empty((0,self.E0.shape[1]),dtype=np.float64)
        n0,d=self.E0.shape
        if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or (len(ids)>0 and (ids.min()<0 or ids.max()>=n0)):
            raise ValueError('ids must be unique existing entity IDs')
        if vectors.shape!=(len(ids),d) or extra.ndim!=2 or extra.shape[1]!=d or R.ndim!=2 or R.shape[1]!=d or len(R)<len(self.R0):
            raise ValueError('shape mismatch')
        if not np.isfinite(vectors).all() or not np.isfinite(extra).all() or not np.isfinite(R).all():raise ValueError('nonfinite patch')
        if added_true is not None and not np.array_equal(np.asarray(added_true,dtype=np.int64).reshape(-1,3),self.added_true):
            raise ValueError('truth protocol changed; rebuild')
        order=np.argsort(ids); ids=ids[order]; vectors=vectors[order]
        # No false 'unchanged' metadata possible: the object IS the update.
        def gather(query_ids):
            z=np.asarray(query_ids,dtype=np.int64)
            flat=z.ravel()
            if len(flat) and (flat.min()<0 or flat.max()>=n0+len(extra)):raise IndexError('invalid entity ID')
            out=np.empty((len(flat),d),dtype=np.float64)
            old=flat<n0
            if np.any(old):
                old_ids=flat[old]
                base=self.E0[old_ids].copy()
                if len(ids):
                    pos=np.searchsorted(ids,old_ids)
                    safe_pos=np.clip(pos,0,len(ids)-1)
                    replace=(pos<len(ids))&(ids[safe_pos]==old_ids)
                    base[replace]=vectors[safe_pos[replace]]
                out[old]=base
            if np.any(~old):out[~old]=extra[flat[~old]-n0]
            return out.reshape((*z.shape,d))
        n=n0+len(extra)
        q=np.where(self.is_tail[:,None],gather(self.p[:,0])+R[self.p[:,1]],
                   gather(self.p[:,2])-R[self.p[:,1]])
        answers=gather(self.answers)
        bounds=self.pair_bounds
        _,anshi=bounds(q,answers)
        _,drifthi=bounds(q,self.q0)
        thresh=np.nextafter(upper_distance(anshi)+upper_distance(drifthi),np.inf)
        allowed=np.isfinite(thresh)&(self.cutoffs>thresh)
        if not bool(np.all(allowed)):
            # This is an explicit fallback to exhaustive candidate evaluation.
            all_entities=np.concatenate((self.E0.copy(),extra),axis=0)
            if len(ids):all_entities[ids]=vectors
            value=super().verify(all_entities,R,added_true=added_true,stop_early=stop_early,exhaustive=True)
            self.last_stats={'method':'sparse_overlay_fallback','fast_queries':0,
                             'fallback_queries':len(self.protected),'changed_entities':len(ids)+len(extra)}
            return CheckResult(value.safe,value.checked_queries,value.inspected_candidates,
                               value.full_candidates,0.,perf_counter()-started,value.failed_query,
                               value.reason,'sparse_overlay_fallback')
        changed=np.concatenate((ids,np.arange(n0,n,dtype=np.int64)))
        Q=len(self.protected);M=len(changed)
        if M:
            candidate_ids=np.concatenate((self.sid,np.broadcast_to(changed,(Q,M))),axis=1)
            valid=np.concatenate((self.svalid,np.ones((Q,M),dtype=bool)),axis=1)
            valid[:,self.L:] &= ~np.any(self.sid[:,:,None]==changed[None,None,:],axis=1)
            for i,banned in enumerate(self.truth_bans):
                if banned:valid[i,self.L:] &= ~np.isin(changed,list(banned))
        else:
            candidate_ids=self.sid
            valid=self.svalid.copy()
        valid &= candidate_ids!=self.answers[:,None]
        examined=0;failure=None;limit=self.query_chunk
        for begin in range(0,Q,limit):
            end=min(Q,begin+limit)
            rivals=gather(candidate_ids[begin:end])
            low,_=bounds(q[begin:end,None,:],rivals)
            count=np.count_nonzero((low<=anshi[begin:end,None])&valid[begin:end],axis=1)
            violated=np.flatnonzero(count>=self.k)
            if len(violated):
                failure=(self.protected[begin+int(violated[0])][0],self.protected[begin+int(violated[0])][1],self.k+1)
                examined=begin+int(violated[0])+1 if stop_early else end
                if stop_early:break
            examined=end
        seen=int(valid[:examined].sum())
        total=int(sum(max(n-1-len(self.truth_bans[i]),0) for i in range(examined)))
        self.last_stats={'method':'sparse_overlay','fast_queries':examined,'fallback_queries':0,
                         'changed_entities':M,'candidates_scored':seen}
        return CheckResult(failure is None,examined,seen,total,0.,perf_counter()-started,
                           failure,'certified' if failure is None else 'not_certifiable_or_rank_violation','sparse_overlay')
