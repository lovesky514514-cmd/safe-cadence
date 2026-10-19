"""Batch sparse-updates and conservative Top-k rank certification.

This shares shortlist construction with ShortlistRankCertificate, vectorizes
candidate distances across protected queries, explicitly incorporates all
changed/new entities, and exhaustively checks queries for which old-outside
lower bounds cannot rule out rank displacement. Each emitted SAFE judgment
is based solely on sufficient conditions (so no false accept under stated
floating model). Unsafe can mean either rank violation or noncertifiability.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .shortlist_certificate_v5 import ShortlistRankCertificate
from .interval_v5 import sqdistance_pairs_fast,sqdistance_pairs_outward,upper_distance
from .certificate_v2 import CheckResult

class BatchedShortlistCertificate(ShortlistRankCertificate):
    def __init__(self,*args,query_chunk=64,strict_rounding=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.query_chunk=int(query_chunk)
        self.strict_rounding=bool(strict_rounding)
        self.pair_bounds=sqdistance_pairs_outward if self.strict_rounding else sqdistance_pairs_fast
        self.p=np.asarray([tr for tr,side in self.protected],dtype=np.int64).reshape(-1,3)
        self.is_tail=np.asarray([side=='tail' for tr,side in self.protected],dtype=bool)
        self.answers=np.where(self.is_tail,self.p[:,2],self.p[:,0])
        self.q0=np.asarray([q for q,ids,low in self.rows],dtype=np.float64).reshape(-1,self.E0.shape[1])
        self.cutoffs=np.asarray([low for q,ids,low in self.rows],dtype=np.float64)
        self.shortlists=[ids for q,ids,low in self.rows]
        self.sid=np.zeros((len(self.shortlists),self.L),dtype=np.int64)
        self.svalid=np.zeros((len(self.shortlists),self.L),dtype=bool)
        for i,ids in enumerate(self.shortlists):
            self.sid[i,:len(ids)]=ids
            self.svalid[i,:len(ids)]=True
        self.truth_bans=[]
        for tr,side in self.protected:
            self.truth_bans.append(self.fixed_truth.ignored(tr,side))

    def verify(self,Enew,Rnew,added_true=None,stop_early=True,exhaustive=False):
        started=perf_counter()
        E=np.asarray(Enew,dtype=np.float64);R=np.asarray(Rnew,dtype=np.float64)
        if E.ndim!=2 or R.ndim!=2 or E.shape[1]!=self.E0.shape[1] or R.shape[1]!=self.R0.shape[1]:raise ValueError('shape mismatch')
        if len(E)<len(self.E0) or len(R)<len(self.R0) or not np.isfinite(E).all() or not np.isfinite(R).all():raise ValueError('invalid proposal')
        if added_true is not None and not np.array_equal(np.asarray(added_true,dtype=np.int64).reshape(-1,3),self.added_true):raise ValueError('truth protocol changed')
        changed=np.flatnonzero(np.any(E[:len(self.E0)]!=self.E0,axis=1))
        changed=np.concatenate((changed,np.arange(len(self.E0),len(E))))
        checked=inspected=full=fast=fall=0; failure=None
        n=len(E);d=E.shape[1]
        # Fused fast path: no per-query Python candidate distance loop. This is
        # valid only when ALL protected queries have a safe outsider bound.
        if len(self.p) and not exhaustive:
            q_all=np.where(self.is_tail[:,None],E[self.p[:,0]]+R[self.p[:,1]],
                           E[self.p[:,2]]-R[self.p[:,1]])
            _,a_hi=self.pair_bounds(q_all,E[self.answers])
            _,dr_hi=self.pair_bounds(q_all,self.q0)
            threshold=np.nextafter(upper_distance(a_hi)+upper_distance(dr_hi),np.inf)
            all_guard=np.all(np.isfinite(threshold)&(self.cutoffs>threshold))
            if all_guard:
                M=len(changed);Q=len(self.p)
                # The duplicated changed entities are counted only once.
                if M:
                    ids=np.concatenate((self.sid,np.broadcast_to(changed,(Q,M))),axis=1)
                    valid=np.concatenate((self.svalid,np.ones((Q,M),dtype=bool)),axis=1)
                    valid[:,self.L:] &= ~np.any(self.sid[:,:,None]==changed[None,None,:],axis=1)
                    for i,banned in enumerate(self.truth_bans):
                        if banned:valid[i,self.L:] &= ~np.isin(changed,list(banned))
                else:
                    ids=self.sid
                    valid=self.svalid.copy()
                valid &= ids!=self.answers[:,None]
                # Chunk to cap peak Q*(L+M)*d memory usage.
                failures=[];total_visited=0
                for left in range(0,Q,self.query_chunk):
                    right=min(Q,left+self.query_chunk)
                    lo,_=self.pair_bounds(q_all[left:right,None,:],E[ids[left:right]])
                    count=np.count_nonzero((lo<=a_hi[left:right,None])&valid[left:right],axis=1)
                    failures.extend(np.flatnonzero(count>=self.k).tolist() if left==0 else (np.flatnonzero(count>=self.k)+left).tolist())
                    total_visited=right
                    if failures and stop_early:break
                first=failures[0] if failures else None
                considered=(first+1 if first is not None and stop_early else total_visited)
                scanned=int(valid[:considered].sum())
                full=int(sum(max(n-1-len(self.truth_bans[i]),0) for i in range(considered)))
                self.last_stats={'fast_queries':considered,'fallback_queries':0,'changed_entities':M,'shortlist':self.L}
                fail=(self.protected[first][0],self.protected[first][1],self.k+1) if first is not None else None
                return CheckResult(first is None,considered,scanned,full,0.,perf_counter()-started,fail,
                                   'certified' if first is None else 'not_certifiable_or_rank_violation','batched_shortlist_fused')
        for left in range(0,len(self.p),self.query_chunk):
            right=min(len(self.p),left+self.query_chunk)
            p=self.p[left:right];tail=self.is_tail[left:right]
            q=np.where(tail[:,None],E[p[:,0]]+R[p[:,1]],E[p[:,2]]-R[p[:,1]])
            a=self.answers[left:right]
            _,ans_hi=self.pair_bounds(q,E[a]);_,dq_hi=self.pair_bounds(q,self.q0[left:right])
            threshold=np.nextafter(upper_distance(ans_hi)+upper_distance(dq_hi),np.inf)
            guards=np.isfinite(threshold)&(self.cutoffs[left:right]>threshold)&(not exhaustive)
            # Safe fast rows: shortlist + ALL moved/new IDs. Other rows fallback.
            for i in range(right-left):
                j=left+i
                if guards[i]:
                    ids=np.union1d(self.shortlists[j],changed)
                    fast+=1
                else:
                    ids=np.arange(n)
                    fall+=1
                banned=self.truth_bans[j]
                valid=ids!=a[i]
                if banned:valid &= ~np.isin(ids,list(banned))
                ids=ids[valid]
                inspected+=len(ids);full+=max(n-1-len(banned),0)
                # Most of the work happens for O(L+M), not O(N).
                rival_lo,_=self.pair_bounds(q[i:i+1],E[ids])
                risk=int(np.count_nonzero(rival_lo<=ans_hi[i]))
                checked+=1
                if risk>=self.k:
                    failure=(self.protected[j][0],self.protected[j][1],risk+1)
                    if stop_early:break
            if failure is not None and stop_early:break
        self.last_stats={'fast_queries':fast,'fallback_queries':fall,'changed_entities':len(changed),'shortlist':self.L}
        return CheckResult(failure is None,checked,inspected,full,0.,perf_counter()-started,
                           failure,'certified' if failure is None else 'not_certifiable_or_rank_violation','batched_shortlist')

    def verify_exhaustive(self,E,R,added_true=None,stop_early=True):
        return self.verify(E,R,added_true=added_true,stop_early=stop_early,exhaustive=True)
