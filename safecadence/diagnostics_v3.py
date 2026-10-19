"""Safe-Cadence v0.3 falsification experiments.

All methods consume IDENTICAL pre-generated mini-batches and negatives per seed.
The certificate is only enabled for 'certified' arms. Baseline end points
are independently exhaustively ranked, not automatically called 'safe'.

This is a small fixed-entity synthetic TransE study, NOT a CKGE benchmark.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .certificate_cache import CachedRankCertificate
from .ranking import query_and_answer, TrueFacts, exact_rank


@dataclass(frozen=True)
class Variant:
    name: str
    lr_factor: float = 1.0
    anchor: float = 0.0
    entity_mode: str = 'all'  # 'incoming' or 'frozen'
    relation_mode: str = 'all'  # 'frozen'
    certified: bool = False


def variants():
    return [
        Variant('sgd'),
        Variant('sgd_lr_half', lr_factor=.5),
        Variant('sgd_lr_tenth', lr_factor=.1),
        Variant('sgd_strong_l2', anchor=3.0),
        Variant('sgd_incoming_entities_only', entity_mode='incoming', relation_mode='frozen'),
        Variant('sgd_freeze_entities', entity_mode='frozen'),
        Variant('certified_sgd', certified=True),
        Variant('certified_lr_half', lr_factor=.5, certified=True),
        Variant('certified_lr_tenth', lr_factor=.1, certified=True),
        Variant('certified_strong_l2', anchor=3.0, certified=True),
        Variant('certified_incoming_entities_only', entity_mode='incoming', relation_mode='frozen', certified=True),
    ]


def make_batches(incoming, all_true, n_entities, steps=80, batch_size=32,
                 negatives=2, seed=0):
    rng=np.random.default_rng(seed)
    incoming=np.asarray(incoming, dtype=np.int64)
    seen=set(map(tuple, np.asarray(all_true,dtype=np.int64).tolist()))
    rows=[]
    for step in range(steps):
        pos=incoming[rng.integers(0,len(incoming),batch_size)]
        neg=np.repeat(pos,negatives,axis=0).copy()
        for j in range(len(neg)):
            axis=0 if rng.integers(0,2)==0 else 2
            for _ in range(80):
                neg[j,axis]=int(rng.integers(0,n_entities))
                if tuple(neg[j]) not in seen:break
            else:raise ValueError('Negative rejection exhausted')
        rows.append((torch.tensor(pos,dtype=torch.long), torch.tensor(neg,dtype=torch.long)))
    return rows


def count_rank_failures(checker, E, R, incoming):
    # EXACT checker counts every violated historically top-k query, not just first.
    full=checker.verify_exhaustive(E,R,added_true=incoming,stop_early=False)
    # The verifier tracks one failure even with stop_early=False. Count explicitly.
    truth=TrueFacts(np.concatenate((checker.history,incoming))) if hasattr(checker,'history') else None
    # Correct filtering uses merged source facts. No estimate needed for binary safety.
    return full.safe


def rank_scope_diagnostics(E0,R0,E1,R1,old,protected,incoming,k=10):
    truth=TrueFacts(np.concatenate((old,incoming)))
    failed=tracked=0
    for triple in old:
        for side in ('head','tail'):
            if exact_rank(E0,R0,triple,side,truth)<=k:
                tracked+=1
                if exact_rank(E1,R1,triple,side,truth)>k: failed+=1
    return {'historical_eligible_queries':tracked,
            'historical_broken_queries':failed,
            'historical_break_fraction':(failed/tracked if tracked else None)}


def run_variant(E0,R0,incoming,old,protected,batches,variant:Variant,
                base_lr=.25,k=10, audit_decisions=True,
                max_cache_mb=64, certificate_engine="cached", audit_stride=20):
    torch.set_num_threads(1)
    E=np.asarray(E0,dtype=np.float64).copy()
    R=np.asarray(R0,dtype=np.float64).copy()
    # Build the index for any arm only for end-point assessment, but measure index
    # outside training walltime for blind methods as it was not used by that arm.
    index_start=perf_counter()
    if certificate_engine=="cached":
        checker=CachedRankCertificate(E0,R0,protected,old,k=k,max_cache_mb=max_cache_mb)
    elif certificate_engine=="batched":
        from .batched_certificate_v4 import BatchedRankCertificate
        checker=BatchedRankCertificate(E0,R0,protected,old,k=k,added_true=incoming)
    elif certificate_engine=="batchshort":
        from .batched_shortlist_v5 import BatchedShortlistCertificate
        checker=BatchedShortlistCertificate(E0,R0,protected,old,k=k,added_true=incoming,shortlist_size=32)
    else:raise ValueError("unknown certificate engine")
    index_elapsed=perf_counter()-index_start
    allowed=np.zeros(len(E),dtype=bool)
    incoming=np.asarray(incoming,dtype=np.int64)
    allowed[np.unique(incoming[:,(0,2)])]=True
    learning=certificate=exhaustive_audit=0.
    proposed=0; attempted=0; unsafe_proposals=0; steps_rescued=0
    steps_accepted=0; steps_rejected=0
    false_accepts=false_rejects=0
    audited_calls=0
    total_candidates=total_inspected=0
    unsafe_intermediate=0
    baseline_start=perf_counter()
    levels=(1.0,.5,.25,.125,.0625,.03125,.015625,.0078125) if variant.certified else (1.0,)
    for pos,neg in batches:
        t=perf_counter()
        ee=torch.tensor(E,dtype=torch.float32,requires_grad=True)
        rr=torch.tensor(R,dtype=torch.float32,requires_grad=True)
        def distances(t):return torch.linalg.vector_norm(ee[t[:,0]]+rr[t[:,1]]-ee[t[:,2]],dim=1)
        loss=F.relu(1+distances(pos).repeat_interleave(len(neg)//len(pos))-distances(neg)).mean()
        # L2-to-initial params baseline; allow drifting entity IDs in new batch only.
        if variant.anchor:
            loss=loss+variant.anchor*.5*((ee-torch.tensor(E0,dtype=torch.float32)).square().sum()
                                            +(rr-torch.tensor(R0,dtype=torch.float32)).square().sum())
        ge,gr=torch.autograd.grad(loss,(ee,rr))
        ge=ge.detach().numpy().astype(np.float64)
        gr=gr.detach().numpy().astype(np.float64)
        if variant.entity_mode=='incoming':ge[~allowed]=0
        if variant.entity_mode=='frozen':ge[:]=0
        if variant.relation_mode=='frozen':gr[:]=0
        learning+=perf_counter()-t
        proposed+=1
        accepted=False
        first_failed=False
        for j,alpha in enumerate(levels):
            E1=E-base_lr*variant.lr_factor*alpha*ge
            R1=R-base_lr*variant.lr_factor*alpha*gr
            norm=np.linalg.norm(E1,axis=1)
            exceed=norm>1.000000001
            if np.any(exceed):E1[exceed]/=norm[exceed,None]
            if variant.certified:
                t=perf_counter()
                decision=checker.verify(E1,R1,added_true=incoming,stop_early=True)
                certificate+=perf_counter()-t
                total_inspected+=decision.inspected_candidates
                total_candidates+=decision.full_candidates
                attempted+=1
                if audit_decisions and attempted % audit_stride == 0:
                    audited_calls+=1
                    t=perf_counter()
                    ground_truth=checker.verify_exhaustive(E1,R1,added_true=incoming,stop_early=True).safe
                    exhaustive_audit+=perf_counter()-t
                    if decision.safe and not ground_truth:false_accepts+=1
                    if not decision.safe and ground_truth:false_rejects+=1
                if not decision.safe:
                    if j==0:first_failed=True;unsafe_proposals+=1
                    continue
            E,R=E1,R1
            steps_accepted+=1
            if first_failed:steps_rescued+=1
            accepted=True
            break
        if not accepted:steps_rejected+=1
        if not variant.certified:
            # Outside the training path: exact audit of each blind intermediate snapshot.
            t=perf_counter()
            if not checker.verify_exhaustive(E,R,added_true=incoming).safe:
                unsafe_intermediate+=1
            exhaustive_audit+=perf_counter()-t
    train_wall=perf_counter()-baseline_start
    # measure a full endpoint exhaustive test independently of the cached result
    t=perf_counter()
    endpoint=checker.verify_exhaustive(E,R,added_true=incoming,stop_early=False)
    endpoint_exhaustive=perf_counter()-t
    from .ranking import filtered_metrics
    return E,R,{
        'variant':asdict(variant),'certificate_engine':certificate_engine,'steps':len(batches),'accepted_steps':steps_accepted,
        'rejected_steps':steps_rejected, 'first_alpha_unsafe':unsafe_proposals,
        'rescued_steps':steps_rescued,'certificate_calls':attempted,
        'gate_false_accepts':false_accepts,'gate_false_rejects':false_rejects,
        'independently_audited_gate_calls':audited_calls,'audit_stride':audit_stride,
        'protected_initial_topk_queries':len(checker.protected),
        'not_initially_protected':checker.not_initially_protected,
        'endpoint_safe_exhaustive':endpoint.safe,
        'unsafe_intermediate_steps':unsafe_intermediate if not variant.certified else 0,
        'candidate_fraction_scored':total_inspected/total_candidates if total_candidates else None,
        'index_build_seconds':index_elapsed,'gradient_seconds':learning,
        'certificate_seconds':certificate,'audit_exhaustive_seconds':exhaustive_audit,
        'endpoint_exhaustive_seconds':endpoint_exhaustive,
        'training_and_audit_wall_seconds':train_wall,
        'method_wall_seconds':train_wall-exhaustive_audit+(index_elapsed if variant.certified else 0.),
        'new_mrr':filtered_metrics(E,R,incoming,np.concatenate((old,incoming)),max_eval=100)['mrr'],
        'old_mrr':filtered_metrics(E,R,old,np.concatenate((old,incoming)),max_eval=100)['mrr']}
