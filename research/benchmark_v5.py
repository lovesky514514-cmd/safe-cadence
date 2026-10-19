"""Scalability/falsification probes, synthetic embedding geometry only.
Timing includes precompute and measured per-proposal verification, independently.
No claim of real CKGE scale or of walltime speedup after index amortization.
"""
import argparse,json
from time import perf_counter
from pathlib import Path
import numpy as np
from safecadence.batched_shortlist_v5 import BatchedShortlistCertificate
from safecadence.batched_certificate_v4 import BatchedRankCertificate


def run(N=1000,Q=40,d=24,scenario='sparse',seed=3,L=32,repeats=12):
    rng=np.random.default_rng(seed)
    E=rng.normal(0,.2,(N,d)).astype(np.float64)
    R=rng.normal(0,.005,(2,d)).astype(np.float64)
    hs=rng.choice(N,size=min(Q,N),replace=False)
    triples=np.stack([hs,np.zeros(len(hs),dtype=np.int64),hs],axis=1)
    v5=BatchedShortlistCertificate(E,R,triples,triples,k=10,shortlist_size=L)
    v4=BatchedRankCertificate(E,R,triples,triples,k=10,added_true=np.empty((0,3),int))
    query_time_v5=query_time_v4=0.
    fast=fallback=checked=full=0
    disagreements=0;unsafe=0
    for i in range(repeats):
        E1=E.copy();R1=R.copy()
        if scenario=='sparse':
            idx=rng.choice(N,max(1,N//1000),replace=False)
            E1[idx]+=rng.normal(0,.01,(len(idx),d))
        elif scenario=='dense':
            E1+=rng.normal(0,.005,(N,d))
        elif scenario=='relation':
            R1+=rng.normal(0,.002,(2,d))
        elif scenario=='mixed':
            idx=rng.choice(N,max(1,N//100),replace=False)
            E1[idx]+=rng.normal(0,.008,(len(idx),d))
            R1+=rng.normal(0,.0005,(2,d))
        else:raise ValueError(scenario)
        t=perf_counter();a=v5.verify(E1,R1);query_time_v5+=perf_counter()-t
        stats=v5.last_stats
        fast+=stats['fast_queries'];fallback+=stats['fallback_queries']
        checked+=a.inspected_candidates;full+=a.full_candidates
        t=perf_counter();b=v4.verify(E1,R1);query_time_v4+=perf_counter()-t
        if a.safe!=b.safe:disagreements+=1
        unsafe+=not b.safe
    return {'N':N,'Q':len(v5.protected),'d':d,'L':L,'scenario':scenario,
        'repeats':repeats,'v5_build_seconds':v5.precompute_seconds,'v5_gate_seconds':query_time_v5,
        'v4_gate_seconds':query_time_v4,'v5_gate_ms':1000*query_time_v5/repeats,
        'v4_gate_ms':1000*query_time_v4/repeats,
        'per_gate_speedup':query_time_v4/max(query_time_v5,1e-12),
        'amortized_speedup':query_time_v4/(query_time_v5+v5.precompute_seconds),
        'candidate_fraction':checked/max(full,1),'fast_query_fraction':fast/max(fast+fallback,1),
        'disagreements':disagreements,'unsafe_or_not_certifiable_batched':unsafe}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,default=6);p.add_argument('--output',default='outputs/v5_geometry.json')
    a=p.parse_args()
    rows=[]
    for n,q in [(120,30),(1000,30),(5000,30)]:
      for s in ('sparse','mixed','dense'):
        z=run(n,q,scenario=s,repeats=a.repeat)
        rows.append(z);print(n,s,'speedup',round(z['per_gate_speedup'],2),'amortized',round(z['amortized_speedup'],2),'candidates',round(z['candidate_fraction'],3),'fast',round(z['fast_query_fraction'],2),flush=True)
    pth=Path(a.output);pth.parent.mkdir(parents=True,exist_ok=True);pth.write_text(json.dumps(rows,indent=2))
