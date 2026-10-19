"""Same sparse patches fed to overlay vs dense exhaustive gate.

V4 dense baseline must materialize a full matrix; v5 overlay does not.
Both preprocessing and per-proposal time reported separately, 100 reuses.
Only synthetic geometry. Real training usually modifies many embeddings,
so these sparse figures DO NOT imply end-to-end CKGE gains.
"""
from time import perf_counter
from pathlib import Path
import argparse,json
import numpy as np
from safecadence.sparse_overlay_v5 import SparseOverlayCertificate,SparsePatch
from safecadence.batched_certificate_v4 import BatchedRankCertificate


def bench(N=5000,Q=30,d=24,M=5,reps=100,seed=12):
    rng=np.random.default_rng(seed)
    E=rng.normal(0,.2,(N,d))
    R=rng.normal(0,.002,(2,d))
    heads=rng.choice(N,Q,replace=False)
    triples=np.stack([heads,np.zeros(Q,dtype=int),heads],axis=1)
    t=perf_counter();sc=SparseOverlayCertificate(E,R,triples,triples,k=10,shortlist_size=32);setup_s=perf_counter()-t
    t=perf_counter();brute=BatchedRankCertificate(E,R,triples,triples,k=10);setup_b=perf_counter()-t
    time_s=time_b=0;scan=full=rejects=diff=0;fallback=0
    for _ in range(reps):
        ix=rng.choice(N,M,replace=False)
        update=E[ix]+rng.normal(0,.004,(M,d))
        patch=SparsePatch(ix,update,R)
        start=perf_counter();res=sc.verify_patch(patch);time_s+=perf_counter()-start
        if sc.last_stats.get('fallback_queries'):fallback+=1
        scan+=res.inspected_candidates;full+=res.full_candidates
        start=perf_counter()
        full_E=E.copy();full_E[ix]=update
        ref=brute.verify(full_E,R)
        time_b+=perf_counter()-start
        diff+=res.safe!=ref.safe
        rejects+=not res.safe
    return dict(N=N,Q=len(sc.protected),d=d,changed=M,proposals=reps,
                overlay_build=setup_s,brute_build=setup_b,
                overlay_proposals_seconds=time_s,brute_proposals_seconds=time_b,
                proposal_speedup=time_b/time_s,
                build_amortized_speedup=(time_b+setup_b)/(time_s+setup_s),
                fraction_scored=scan/max(1,full),fallback_proposals=fallback,
                safe_decision_disagreements=diff,unsafe_decisions=rejects)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--reps',type=int,default=100);a.add_argument('--output',default='outputs/v5_overlay_benchmark.json')
    opts=a.parse_args();rows=[]
    for N,Q,M in [(1000,30,3),(5000,30,5),(20000,30,15),(5000,30,150)]:
        x=bench(N,Q,M=M,reps=opts.reps)
        rows.append(x)
        print(N,M,'speedup',round(x['proposal_speedup'],2),'amortized',round(x['build_amortized_speedup'],2),'fraction',round(x['fraction_scored'],4),'disagree',x['safe_decision_disagreements'],flush=True)
    path=Path(opts.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rows,indent=2))
