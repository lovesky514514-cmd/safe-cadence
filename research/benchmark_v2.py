"""Compare OLD global-drift radius, NEW sparse-aware radius and exhaustive.
No training data or learned benchmark metrics. Random initialized geometry test.
Includes every certification runtime and index-build time, never hides these.
"""
import argparse,json
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.spatial import cKDTree
from safecadence.certificate import RadiusCertificate
from safecadence.certificate_v2 import SparseRadiusCertificate
from safecadence.certificate_cache import CachedRankCertificate


def one(seed,n,dim,nq,scenario):
    rng=np.random.default_rng(seed)
    E=rng.normal(0,1,(n,dim))
    R=rng.normal(0,.09,(4,dim))
    tree=cKDTree(E)
    triples=[]
    for i in range(nq):
        head=int(rng.integers(0,n));rel=int(rng.integers(0,4))
        tails=tree.query(E[head]+R[rel],k=3)[1]
        t=int(tails[1] if int(tails[0])==head else tails[0])
        triples.append((head,rel,t))
    triples=np.asarray(triples,dtype=np.int64)
    E1=E.copy();R1=R.copy()
    if scenario=='sparse':
        changed=rng.choice(n,max(1,int(n*.01)),replace=False)
        E1[changed]+=rng.normal(0,.12,(len(changed),dim))
        R1+=rng.normal(0,.004,R.shape)
    elif scenario=='dense':
        E1+=rng.normal(0,.03,E.shape)
        R1+=rng.normal(0,.012,R.shape)
    elif scenario=='appended':
        changed=rng.choice(n,max(1,int(n*.01)),replace=False)
        E1[changed]+=rng.normal(0,.05,(len(changed),dim))
        E1=np.vstack([E1,rng.normal(0,1,(max(1,n//100),dim))])
        R1+=rng.normal(0,.004,R.shape)
    from time import perf_counter
    t=perf_counter()
    fast=SparseRadiusCertificate(E,R,triples,triples,k=10,sides=('tail',))
    build_new=perf_counter()-t
    new=fast.verify(E1,R1,stop_early=False)
    exact=fast.verify_exhaustive(E1,R1,stop_early=False)
    if new.safe!=exact.safe: raise AssertionError('v2 unsound')
    row={'seed':seed,'n':n,'dim':dim,'queries':nq,'scenario':scenario,
         'protected':len(fast.protected),'safe':new.safe,
         'fraction':new.inspected_candidates/max(1,new.full_candidates),
         'build_v2_s':build_new,'new_s':new.verification_seconds,
         'exact_s':exact.verification_seconds,
         'amortized_10_new_s':new.verification_seconds+build_new/10,
         'full_new_s':new.verification_seconds+build_new,
         'amortized_10_speedup':exact.verification_seconds/(new.verification_seconds+build_new/10),
         'first_speedup':exact.verification_seconds/(new.verification_seconds+build_new)}
    t=perf_counter()
    cache=CachedRankCertificate(E,R,triples,triples,k=10,sides=('tail',))
    build_cache=perf_counter()-t
    cached=cache.verify(E1,R1,stop_early=False)
    if cached.safe!=exact.safe: raise AssertionError('cached unsound')
    row.update(cache_s=cached.verification_seconds,cache_build_s=build_cache,
               cache_fraction=cached.inspected_candidates/max(1,cached.full_candidates),
               cache_mem_mb=cache.cache_bytes/1048576,
               cache_first_speedup=exact.verification_seconds/(cached.verification_seconds+build_cache),
               cache_10_speedup=exact.verification_seconds/(cached.verification_seconds+build_cache/10),
               cache_100_speedup=exact.verification_seconds/(cached.verification_seconds+build_cache/100))
    if scenario!='appended':
        t=perf_counter(); old=RadiusCertificate(E,R,triples,triples,k=10,sides=('tail',))
        old_build=perf_counter()-t
        old_result=old.verify(E1,R1,stop_early=False)
        if old_result.safe!=exact.safe: raise AssertionError('v0.1 mismatch')
        row.update(old_s=old_result.checking_seconds,old_build_s=old_build,
                   old_fraction=old_result.inspected_candidates/max(1,old_result.full_candidates))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sizes',type=int,nargs='+',default=[120,1000,5000])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--queries',type=int,default=60)
    p.add_argument('--dim',type=int,default=24)
    p.add_argument('--out',type=Path,default=Path('outputs/v2_benchmark.json'))
    args=p.parse_args()
    rows=[]
    for n in args.sizes:
        for scenario in ('sparse','dense','appended'):
            for seed in args.seeds:
                a=one(seed,n,args.dim,args.queries,scenario)
                rows.append(a)
                print(f'{n:>6} {scenario:>8} seed={seed} candidates={a["fraction"]:.3f} speed1={a["first_speedup"]:.2f} speed10={a["amortized_10_speedup"]:.2f} safe={a["safe"]}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'protocol':'synthetic_geometry_only','rows':rows},indent=2),encoding='utf-8')
    print('SAVED',args.out)

if __name__=='__main__':main()
