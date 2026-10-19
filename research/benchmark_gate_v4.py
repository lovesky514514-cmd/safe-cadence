"""Synthetic geometry gate comparison; includes index build and full-candidate cost."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from safecadence.certificate_cache import CachedRankCertificate
from safecadence.batched_certificate_v4 import BatchedRankCertificate

def case(n,d=24,q=50,repeats=12,dense=False,seed=19):
    rng=np.random.default_rng(seed+n+(7 if dense else 0))
    E=rng.standard_normal((n,d));E/=np.linalg.norm(E,axis=1,keepdims=True)
    R=rng.standard_normal((4,d))*.1
    triples=[]
    for _ in range(q):
        h=int(rng.integers(n));r=int(rng.integers(4))
        query=E[h]+R[r]
        dist=np.linalg.norm(E-query,axis=1)
        dist[h]=np.inf
        t=int(np.argmin(dist));triples.append([h,r,t])
    old=np.asarray(triples,dtype=np.int64)
    setup={}
    for engine,cls in [('cached',CachedRankCertificate),('batched',BatchedRankCertificate)]:
        start=perf_counter()
        if engine=='cached': cert=cls(E,R,old,old,k=10,sides=('tail',),max_cache_mb=128)
        else: cert=cls(E,R,old,old,k=10,sides=('tail',),added_true=old)
        setup[engine]={'checker':cert,'build_s':perf_counter()-start}
    trials=[]
    for _ in range(repeats):
        E1=E.copy()
        idx=rng.choice(n,size=(n if dense else max(1,n//100)),replace=False)
        E1[idx]+=rng.normal(0,.01,size=(len(idx),d))
        for engine in ('cached','batched'):
            t=perf_counter()
            res=setup[engine]['checker'].verify(E1,R,added_true=old,stop_early=True)
            elapsed=perf_counter()-t
            setup[engine].setdefault('times',[]).append(elapsed)
            setup[engine].setdefault('scored',[]).append(res.inspected_candidates/res.full_candidates if res.full_candidates else 0)
            setup[engine].setdefault('safety',[]).append(res.safe)
        assert setup['cached']['safety'][-1]==setup['batched']['safety'][-1]
    return {'n_entities':n,'n_triples':q,'dimension':d,'dense':dense,'repeats':repeats,
            'engines':{name:{'build_s':v['build_s'],'avg_verify_s':float(np.mean(v['times'])),
                            'amortized_total_s':float(v['build_s']+np.sum(v['times'])),
                            'avg_fraction_scored':float(np.mean(v['scored']))}
                       for name,v in setup.items()}}
if __name__=='__main__':
    out=[]
    for n in [120,1000,5000]:
        for dense in [False,True]:
            row=case(n,dense=dense);out.append(row)
            print(n,'dense' if dense else 'sparse',
                  {k:(round(v['build_s'],4),round(v['avg_verify_s'],5),round(v['amortized_total_s'],3),round(v['avg_fraction_scored'],3)) for k,v in row['engines'].items()},flush=True)
    Path('outputs/v4_gate_benchmark.json').write_text(json.dumps({'protocol':'synthetic geometry; same candidate models; all-true filtered; 12 independent updates', 'rows':out},indent=2))
