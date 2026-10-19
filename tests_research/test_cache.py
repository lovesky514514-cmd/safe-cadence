import numpy as np
from safecadence.certificate_cache import CachedRankCertificate

def test_cache_exhaustive_sparse_dense_new_entities_and_filter():
    for seed in range(5):
        rng=np.random.default_rng(seed)
        E=rng.normal(size=(480,14)); R=rng.normal(size=(4,14))
        old=np.array([(i,i%4,(i*11+31)%480) for i in range(95)])
        ch=CachedRankCertificate(E,R,old,old,k=30,sides=('head','tail'))
        assert ch.cache_enabled
        for kind in ('nochange','sparse','dense','appended'):
            E1=E.copy();R1=R.copy()
            if kind=='sparse':
                E1[rng.choice(480,5,replace=False)]+=rng.normal(0,.03,(5,14))
                R1+=rng.normal(0,.01,R.shape)
            if kind=='dense':
                E1+=rng.normal(0,.2,E.shape);R1+=rng.normal(0,.1,R.shape)
            if kind=='appended':
                E1=np.vstack((E,rng.normal(size=(8,14))))
            f=ch.verify(E1,R1,stop_early=False)
            x=ch.verify_exhaustive(E1,R1,stop_early=False)
            assert f.safe == x.safe, (seed,kind)
            assert f.inspected_candidates <= x.inspected_candidates

def test_cache_fallback_matches_exhaustive():
    rng=np.random.default_rng(2);E=rng.normal(size=(400,3));R=rng.normal(size=(1,3))
    old=np.array([(i,0,(i+2)%400) for i in range(50)])
    ch=CachedRankCertificate(E,R,old,old,k=20,max_cache_mb=.00001)
    assert not ch.cache_enabled
    E1=E+rng.normal(0,.01,E.shape)
    assert ch.verify(E1,R).safe == ch.verify_exhaustive(E1,R).safe
