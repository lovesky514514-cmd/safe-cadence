import numpy as np
from safecadence.hybrid_certificate_v5 import HybridRankCertificate

def test_small_graph_uses_exhaustive():
    rng=np.random.default_rng(3)
    E=rng.normal(size=(40,5));R=np.zeros((1,5))
    tr=np.asarray([[1,0,1],[2,0,2]])
    c=HybridRankCertificate(E,R,tr,tr,k=5,threshold_entities=100)
    r=c.verify(E,R)
    assert c.last_engine=='batched' and r.safe
    assert c.shortlist_construction_count==0

def test_sparse_large_lazy_index_and_dense_fallback():
    rng=np.random.default_rng(7)
    E=rng.normal(size=(220,6));R=np.zeros((1,6))
    tr=np.asarray([[i,0,i] for i in range(10)],dtype=np.int64)
    c=HybridRankCertificate(E,R,tr,tr,k=10,threshold_entities=200,max_changed_fraction=.03)
    E1=E.copy();E1[200]+=.001
    assert c.verify(E1,R).safe
    assert c.last_engine=='shortlist' and c.shortlist_construction_count==1
    assert c.verify(E1,R).safe and c.shortlist_construction_count==1
    E1=E+.001
    _=c.verify(E1,R)
    assert c.last_engine=='batched'
