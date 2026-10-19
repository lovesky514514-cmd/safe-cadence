"""Property tests against exhaustive enumeration, with held-out and new entities."""
import numpy as np
from safecadence.certificate_v2 import SparseRadiusCertificate

def test_sparse_dense_updates_equal_exhaustive():
    for seed in (1,2,3,4):
        rng=np.random.default_rng(seed)
        E=rng.normal(size=(420,12)); R=rng.normal(size=(5,12))*.15
        triples=np.array([(i,i%5,(i*7+19)%420) for i in range(75)])
        for k in (1,10,75,420):
            checker=SparseRadiusCertificate(E,R,triples,triples,k=k)
            for t in range(4):
                E1=E.copy(); R1=R.copy()
                touched=rng.choice(420,size=3 if t<3 else 420,replace=False)
                E1[touched]+=rng.normal(0,[.001,.05,.7,.2][t],(len(touched),12))
                if t%2: R1+=rng.normal(0,.03,R.shape)
                quick=checker.verify(E1,R1,stop_early=False)
                exact=checker.verify_exhaustive(E1,R1,stop_early=False)
                assert quick.safe==exact.safe
                assert quick.inspected_candidates<=exact.inspected_candidates

def test_appended_entity_can_displace_protected_answer():
    E=np.array([[0.],[3.],[1.],[8.],[15.]])
    R=np.array([[1.]])
    triple=np.array([[0,0,2]])
    checker=SparseRadiusCertificate(E,R,triple,triple,k=1,sides=('tail',),min_index_entities=1)
    assert checker.verify(E,R).safe
    with_added=np.vstack([E,[1.]])
    assert not checker.verify(with_added,R).safe
    assert checker.verify(with_added,R).safe==checker.verify_exhaustive(with_added,R).safe
    assert checker.verify(with_added,R,added_true=[[0,0,5]]).safe

def test_dense_parameter_change():
    rng=np.random.default_rng(42)
    E=rng.normal(size=(355,7));R=rng.normal(size=(3,7))
    triples=np.array([(i,i%3,(i*9+17)%355) for i in range(80)])
    checker=SparseRadiusCertificate(E,R,triples,triples,k=25)
    E1=E+rng.normal(0,.45,E.shape);R1=R+rng.normal(0,.2,R.shape)
    assert checker.verify(E1,R1,stop_early=False).safe==checker.verify_exhaustive(E1,R1,stop_early=False).safe

def test_empty_protected_is_vacuous_and_counted():
    E=np.array([[0.],[1.],[2.],[10.]])
    R=np.array([[0.]])
    checker=SparseRadiusCertificate(E,R,np.array([[0,0,3]]),[[0,0,3]],k=1,sides=('tail',))
    assert not checker.protected
    assert checker.verify(E,R).checked_queries==0
