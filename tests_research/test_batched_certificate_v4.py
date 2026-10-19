import numpy as np
from safecadence.batched_certificate_v4 import BatchedRankCertificate


def test_randomized_against_independent_exhaustive():
    rng=np.random.default_rng(9)
    for n in (20,80):
        for _ in range(8):
            E0=rng.normal(size=(n,7));R0=rng.normal(size=(3,7))
            known=np.stack((rng.integers(n,size=20),rng.integers(3,size=20),rng.integers(n,size=20)),axis=1)
            added=known[:3].copy()
            checker=BatchedRankCertificate(E0,R0,known,known,k=10,added_true=added)
            E1=E0+rng.normal(size=E0.shape)*rng.choice([0.,.02,.2])
            R1=R0+rng.normal(size=R0.shape)*.01
            assert checker.verify(E1,R1).safe==checker.verify_exhaustive(E1,R1,added_true=added).safe

def test_appended_and_extreme_failures():
    E=np.asarray([[0.,0.],[2.,0.],[10.,0.]])
    R=np.asarray([[2.,0.]])
    triples=np.asarray([[0,0,1]])
    checker=BatchedRankCertificate(E,R,triples,triples,k=1,added_true=triples)
    assert checker.verify(E,R).safe is True
    append=np.concatenate((E,np.asarray([[2.,0.]])))
    assert checker.verify(append,R).safe is False

def test_unknown_truth_change_rejected():
    E=np.asarray([[0.,0.],[1.,0.]])
    R=np.asarray([[0.,0.]])
    triples=np.asarray([[0,0,1]])
    c=BatchedRankCertificate(E,R,triples,triples,k=2,added_true=triples)
    import pytest
    with pytest.raises(ValueError):c.verify(E,R,added_true=np.asarray([[1,0,0]]))
