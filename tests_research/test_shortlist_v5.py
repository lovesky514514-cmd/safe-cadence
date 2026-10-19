import numpy as np
import pytest
from safecadence.shortlist_certificate_v5 import ShortlistRankCertificate
from safecadence.interval_v5 import sqdistance_bounds
from safecadence.ranking import TrueFacts,exact_rank


def _fixture(seed=0,N=160,d=8):
    rng=np.random.default_rng(seed)
    E=rng.normal(0,.1,(N,d));R=rng.normal(0,.01,(4,d))
    triples=np.asarray([[int(rng.integers(N)),int(rng.integers(4)),int(rng.integers(N))] for _ in range(28)])
    # k=N guarantees all protected, exercises soundness in tests with lower k too
    return E,R,triples


def test_interval_enclosure_decimal():
    from fractions import Fraction
    rng=np.random.default_rng(99)
    for _ in range(50):
        q=rng.normal(size=5); x=rng.normal(size=(7,5))
        lo,hi=sqdistance_bounds(q,x)
        for i in range(len(x)):
            exact=sum((Fraction(float(q[j]))-Fraction(float(x[i,j])))**2 for j in range(5))
            assert Fraction(float(lo[i]))<=exact<=Fraction(float(hi[i]))

@pytest.mark.parametrize('seed',[1,2,3,4,5])
def test_never_false_accept(seed):
    E,R,tr=_fixture(seed,N=125)
    checker=ShortlistRankCertificate(E,R,tr,tr,k=10,shortlist_size=12)
    rng=np.random.default_rng(seed)
    for case in range(9):
        E1=E.copy();R1=R.copy()
        if case%3==0:E1[rng.choice(len(E),3,replace=False)]+=rng.normal(0,.04,(3,E.shape[1]))
        elif case%3==1:R1+=rng.normal(0,.003,R1.shape)
        else:E1+=rng.normal(0,.002,E1.shape)
        if case in (5,7):E1=np.vstack((E1,rng.normal(0,.2,(3,E.shape[1]))))
        quick=checker.verify(E1,R1)
        full=checker.verify_exhaustive(E1,R1)
        if quick.safe: assert full.safe
        assert quick.inspected_candidates<=quick.full_candidates


def test_added_true_protocol_and_entities():
    E,R,tr=_fixture(17,N=100)
    added=np.asarray([[2,1,11]])
    cert=ShortlistRankCertificate(E,R,tr,tr,k=10,added_true=added)
    E2=np.vstack((E,np.zeros((2,E.shape[1]))))
    assert cert.verify(E2,R,added_true=added).safe == cert.verify_exhaustive(E2,R,added_true=added).safe
    with pytest.raises(ValueError):cert.verify(E2,R,added_true=np.asarray([[3,1,11]]))


def test_tie_conservatism():
    E=np.asarray([[0.],[0.],[0.],[2.],[3.]])
    R=np.zeros((1,1));tr=np.array([[0,0,0]],dtype=np.int64)
    cert=ShortlistRankCertificate(E,R,tr,tr,k=3,shortlist_size=3)
    assert cert.verify(E,R).safe
    assert cert.verify_exhaustive(E,R).safe
    # With k=2 the initial answer is not eligible (two exact ties).
    with pytest.raises(ValueError, match='Zero initially Top-k'):
        ShortlistRankCertificate(E,R,tr,tr,k=2,shortlist_size=2)

@pytest.mark.parametrize('seed',[1,3,5])
def test_batched_never_false_accept(seed):
    from safecadence.batched_shortlist_v5 import BatchedShortlistCertificate
    E,R,tr=_fixture(seed,N=150)
    cert=BatchedShortlistCertificate(E,R,tr,tr,k=10,shortlist_size=14,query_chunk=13)
    rng=np.random.default_rng(seed+10)
    for j in range(10):
        E1=E.copy();R1=R.copy()
        if j%2:E1[rng.choice(len(E),5,replace=False)]+=rng.normal(0,.1,(5,E.shape[1]))
        else:E1+=rng.normal(0,.003,E.shape);R1+=rng.normal(0,.002,R.shape)
        if j==9:E1=np.vstack((E1,rng.normal(0,.1,(4,E.shape[1]))))
        a=cert.verify(E1,R1); b=cert.verify_exhaustive(E1,R1)
        assert not a.safe or b.safe
