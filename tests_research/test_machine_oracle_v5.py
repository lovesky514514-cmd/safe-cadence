"""Independent exact-dyadic integer-rational oracle, separate from NumPy ranks."""
from fractions import Fraction
import numpy as np
from safecadence.batched_shortlist_v5 import BatchedShortlistCertificate
from safecadence.interval_v5 import sqdistance_pairs_fast
from safecadence.ranking import query_and_answer,TrueFacts


def exact_rank_stored_query(E,R,tr,side,truth):
    q,ans=query_and_answer(E,R,tr,side) # model defines q as binary64 h+r or t-r
    def dist2(i):
        return sum((Fraction(float(q[j]))-Fraction(float(E[i,j])))**2 for j in range(len(q)))
    answer=dist2(ans)
    ignored=truth.ignored(tr,side)
    return 1+sum(1 for i in range(len(E)) if i!=ans and i not in ignored and dist2(i)<=answer)


def test_vector_interval_bounds_against_exact_rationals():
    rng=np.random.default_rng(8)
    for _ in range(60):
        q=rng.normal(size=(5,3));x=rng.normal(size=(5,3))
        lo,hi=sqdistance_pairs_fast(q,x)
        for i in range(len(q)):
            exact=sum((Fraction(float(q[i,j]))-Fraction(float(x[i,j])))**2 for j in range(3))
            assert Fraction(float(lo[i]))<=exact<=Fraction(float(hi[i]))


def test_full_machine_oracle_near_ties_and_edits():
    rng=np.random.default_rng(99)
    E=rng.normal(0,.1,(35,3));R=rng.normal(0,.02,(2,3))
    tris=np.array([[j%35,j%2,j%35] for j in range(10)],dtype=np.int64)
    cert=BatchedShortlistCertificate(E,R,tris,tris,k=6,shortlist_size=8)
    truth=TrueFacts(tris)
    for step in range(45):
        x=E.copy();r=R.copy()
        if step%3==0:
            x[rng.integers(len(E))]+=rng.normal(0,.04,3)
        elif step%3==1:
            r+=rng.normal(0,.0001,r.shape)
        else:
            x+=rng.normal(0,.001,x.shape)
        if step%7==0:
            x[step%len(E)]=np.nextafter(x[(step+1)%len(E)],np.inf)
        if step%9==0:x=np.vstack((x,rng.normal(0,.1,(2,3))))
        result=cert.verify(x,r)
        exact_safe=all(exact_rank_stored_query(x,r,tr,side,truth)<=6 for tr,side in cert.protected)
        assert not result.safe or exact_safe


def test_outward_pairs_near_subnormal_and_ties():
    from safecadence.interval_v5 import sqdistance_pairs_outward
    xs=np.asarray([[0.,0.],[1e-190,2e-190],[1.,-1.],[1+2**-50,-1.]])
    qs=np.asarray([[1e-190,0.],[0.,1e-190],[1.,-1.],[1.,-1.]])
    lo,hi=sqdistance_pairs_outward(qs,xs)
    for i in range(len(xs)):
        exact=sum((Fraction(float(qs[i,j]))-Fraction(float(xs[i,j])))**2 for j in range(2))
        assert Fraction(float(lo[i]))<=exact<=Fraction(float(hi[i]))


def test_strict_outward_gate_no_false_accept():
    rng=np.random.default_rng(4)
    E=rng.normal(size=(50,4));R=np.zeros((1,4))
    triples=np.asarray([[i,0,i] for i in range(8)])
    cert=BatchedShortlistCertificate(E,R,triples,triples,k=5,shortlist_size=7,strict_rounding=True)
    for j in range(12):
        E1=E.copy();E1[rng.integers(len(E))]+=rng.normal(0,.1,4)
        z=cert.verify(E1,R)
        if z.safe:
            assert all(exact_rank_stored_query(E1,R,tr,side,TrueFacts(triples))<=5 for tr,side in cert.protected)
