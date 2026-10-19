from fractions import Fraction
from itertools import product
import numpy as np
import pytest
from safecadence_preflight.frozen_append_guard import FrozenAppendGuard


def exact_dist_sq(q, x):
    return sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2
               for a,b in zip(q,x))


def exact_tail_safe(E0,R,old_triple,truth,appended,k):
    h,r,t=old_triple
    q=E0[h]+R[r]
    ans=exact_dist_sq(q,E0[t])
    full=np.concatenate((E0,appended))
    banned={z for a,b,z in truth if (a,b)==(h,r)}-{t}
    rivals=sum((exact_dist_sq(q,row)<=ans) for j,row in enumerate(full)
               if j!=t and j not in banned)
    return rivals<k


def test_append_unsafe_and_safe():
    E=np.array([[0.],[2.],[4.]],dtype=np.float64);R=np.array([[0.]])
    prot=np.array([[0,0,1]]);guard=FrozenAppendGuard(E,R,prot,prot,k=2)
    assert len(guard.query)>=1
    assert guard.verify_appended(np.array([[20.]])).certified
    assert not guard.verify_appended(np.array([[1.5]])).certified


def test_known_true_appended_candidate_is_filtered():
    E=np.array([[0.],[2.],[4.]])
    prot=np.array([[0,0,1]])
    true=np.array([[0,0,1],[0,0,3]])
    guard=FrozenAppendGuard(E,np.array([[0.]]),prot,true,k=2)
    assert guard.verify_appended(np.array([[1.5]])).certified


def test_oracle_no_false_accept_for_random_appends():
    rng=np.random.default_rng(123)
    for _ in range(30):
        E=rng.uniform(-2,2,(7,2)).astype(np.float64)
        R=rng.uniform(-.1,.1,(1,2)).astype(np.float64)
        tr=np.array([[0,0,1]],dtype=np.int64)
        guard=FrozenAppendGuard(E,R,tr,tr,k=3)
        for j in range(3):
            a=rng.uniform(-2,2,(4,2)).astype(np.float64)
            d=guard.verify_appended(a)
            # Initially ineligible queries are excluded by definition, so an
            # oracle statement about them would not test the certificate.
            if d.certified and exact_tail_safe(E,R,(0,0,1),[(0,0,1)],np.empty((0,2)),3):
                assert exact_tail_safe(E,R,(0,0,1),[(0,0,1)],a,3)
    # This compares the exact rational distance of the STORED binary64 q.


def test_fail_closed_on_nonfinite_and_bad_shape():
    E=np.array([[0.],[1.],[2.]])
    g=FrozenAppendGuard(E,np.zeros((1,1)),np.array([[0,0,1]]),np.array([[0,0,1]]),k=2)
    with pytest.raises(ValueError):g.verify_appended(np.array([[float('nan')]]))
    with pytest.raises(ValueError):g.verify_appended(np.array([[1.,2.]]))


def test_cpp_style_ties_are_conservatively_counted():
    E=np.array([[0.],[2.],[4.]])
    g=FrozenAppendGuard(E,np.zeros((1,1)),np.array([[0,0,1]]),np.array([[0,0,1]]),k=2)
    assert not g.verify_appended(np.array([[2.]])).certified


def test_no_initial_eligible_queries_fails_closed():
    # Answer at distance 100 already ranks worse than at least two other old
    # candidates for k=1. There is no initially valid rank to preserve.
    E = np.array([[0.], [100.], [1.], [2.]])
    g = FrozenAppendGuard(E, np.zeros((1, 1)), np.array([[0, 0, 1]]),
                          np.array([[0, 0, 1]]), k=1)
    d = g.verify_appended(np.array([[1000.]]))
    assert not d.certified
    assert d.status == 'NOT_CERTIFIED_EMPTY_SCOPE'
    assert g.excluded_initial == 2


def test_invalid_protection_limits_do_not_silently_coerce():
    ent = np.array([[0.], [1.]])
    triple = np.array([[0, 0, 1]])
    for kwargs in ({'max_protected': 0}, {'chunk_size': 0}, {'k': 0}):
        with pytest.raises(ValueError):
            FrozenAppendGuard(ent, np.zeros((1, 1)), triple, triple, **kwargs)

def test_scope_diagnostics_are_reported():
    ent = np.array([[0.], [2.], [4.]])
    triple = np.array([[0, 0, 1]])
    guard = FrozenAppendGuard(ent, np.zeros((1, 1)), triple, triple, k=2)
    outcome = guard.verify_appended(np.array([[20.]]))
    assert outcome.requested_triples == 1
    assert outcome.excluded_initial == guard.excluded_initial
    assert outcome.protected_queries == len(guard.query)
