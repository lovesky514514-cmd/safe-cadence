import numpy as np
import pytest
from safecadence.sparse_overlay_v5 import SparseOverlayCertificate,SparsePatch
from safecadence.batched_certificate_v4 import BatchedRankCertificate

@pytest.mark.parametrize('seed',[11,12,13,14,15])
def test_overlay_never_falsely_accepts(seed):
    rng=np.random.default_rng(seed)
    N=220;d=12
    E=rng.normal(0,.1,(N,d));R=rng.normal(0,.001,(2,d))
    hs=rng.choice(N,25,replace=False)
    triples=np.stack([hs,np.zeros(len(hs),dtype=int),hs],axis=1)
    cert=SparseOverlayCertificate(E,R,triples,triples,k=10,shortlist_size=16)
    brute=BatchedRankCertificate(E,R,triples,triples,k=10)
    for j in range(14):
        ids=rng.choice(N,3,replace=False)
        vectors=E[ids]+rng.normal(0,.01,(3,d))
        new_r=R+rng.normal(0,.0003,R.shape)
        added=rng.normal(0,.1,(2,d)) if j%4==0 else np.empty((0,d))
        patch=SparsePatch(ids,vectors,new_r,added)
        actual=np.concatenate((E.copy(),added))
        actual[ids]=vectors
        z=cert.verify_patch(patch)
        ref=brute.verify(actual,new_r)
        if z.safe:assert ref.safe
        assert z.inspected_candidates<=z.full_candidates


def test_patch_cannot_hide_entity_changes():
    E=np.asarray([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    R=np.zeros((1,2));tr=np.array([[0,0,0]],dtype=np.int64)
    c=SparseOverlayCertificate(E,R,tr,tr,k=2,shortlist_size=2)
    with pytest.raises(ValueError):
        c.verify_patch(SparsePatch(np.array([1,1]),np.zeros((2,2)),R))
    with pytest.raises(ValueError):
        c.verify_patch(SparsePatch(np.array([5]),np.zeros((1,2)),R))
    assert c.verify_patch(SparsePatch(np.array([],dtype=int),np.empty((0,2)),R)).safe
