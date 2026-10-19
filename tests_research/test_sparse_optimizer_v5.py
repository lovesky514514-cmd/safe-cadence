import numpy as np
import pytest
import torch
from torch.nn import functional as F
from safecadence.sparse_optimizer_v5 import sparse_entity_margin_gradient, run_certified_sparse_entity
from safecadence.diagnostics_v3 import make_batches


def test_analytic_sparse_margin_gradient_matches_torch():
    rng=np.random.default_rng(44)
    E=rng.normal(0,.2,(24,5));R=rng.normal(0,.1,(3,5))
    pos=np.array([[0,0,1],[2,1,3],[4,2,5]])
    neg=np.array([[0,0,7],[2,0,1],[2,1,8],[9,1,3],[7,2,5],[4,2,0]])
    ids,g,loss=sparse_entity_margin_gradient(E,R,pos,neg)
    et=torch.tensor(E,dtype=torch.float64,requires_grad=True)
    rt=torch.tensor(R,dtype=torch.float64)
    def dist(tr):return torch.linalg.vector_norm(et[tr[:,0]]+rt[tr[:,1]]-et[tr[:,2]],dim=1)
    obj=F.relu(1.+dist(pos).repeat_interleave(2)-dist(neg)).mean()
    grad=torch.autograd.grad(obj,et)[0].numpy()
    assert np.allclose(g,grad[ids],atol=1e-14)
    assert np.isclose(loss,obj.item())
    assert np.allclose(grad[np.setdiff1d(np.arange(len(E)),ids)],0)


@pytest.mark.parametrize('seed',[4,5])
def test_same_patch_trajectory_as_exhaustive(seed):
    rng=np.random.default_rng(seed)
    E=rng.normal(0,.1,(70,6));R=rng.normal(0,.002,(1,6))
    all_tr=np.asarray([[i,0,i] for i in range(35)],dtype=np.int64)
    old=all_tr[:22];incoming=all_tr[22:]
    batches=make_batches(incoming,all_tr,len(E),steps=12,batch_size=8,negatives=2,seed=seed)
    o=run_certified_sparse_entity(E,R,old,incoming,old,batches,engine='overlay',shortlist_size=12)
    b=run_certified_sparse_entity(E,R,old,incoming,old,batches,engine='batched',shortlist_size=12)
    assert o.stats['endpoint_safe'] and b.stats['endpoint_safe']
    assert o.stats['accepted']==b.stats['accepted']
    assert np.allclose(o.E,b.E,atol=1e-14)
