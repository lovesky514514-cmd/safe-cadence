import numpy as np
from safecadence.diagnostics_v3 import make_batches, run_variant, Variant
from safecadence.certificate_cache import CachedRankCertificate


def toy():
    rng=np.random.default_rng(31)
    E=rng.normal(size=(20,6)); E/=np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1)
    R=rng.normal(size=(2,6))*.1
    old=np.asarray([(i,i%2,(i+3)%20) for i in range(12)],dtype=np.int64)
    new=np.asarray([(i,i%2,(i+4)%20) for i in range(12)],dtype=np.int64)
    return E,R,old,new


def test_same_batch_stream_for_comparison():
    E,R,old,new=toy()
    aa=make_batches(new,np.concatenate((old,new)),len(E),steps=4,seed=16)
    bb=make_batches(new,np.concatenate((old,new)),len(E),steps=4,seed=16)
    for (ap,an),(bp,bn) in zip(aa,bb):
        assert np.array_equal(ap,bp)
        assert np.array_equal(an,bn)


def test_certified_endpoint_agrees_with_exhaustive_checker():
    E,R,old,new=toy()
    data=make_batches(new,np.concatenate((old,new)),len(E),steps=6,seed=7)
    e,r,stats=run_variant(E,R,new,old,old,data,Variant('cert',certified=True),
                          base_lr=.2,k=10,audit_decisions=True)
    assert stats['gate_false_accepts']==0
    assert stats['gate_false_rejects']==0
    assert stats['endpoint_safe_exhaustive']
    assert CachedRankCertificate(E,R,old,old,k=10).verify_exhaustive(e,r,added_true=new).safe
    assert stats['rescued_steps']<=stats['first_alpha_unsafe']


def test_blind_baseline_has_no_certificate_overhead():
    E,R,old,new=toy()
    data=make_batches(new,np.concatenate((old,new)),len(E),steps=4,seed=11)
    e,r,stats=run_variant(E,R,new,old,old,data,Variant('baseline',lr_factor=.1),
                          base_lr=.2,k=10)
    assert stats['certificate_calls']==0
    assert stats['certificate_seconds']==0.0
    assert stats['method_wall_seconds']<stats['training_and_audit_wall_seconds']
