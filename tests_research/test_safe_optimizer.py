import numpy as np
from safecadence.certificate_cache import CachedRankCertificate
from safecadence.safe_optimizer import certified_minibatch_sgd

def test_certified_optimizer_always_satisfies_exact_checker():
    rng=np.random.default_rng(14)
    E=rng.normal(size=(42,7)); E/=np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1)
    R=rng.normal(size=(2,7))*.1
    old=np.array([(i,i%2,(i*3+7)%42) for i in range(25)])
    incoming=np.array([[(i*5)%42,i%2,(i*9+2)%42] for i in range(15)])
    E2,R2,meta=certified_minibatch_sgd(E,R,incoming,old,old,
                                      steps=8,batch_size=12,negatives=1,
                                      lr=.15,k=10,seed=12)
    checker=CachedRankCertificate(E,R,old,old,k=10)
    assert checker.verify_exhaustive(E2,R2,added_true=incoming).safe
    assert meta['accepted_steps']+meta['rejected_steps']==8
