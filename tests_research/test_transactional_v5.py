import numpy as np
from safecadence.transactional_v5 import transactional_updates
from safecadence.diagnostics_v3 import make_batches
from safecadence.batched_certificate_v4 import BatchedRankCertificate


def test_public_checkpoints_always_safe_even_if_staging_unsafe():
    rng=np.random.default_rng(16)
    E=rng.normal(0,.15,(40,6));R=rng.normal(0,.002,(1,6))
    tris=np.asarray([[i,0,i] for i in range(28)])
    batches=make_batches(tris[20:],tris,len(E),steps=18,batch_size=8,negatives=2,seed=16)
    En,Rn,z=transactional_updates(E,R,tris[:20],tris[20:],tris[:20],batches,commit_every=3)
    assert z['published_checkpoints_safe']
    assert z['commits']+z['rollbacks']==6
    assert BatchedRankCertificate(E,R,tris[:20],tris[:20],added_true=tris[20:]).verify(En,Rn).safe
