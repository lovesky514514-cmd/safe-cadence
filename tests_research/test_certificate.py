"""Property-style tests compare every fast answer to exact enumeration."""
import numpy as np
import pytest
from safecadence.certificate import RadiusCertificate
from safecadence.ranking import exact_rank, TrueFacts
from safecadence.data import synthetic_graph, audit_splits
from safecadence.transe import TrainConfig, train_transe


def test_rank_filtered_removes_other_true_answers():
    E = np.array([[0.], [1.], [2.], [3.]], dtype=float)
    R = np.zeros((1,1))
    facts = TrueFacts([(0,0,1), (0,0,2)])
    # 0 is closer to query=0 than the answer=2, 1 is true and filtered.
    assert exact_rank(E, R, (0,0,2), 'tail', facts) == 2


def test_certificate_agrees_with_exhaustive_on_random_updates():
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        E = rng.normal(size=(55, 7))
        R = rng.normal(size=(3, 7)) * 0.1
        triples = np.array([(i, i%3, (i*7+13)%55) for i in range(30)])
        for k in (1, 3, 10, 30):
            cert = RadiusCertificate(E, R, triples, triples, k=k)
            for trial in range(12):
                next_E, next_R = E.copy(), R.copy()
                ix = rng.choice(len(E), size=7, replace=False)
                next_E[ix] += rng.normal(size=(len(ix),7))*[0.001,0.04,0.15][trial%3]
                next_R += rng.normal(size=R.shape)*[0.001,0.04,0.15][trial%3]
                fast = cert.verify(next_E, next_R, stop_early=False)
                full = cert.verify_exhaustive(next_E, next_R, stop_early=False)
                assert fast.safe == full.safe
                assert fast.inspected_candidates <= full.inspected_candidates


def test_intentionally_break_rank_is_rejected():
    E = np.array([[0.0], [10.0], [1.0], [20.0]])
    R = np.array([[1.0]])
    triple = np.array([[0, 0, 2]])
    cert = RadiusCertificate(E, R, triple, triple, k=1, sides=('tail',))
    assert len(cert.protected) == 1
    broken = E.copy()
    broken[2, 0] = 20.0
    broken[1, 0] = 1.0
    assert not cert.verify(broken, R).safe
    assert not cert.verify_exhaustive(broken, R).safe


def test_unchanged_embeddings_certify():
    rng = np.random.default_rng(19)
    E = rng.standard_normal((16, 5))
    R = rng.standard_normal((2, 5))
    facts = np.array([[0,0,1], [2,1,3]])
    cert = RadiusCertificate(E, R, facts, facts, k=16)
    assert cert.verify(E, R).safe


def test_reject_entity_addition_without_rebuild():
    E = np.arange(20).reshape(10,2).astype(float)
    R = np.zeros((2,2))
    facts = np.array([[0,0,1]])
    cert = RadiusCertificate(E, R, facts, facts, k=10)
    with pytest.raises(ValueError, match='universe'):
        cert.verify(np.vstack((E, [0.,0.])), R)


def test_data_audit_and_real_margin_loss():
    kg = synthetic_graph(seed=2, n_entities=35, n_relations=3, n_train=130,
                         n_valid=15, n_test=15)
    audit = audit_splits(kg)
    assert audit['triple_overlap_train_test'] == 0
    E,R,info = train_transe(kg.train, kg.n_entities, kg.n_relations,
                           TrainConfig(dim=8, steps=12, batch_size=24, negatives=2))
    assert np.isfinite(E).all() and np.isfinite(R).all()
    assert 'margin' in info['loss']
