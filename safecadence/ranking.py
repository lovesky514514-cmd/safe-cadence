"""TransE filtered ranking. Correct and rival ties are conservatively handled."""
from collections import defaultdict
import numpy as np


class TrueFacts:
    def __init__(self, triples):
        self.heads = defaultdict(set)
        self.tails = defaultdict(set)
        for h, r, t in triples:
            self.tails[int(h), int(r)].add(int(t))
            self.heads[int(r), int(t)].add(int(h))

    def ignored(self, triple, side):
        h, r, t = map(int, triple)
        true = self.tails[h, r] if side == 'tail' else self.heads[r, t]
        return true - {t if side == 'tail' else h}


def query_and_answer(entities, relations, triple, side):
    h, r, t = map(int, triple)
    if side == 'tail':
        return entities[h] + relations[r], t
    if side == 'head':
        return entities[t] - relations[r], h
    raise ValueError('side must be head or tail')


def exact_rank(entities, relations, triple, side, facts, conservative_ties=True):
    """Rank among ALL candidate entities. By default ties count against answer."""
    q, answer = query_and_answer(entities, relations, triple, side)
    dist = np.linalg.norm(entities - q, axis=1)
    correct = float(dist[answer])
    dist[sorted(facts.ignored(triple, side))] = np.inf
    others = np.ones(len(dist), dtype=bool)
    others[answer] = False
    # conservative rank is an upper-bound under every tie-breaking policy
    if conservative_ties:
        return int(1 + np.count_nonzero((dist <= correct) & others))
    atol = 1e-10
    return float(1 + np.count_nonzero((dist < correct-atol) & others)
                 + 0.5 * np.count_nonzero((abs(dist-correct) <= atol) & others))


def filtered_metrics(entities, relations, triples, all_true, max_eval=None):
    facts = TrueFacts(all_true)
    rows = triples[:max_eval] if max_eval is not None else triples
    if not len(rows):
        return {'mrr': None, 'hits@1': None, 'hits@10': None, 'ranks': 0}
    ranks = [exact_rank(entities, relations, triple, side, facts, False)
             for triple in rows for side in ('head', 'tail')]
    scores = np.asarray(ranks)
    return {'mrr': round(float((1 / scores).mean()), 6),
            'hits@1': round(float((scores <= 1).mean()), 6),
            'hits@10': round(float((scores <= 10).mean()), 6), 'ranks': len(ranks)}
