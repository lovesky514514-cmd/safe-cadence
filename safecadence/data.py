"""Data generation and text-triple loading, with explicit split audits."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class KnowledgeGraph:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    n_entities: int
    n_relations: int
    source: str

    @property
    def all_true(self):
        return np.concatenate([self.train, self.valid, self.test], axis=0)


def synthetic_graph(seed=7, n_entities=120, n_relations=4, n_train=400,
                    n_valid=60, n_test=80):
    """Construct a learnable fixed-entity relational toy graph, not a benchmark."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n_entities, 5)).astype(np.float64)
    translations = rng.normal(0.0, 0.65, size=(n_relations, 5))
    edges = []
    for h in range(n_entities):
        for r in range(n_relations):
            distance = np.linalg.norm(latent - (latent[h] + translations[r]), axis=1)
            distance[h] = np.inf
            # Each (head, relation) receives up to two valid facts.
            for t in np.argsort(distance)[:2]:
                edges.append((h, r, int(t)))
    rng.shuffle(edges)
    total = n_train + n_valid + n_test
    if total > len(edges):
        raise ValueError('Too many requested triples')
    arr = np.asarray(edges[:total], dtype=np.int64)
    return KnowledgeGraph(arr[:n_train], arr[n_train:n_train+n_valid],
                          arr[n_train+n_valid:], n_entities, n_relations,
                          'synthetic_latent_translation')


def load_tsv_dataset(root):
    """Load WN18RR-style h, r, t string TSVs; no random split or transductive filtering."""
    root = Path(root)
    files = [root / f'{part}.txt' for part in ('train', 'valid', 'test')]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('Expected train.txt, valid.txt, test.txt in ' + str(root))
    raw = []
    for path in files:
        examples = []
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            tokens = line.split()
            if len(tokens) != 3:
                raise ValueError(f'{path}:{line_no}: expected 3 tokens')
            examples.append(tokens)
        raw.append(examples)
    # Include evaluation-only entities in candidate universe (transductive protocol).
    entities = sorted(set(e for split in raw for h, _, t in split for e in (h, t)))
    relations = sorted(set(r for split in raw for _, r, _ in split))
    emap = {e: idx for idx, e in enumerate(entities)}
    rmap = {r: idx for idx, r in enumerate(relations)}
    def encode(rows):
        return np.asarray([(emap[h], rmap[r], emap[t]) for h, r, t in rows], dtype=np.int64).reshape(-1, 3)
    return KnowledgeGraph(*(encode(split) for split in raw), len(entities),
                          len(relations), str(root))


def audit_splits(kg):
    s = [set(map(tuple, x.tolist())) for x in (kg.train, kg.valid, kg.test)]
    train_e = set(kg.train[:, 0].tolist() + kg.train[:, 2].tolist())
    def unseen(split):
        return len({int(e) for h, _, t in split for e in (h, t)} - train_e)
    return {
        'source': kg.source,
        'sizes': dict(zip(('train', 'valid', 'test'), [len(x) for x in s])),
        'n_entities': kg.n_entities,
        'n_relations': kg.n_relations,
        'triple_overlap_train_valid': len(s[0] & s[1]),
        'triple_overlap_train_test': len(s[0] & s[2]),
        'triple_overlap_valid_test': len(s[1] & s[2]),
        'unseen_entities_valid': unseen(kg.valid),
        'unseen_entities_test': unseen(kg.test),
    }
