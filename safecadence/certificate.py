"""Exact Top-k preservation certificate with a sound KD-tree radius filter.

For TransE distance d(q,e), if the old correct distance is d_a and updates are
bounded by delta_q=||q'-q||, delta_a=||e'_a-e_a||,
D=max_c ||e'_c-e_c||, then any competing entity with old distance
  d(q,e_c)>d_a + 2*delta_q + delta_a + D
CANNOT be at least as close as the answer after the update.
The radius is conservative; included entities are scored EXACTLY after updating.
The proof applies to fixed candidate universe, all protected queries, L2 TransE.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from .ranking import exact_rank, query_and_answer, TrueFacts


@dataclass
class CertificateResult:
    safe: bool
    checked_queries: int
    inspected_candidates: int
    full_candidates: int
    kdtree_seconds: float
    checking_seconds: float
    failed_query: tuple | None = None
    reason: str = 'certified'


class RadiusCertificate:
    def __init__(self, old_entities, old_relations, protected_triples, known_true, k=10,
                 sides=('head', 'tail')):
        self.E0 = np.asarray(old_entities, dtype=np.float64).copy()
        self.R0 = np.asarray(old_relations, dtype=np.float64).copy()
        self.truth = TrueFacts(known_true)
        self.k = int(k)
        if self.k < 1:
            raise ValueError('k must be positive')
        self.protected = []
        self.not_initially_protected = 0
        self.build_start = perf_counter()
        self.tree = cKDTree(self.E0)
        self.build_seconds = perf_counter() - self.build_start
        for triple in protected_triples:
            for side in sides:
                # Only claims preservation of originally satisfactory queries.
                rank = exact_rank(self.E0, self.R0, triple, side, self.truth)
                if rank <= self.k:
                    self.protected.append((tuple(map(int, triple)), side))
                else:
                    self.not_initially_protected += 1

    def verify(self, new_entities, new_relations, exhaustive=False, stop_early=True):
        t0 = perf_counter()
        E1 = np.asarray(new_entities, dtype=np.float64)
        R1 = np.asarray(new_relations, dtype=np.float64)
        if E1.shape != self.E0.shape or R1.shape != self.R0.shape:
            raise ValueError('Candidate universe changed; rebuild certificate for enlarged universe')
        if not (np.isfinite(E1).all() and np.isfinite(R1).all()):
            raise ValueError('Embedding update contains nonfinite values')
        max_delta = float(np.linalg.norm(E1-self.E0, axis=1).max())
        checked = inspected = full = 0
        query_ball_time = 0.0
        failure = None
        for triple, side in self.protected:
            q0, a = query_and_answer(self.E0, self.R0, triple, side)
            q1, _ = query_and_answer(E1, R1, triple, side)
            da = float(np.linalg.norm(q0-self.E0[a]))
            dq = float(np.linalg.norm(q1-q0))
            d_answer = float(np.linalg.norm(q1-E1[a]))
            delta_a = float(np.linalg.norm(E1[a]-self.E0[a]))
            # Small positive slack on KD-tree radius avoids boundary rounding misses.
            radius = float(np.nextafter(da + 2*dq + delta_a + max_delta + 1e-11, np.inf))
            if exhaustive:
                ids = np.arange(len(E1), dtype=np.int64)
            else:
                b0 = perf_counter()
                ids = np.asarray(self.tree.query_ball_point(q0, radius), dtype=np.int64)
                query_ball_time += perf_counter() - b0
            mask = (ids != a)
            excluded = self.truth.ignored(triple, side)
            if excluded:
                mask &= ~np.isin(ids, list(excluded))
            ids = ids[mask]
            # Allow other true facts to rank ahead only if they are not filtered.
            competing_distances = np.linalg.norm(E1[ids]-q1, axis=1)
            rank = 1 + int(np.count_nonzero(competing_distances <= d_answer))
            checked += 1
            inspected += len(ids)
            full += len(E1)-1-len(excluded)
            if rank > self.k:
                failure = (triple, side, rank)
                if stop_early:
                    break
        return CertificateResult(failure is None, checked, inspected, full,
                                 query_ball_time, perf_counter()-t0, failure,
                                 'rank_violation' if failure else 'certified')

    def verify_exhaustive(self, E1, R1, stop_early=True):
        return self.verify(E1, R1, exhaustive=True, stop_early=stop_early)
