"""Artificial scaling test for a fixed-entity L2 TransE certificate, not CKGE quality."""
import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from safecadence.certificate import RadiusCertificate


def benchmark(n, seed=3, d=8, q_count=60, trials=3, drift=0.003):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n,d))
    R = np.zeros((1,d))
    R[0,0] = 1.5
    rng_ix = rng.choice(n, size=q_count, replace=False)
    # True tails are the nearest neighbours under the planted old TransE geometry.
    old_tree = cKDTree(E)
    _, ans = old_tree.query(E[rng_ix]+R[0])
    triples = np.c_[rng_ix, np.zeros(q_count, dtype=int), ans].astype(int)
    build_start = perf_counter()
    cert = RadiusCertificate(E,R,triples,triples,k=1,sides=('tail',))
    build_secs = perf_counter()-build_start
    rows=[]
    for _ in range(trials):
        E1 = E.copy()
        change = rng.choice(n, size=min(10,n), replace=False)
        E1[change] += rng.normal(size=(len(change),d))*drift
        R1 = R.copy()
        R1 += rng.normal(size=R.shape)*drift
        fast = cert.verify(E1,R1,stop_early=False)
        brute = cert.verify_exhaustive(E1,R1,stop_early=False)
        assert fast.safe == brute.safe
        rows.append({'fast': fast.checking_seconds,'full': brute.checking_seconds,
                     'fraction': fast.inspected_candidates/max(fast.full_candidates,1)})
    return {'entities': n, 'dim':d, 'drift':drift,'queries': len(cert.protected),
            'index_and_rank_cache_build_seconds':round(build_secs,6),
            'fast_mean_seconds':round(mean(r['fast'] for r in rows),6),
            'full_mean_seconds':round(mean(r['full'] for r in rows),6),
            'candidate_fraction':round(mean(r['fraction'] for r in rows),6),
            'speedup_excluding_build':round(mean(r['full'] for r in rows)/mean(r['fast'] for r in rows),3),
            'speedup_including_build_one_update':round(mean(r['full'] for r in rows)/(build_secs+mean(r['fast'] for r in rows)),3)}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+', default=[120,1000,5000,20000])
    p.add_argument('--dim', type=int, default=8)
    p.add_argument('--drift', type=float, default=0.003)
    p.add_argument('--out', type=Path, default=Path('outputs/scaling.json'))
    args=p.parse_args()
    results = [benchmark(n, d=args.dim, drift=args.drift) for n in args.sizes]
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps({'status':'artificial_index_benchmark_not_ckge', 'results':results}, indent=2))
    print(json.dumps(results, indent=2))
