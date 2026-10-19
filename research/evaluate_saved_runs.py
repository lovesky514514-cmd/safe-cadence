"""Audits ORIGINAL v0.1 five-seed synthetic run checkpoints (no claim of CKGE SOTA).
Saves independent exact-vs-sorted-index verification and useful learning delta.
"""
import json
from pathlib import Path
from statistics import mean
import numpy as np
from safecadence.data import synthetic_graph
from safecadence.ranking import filtered_metrics
from safecadence.certificate_cache import CachedRankCertificate

ALPHAS=(1.,.5,.25,.125,.0625,.03125,.015625,0.)

def main():
    records=[]
    for seed in range(1,6):
        f=Path(f'outputs/multiseed_{seed}.npz')
        if not f.exists(): raise FileNotFoundError(f)
        ck=np.load(f)
        E0,R0,E1,R1= [ck[k] for k in ('old_entities','old_relations','new_entities','new_relations')]
        kg=synthetic_graph(seed=seed)
        rng=np.random.default_rng(seed)
        perm=rng.permutation(len(kg.train)); split=max(1,int(len(perm)*.75))
        old=kg.train[perm[:split]]; incoming=kg.train[perm[split:]]
        sample=old[rng.choice(len(old),min(len(old),50),replace=False)]
        checker=CachedRankCertificate(E0,R0,sample,np.concatenate((old,incoming)),k=10)
        best=None; attempts=[]
        for alpha in ALPHAS:
            Ea=E0+alpha*(E1-E0);Ra=R0+alpha*(R1-R0)
            fast=checker.verify(Ea,Ra,stop_early=False)
            exact=checker.verify_exhaustive(Ea,Ra,stop_early=False)
            if fast.safe != exact.safe:raise AssertionError('UNSOUND')
            attempts.append({'alpha':alpha,'safe':fast.safe,'fraction':fast.inspected_candidates/max(1,fast.full_candidates),
                             'cache_s':fast.verification_seconds,'exact_s':exact.verification_seconds})
            if fast.safe and best is None:
                best=(alpha,Ea,Ra)
        a,Es,Rs=best
        def measure(E,R,triples):return filtered_metrics(E,R,triples,kg.all_true,max_eval=100)['mrr']
        records.append({'seed':seed,'protected':len(checker.protected),
                        'total_protection_queries':min(len(old),50)*2,
                        'chosen_alpha':a,
                        'initial_old_mrr':measure(E0,R0,old),
                        'raw_old_mrr':measure(E1,R1,old),
                        'safe_old_mrr':measure(Es,Rs,old),
                        'initial_new_mrr':measure(E0,R0,incoming),
                        'raw_new_mrr':measure(E1,R1,incoming),
                        'safe_new_mrr':measure(Es,Rs,incoming),
                        'cache_size_mb':checker.cache_bytes/1048576,
                        'build_s':checker.indexing_seconds,
                        'attempts':attempts})
    out={'type':'synthetic_only_reused_v0.1_weights','records':records,'summary':{
        'mean_alpha':mean(r['chosen_alpha'] for r in records),
        'mean_protection_fraction':mean(r['protected']/r['total_protection_queries'] for r in records),
        'mean_raw_new_mrr':mean(r['raw_new_mrr'] for r in records),
        'mean_safe_new_mrr':mean(r['safe_new_mrr'] for r in records),
        'mean_initial_new_mrr':mean(r['initial_new_mrr'] for r in records),
        'mean_old_safe_mrr':mean(r['safe_old_mrr'] for r in records),
        'all_exact_agree':True}}
    Path('outputs/v2_saved_runs.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['summary'],indent=2))
    for r in records:
        print(r['seed'],'protected',r['protected'],'alpha',r['chosen_alpha'],
              'new',r['initial_new_mrr'],r['raw_new_mrr'],r['safe_new_mrr'])

if __name__=='__main__':main()
