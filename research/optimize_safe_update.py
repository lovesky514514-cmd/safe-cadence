"""Proof-of-concept candidate search: two independently scaled parameter blocks.
Optimizes incoming TRAIN MRR, never test MRR. Is a feasibility ablation,
not a validated new algorithm. Certified only on selected historical queries.
"""
import json
from pathlib import Path
from statistics import mean
import numpy as np
from safecadence.data import synthetic_graph
from safecadence.ranking import filtered_metrics
from safecadence.certificate_cache import CachedRankCertificate

LEVELS=[0.0,.03125,.0625,.125,.25,.5,.75,1.0]

def main():
    allrows=[]
    for seed in range(1,6):
        w=np.load(f'outputs/multiseed_{seed}.npz')
        E0,R0,E1,R1=(w[k] for k in ['old_entities','old_relations','new_entities','new_relations'])
        kg=synthetic_graph(seed=seed)
        rng=np.random.default_rng(seed)
        perm=rng.permutation(len(kg.train));split=int(len(perm)*.75)
        old,incoming=kg.train[perm[:split]],kg.train[perm[split:]]
        sample=old[rng.choice(len(old),50,replace=False)]
        c=CachedRankCertificate(E0,R0,sample,np.concatenate((old,incoming)),k=10)
        ed,rd=E1-E0,R1-R0
        attempts=[]
        for ae in LEVELS:
            EE=E0+ae*ed
            for ar in LEVELS:
                RR=R0+ar*rd
                cert=c.verify(EE,RR,stop_early=True)
                if cert.safe:
                    m=filtered_metrics(EE,RR,incoming,kg.all_true,max_eval=100)['mrr']
                    attempts.append(dict(entity_scale=ae,relation_scale=ar,new_mrr=m))
        best=max(attempts,key=lambda x:x['new_mrr'])
        baseline_candidates=[x for x in attempts if abs(x['entity_scale']-x['relation_scale'])<1e-10]
        line=max(baseline_candidates,key=lambda x:x['new_mrr'])
        allrows.append(dict(seed=seed,protected=len(c.protected),best=best,one_scale=line,
                            safe_combinations=len(attempts),total_combinations=len(LEVELS)**2,
                            raw_new_mrr=filtered_metrics(E1,R1,incoming,kg.all_true,max_eval=100)['mrr']))
        print('seed',seed,'joint',line,'2d',best,'safe fraction',len(attempts)/64,flush=True)
    summary={'mean_1d_new_mrr':mean(r['one_scale']['new_mrr'] for r in allrows),
             'mean_2d_new_mrr':mean(r['best']['new_mrr'] for r in allrows),
             'mean_raw_new_mrr':mean(r['raw_new_mrr'] for r in allrows),
             'mean_safe_fraction':mean(r['safe_combinations']/r['total_combinations'] for r in allrows)}
    Path('outputs/grid_search.json').write_text(json.dumps({'protocol':'synthetic_train_objective_safety_grid','results':allrows,'summary':summary},indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary,indent=2))

if __name__=='__main__':main()
