"""Sweep historical triple protection budget and measure continuous-safety/plasticity cost."""
import argparse,json
from pathlib import Path
import numpy as np
from safecadence.data import synthetic_graph
from safecadence.diagnostics_v3 import run_variant,Variant,make_batches,rank_scope_diagnostics
from safecadence.certificate_v2 import SparseRadiusCertificate
from safecadence.ranking import TrueFacts,exact_rank

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=5)
    p.add_argument('--fractions',nargs='*',type=float,default=[.1,.25,.5,1.])
    p.add_argument('--output',default='outputs/v4_protection_sweep.json')
    a=p.parse_args();rows=[]
    for seed in range(1,a.seeds+1):
        st=np.load(f'outputs/multiseed_{seed}.npz')
        E0,R0=st['old_entities'],st['old_relations']
        kg=synthetic_graph(seed=seed);rng=np.random.default_rng(seed)
        order=rng.permutation(len(kg.train));old=kg.train[order[:300]];incoming=kg.train[order[300:]]
        randomized=rng.permutation(len(old))
        batches=make_batches(incoming,np.concatenate((old,incoming)),len(E0),steps=80,seed=seed*19)
        for f in a.fractions:
            chosen=old[randomized[:max(1,int(round(f*len(old))))]]
            E,R,stats=run_variant(E0,R0,incoming,old,chosen,batches,Variant('certified_sgd',certified=True),
                                  audit_decisions=False,certificate_engine='batched')
            dmg=rank_scope_diagnostics(E0,R0,E,R,old,chosen,incoming)
            # Match protected-query initial eligibility: OLD facts at t=0, merged facts after the update.
            pre_truth=TrueFacts(old);after_truth=TrueFacts(np.concatenate((old,incoming)))
            matched_eligible=matched_broken=0
            for triple in old:
                for side in ('head','tail'):
                    if exact_rank(E0,R0,triple,side,pre_truth)<=10:
                        matched_eligible+=1
                        matched_broken+=int(exact_rank(E,R,triple,side,after_truth)>10)
            row={'seed':seed,'protected_fraction_of_old_triples':f,
                 'eligible_protected_queries':stats['protected_initial_topk_queries'],
                 'new_mrr':stats['new_mrr'],'method_wall_seconds':stats['method_wall_seconds'],
                 'certificate_seconds':stats['certificate_seconds'],
                 'certificate_calls':stats['certificate_calls'],
                 'steps_accepted':stats['accepted_steps'],
                 'steps_rejected':stats['rejected_steps'],
                 'all_old_damage':dmg['historical_break_fraction'],
                 'all_old_damage_matched_initial_truth':matched_broken/matched_eligible,
                 'endpoint_safe_exhaustive':stats['endpoint_safe_exhaustive']}
            rows.append(row)
            print(row,flush=True)
            Path(a.output).parent.mkdir(exist_ok=True)
            Path(a.output).write_text(json.dumps({'rows':rows},indent=2))
    agg={}
    for f in a.fractions:
        subset=[r for r in rows if r['protected_fraction_of_old_triples']==f]
        agg[str(f)]={k:float(np.mean([r[k] for r in subset])) for k in
            ['eligible_protected_queries','new_mrr','method_wall_seconds','certificate_seconds','certificate_calls','steps_accepted','steps_rejected','all_old_damage','all_old_damage_matched_initial_truth']}
    obj={'setting':'synthetic 120-entity, 300 old triples, 100 incoming, 80 steps per run',
         'fractions':'fractions sampled from historical triples, not protected-query eligibility fraction',
         'rows':rows,'summary':agg}
    Path(a.output).write_text(json.dumps(obj,indent=2))
    print('FINAL',json.dumps(agg,indent=2))
if __name__=='__main__':main()
