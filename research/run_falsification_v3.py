"""Run 5-seed matched-proposal simple-baseline falsification study."""
import argparse
import json
from pathlib import Path
from statistics import mean,stdev
import numpy as np
from safecadence.data import synthetic_graph
from safecadence.diagnostics_v3 import variants,make_batches,run_variant,rank_scope_diagnostics
from safecadence.ranking import filtered_metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--steps',type=int,default=80)
    ap.add_argument('--no-audit-decisions',action='store_true')
    ap.add_argument('--output',default='outputs/v3_falsification.json')
    args=ap.parse_args()
    rows=[]
    for seed in range(1,args.seeds+1):
        state=np.load(f'outputs/multiseed_{seed}.npz')
        E0,R0=state['old_entities'],state['old_relations']
        graph=synthetic_graph(seed=seed)
        rng=np.random.default_rng(seed)
        order=rng.permutation(len(graph.train)); split=int(len(order)*.75)
        old=graph.train[order[:split]]; incoming=graph.train[order[split:]]
        protected=old[rng.choice(len(old),50,replace=False)]
        batches=make_batches(incoming,np.concatenate((old,incoming)),len(E0),
                             steps=args.steps,seed=seed*19)
        seed_rows={}
        for var in variants():
            E,R,metrics=run_variant(E0,R0,incoming,old,protected,batches,var,
                                    audit_decisions=not args.no_audit_decisions)
            metrics['all_old_rank_scope']=rank_scope_diagnostics(E0,R0,E,R,old,protected,incoming)
            metrics['test_mrr']=filtered_metrics(E,R,graph.test,graph.all_true,max_eval=80)['mrr']
            seed_rows[var.name]=metrics
        print(f'seed={seed} '+ ' '.join(f'{k}:MRR={v["new_mrr"]:.3f},safe={int(v["endpoint_safe_exhaustive"])}' for k,v in seed_rows.items()),flush=True)
        rows.append({'seed':seed,'methods':seed_rows})
    aggr={}
    for var in variants():
        results=[r['methods'][var.name] for r in rows]
        fields=['new_mrr','old_mrr','test_mrr','method_wall_seconds','gradient_seconds',
                'certificate_seconds','index_build_seconds','audit_exhaustive_seconds',
                'certificate_calls','independently_audited_gate_calls','candidate_fraction_scored','unsafe_intermediate_steps','accepted_steps',
                'rejected_steps','rescued_steps','first_alpha_unsafe',
                'all_old_rank_scope']
        summary={}
        for f in fields:
            if f=='all_old_rank_scope':
                vals=[r[f]['historical_break_fraction'] for r in results]
                f='all_old_break_fraction'
            else:vals=[r[f] for r in results]
            vals=[float(x) for x in vals if x is not None]
            if vals:summary[f]={'mean':round(mean(vals),7),'sd':round(stdev(vals),7) if len(vals)>1 else None}
        summary['safe_endpoints']=sum(r['endpoint_safe_exhaustive'] for r in results)
        summary['false_accepts']=sum(r['gate_false_accepts'] for r in results)
        summary['false_rejects']=sum(r['gate_false_rejects'] for r in results)
        aggr[var.name]=summary
    payload={'protocol':'synthetic_fixed_entity_TransE_5seeds_matched_proposals','note':'NOT official continual KG benchmark; same 80 sampled minibatches and negatives within seed; computation uses CPU; strict Top-10 for sampled initially Top-10 old queries; certification does not imply protection for every old query.', 'n_seeds':args.seeds,'steps_per_arm':args.steps,'rows':rows,'aggregate':aggr}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    print('SAVED',args.output)
    return payload

if __name__=='__main__':main()
