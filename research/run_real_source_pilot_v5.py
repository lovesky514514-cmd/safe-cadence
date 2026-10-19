"""Real-source KG smoke with randomly simulated updates (NOT standard CKGE)."""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from safecadence.real_graph_v5 import countryinfo_kg
from safecadence.data import audit_splits
from safecadence.transe import TrainConfig,train_transe
from safecadence.diagnostics_v3 import make_batches,run_variant,Variant
from safecadence.ranking import filtered_metrics,TrueFacts,exact_rank
from safecadence.batched_shortlist_v5 import BatchedShortlistCertificate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--pretrain',type=int,default=170)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--output',default='outputs/v5_country_source.json')
    a=parser.parse_args()
    res=[]
    for seed in range(1,a.seeds+1):
        kg,metadata=countryinfo_kg(seed)
        audit=audit_splits(kg)
        rng=np.random.default_rng(seed+110)
        order=rng.permutation(len(kg.train))
        split=int(len(kg.train)*.70)
        old=kg.train[order[:split]]
        incoming=kg.train[order[split:]]
        known=np.concatenate((old,incoming))
        E0,R0,log=train_transe(old,kg.n_entities,kg.n_relations,TrainConfig(dim=24,steps=a.pretrain,batch_size=96,negatives=2,seed=seed,lr=.035))
        eligible=[]
        known_truth=TrueFacts(known)
        for triple in old:
            if any(exact_rank(E0,R0,triple,side,known_truth)<=10 for side in ('tail','head')):
                eligible.append(triple)
        eligible=np.asarray(eligible,dtype=np.int64).reshape(-1,3)
        if not len(eligible):raise RuntimeError('No initially Top-10 queries; safety evaluation invalid')
        protected=eligible[rng.choice(len(eligible),min(50,len(eligible)),replace=False)]
        batches=make_batches(incoming,known,kg.n_entities,steps=a.steps,batch_size=24,negatives=2,seed=seed*190)
        arms=[Variant('finetune'),Variant('l2_strong',anchor=3.0),Variant('lr_tenth',lr_factor=.1),Variant('certified_sgd_v4',certified=True),Variant('certified_sgd_v5',certified=True)]
        out={'seed':seed,'data':metadata,'split_audit':audit,'initial_old_mrr':filtered_metrics(E0,R0,old,kg.all_true,max_eval=100)['mrr'],
             'initial_new_mrr':filtered_metrics(E0,R0,incoming,kg.all_true,max_eval=100)['mrr'],
             'initial_test_mrr':filtered_metrics(E0,R0,kg.test,kg.all_true,max_eval=100)['mrr'],
             'protected_candidate_triples':len(protected),'results':{}}
        for variant in arms:
            t=perf_counter()
            E1,R1,diag=run_variant(E0,R0,incoming,old,protected,batches,variant,base_lr=.45,
                                  audit_decisions=False,certificate_engine=('batchshort' if variant.name=='certified_sgd_v5' else 'batched'),audit_stride=100000)
            diag['test_mrr']=filtered_metrics(E1,R1,kg.test,kg.all_true,max_eval=100)['mrr']
            diag['wall_observed']=perf_counter()-t
            out['results'][variant.name]=diag
            print(seed,variant.name,'new=',diag['new_mrr'],'old=',diag['old_mrr'],
                  'test=',diag['test_mrr'],'safe=',diag['endpoint_safe_exhaustive'],
                  'time=',round(diag['method_wall_seconds'],3),flush=True)
        res.append(out)
    p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(res,indent=2))

if __name__=='__main__':main()
