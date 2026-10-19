"""Equal CPU-wall budget on real-source, PSEUDO-temporal country graph.

Only simple baselines; NO TADG/LKGE/IncDE reproduction. Intermediate states
are audited OFFLINE, excluded from method walltime. The model training
budget excludes pretraining (identical shared initialization per seed).
"""
import argparse,json
from pathlib import Path
import numpy as np
from safecadence.real_graph_v5 import countryinfo_kg
from safecadence.transe import TrainConfig,train_transe
from safecadence.diagnostics_v3 import make_batches,run_variant,Variant
from safecadence.ranking import TrueFacts,exact_rank
from run_matched_compute_v4 import run_baseline,Baseline

CONFIGS=[Baseline('SGD_0.5x',lr=.225),Baseline('SGD_0.1x',lr=.045),
         Baseline('SGD_0.03x',lr=.0135),Baseline('L2_strong',lr=.45,anchor=3.),
         Baseline('Replay_25pct',lr=.45,replay=.25),Baseline('Incoming_only',lr=.45,update_mode='incoming')]

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);p.add_argument('--start-seed',type=int,default=1);p.add_argument('--output',default='outputs/v5_country_matched.json')
    a=p.parse_args();results=[]
    for seed in range(a.start_seed,a.start_seed+a.seeds):
        kg,meta=countryinfo_kg(seed)
        rng=np.random.default_rng(seed+110)
        x=rng.permutation(len(kg.train));old=kg.train[x[:int(len(x)*.7)]];incoming=kg.train[x[int(len(x)*.7):]]
        E0,R0,_=train_transe(old,kg.n_entities,kg.n_relations,TrainConfig(dim=24,steps=170,batch_size=96,negatives=2,seed=seed,lr=.035))
        known=np.concatenate((old,incoming));truth=TrueFacts(known)
        eligible=np.asarray([tr for tr in old if any(exact_rank(E0,R0,tr,side,truth)<=10 for side in ('tail','head'))],dtype=np.int64)
        protected=eligible[rng.choice(len(eligible),min(50,len(eligible)),replace=False)]
        train80=make_batches(incoming,known,len(E0),steps=90,batch_size=24,negatives=2,seed=seed*190)
        _,_,cert=run_variant(E0,R0,incoming,old,protected,train80,Variant('certified_sgd',certified=True),
                             base_lr=.45,audit_decisions=False,certificate_engine='batched')
        budget=cert['method_wall_seconds']
        all_batches=make_batches(incoming,known,len(E0),steps=1100,batch_size=24,negatives=2,seed=seed*190)
        vals=[]
        for c in CONFIGS:
            z=run_baseline(E0,R0,old,incoming,protected,all_batches,c,budget,seed*93)
            vals.append(z)
            print(seed,c.name,round(z['new_mrr'],3),'safe',z['continuously_safe'],'steps',z['steps'],flush=True)
        results.append({'seed':seed,'source':meta,'budget_seconds':budget,'protected_count':len(protected),
                        'certified':cert,'baselines':vals})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
