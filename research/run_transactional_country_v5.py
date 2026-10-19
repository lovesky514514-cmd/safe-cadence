"""Hypothesis falsification: safe public commits without per-SGD-step gates."""
import argparse,json
from pathlib import Path
import numpy as np
from safecadence.real_graph_v5 import countryinfo_kg
from safecadence.transe import TrainConfig,train_transe
from safecadence.ranking import TrueFacts,exact_rank
from safecadence.diagnostics_v3 import make_batches,run_variant,Variant
from safecadence.transactional_v5 import transactional_updates


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--output',default='outputs/v5_country_transactional.json')
    a=parser.parse_args();results=[]
    for seed in range(1,a.seeds+1):
        kg,meta=countryinfo_kg(seed)
        rng=np.random.default_rng(seed+110)
        x=rng.permutation(len(kg.train));old=kg.train[x[:int(len(x)*.7)]];inc=kg.train[x[int(len(x)*.7):]]
        E0,R0,_=train_transe(old,kg.n_entities,kg.n_relations,TrainConfig(dim=24,steps=170,batch_size=96,negatives=2,seed=seed,lr=.035))
        known=np.concatenate((old,inc));truth=TrueFacts(known)
        eligible=np.asarray([tr for tr in old if any(exact_rank(E0,R0,tr,side,truth)<=10 for side in ('tail','head'))],dtype=np.int64)
        prot=eligible[rng.choice(len(eligible),min(50,len(eligible)),replace=False)]
        batches=make_batches(inc,known,len(E0),steps=90,batch_size=24,negatives=2,seed=seed*190)
        _,_,online=run_variant(E0,R0,inc,old,prot,batches,Variant('certified_sgd',certified=True),base_lr=.45,audit_decisions=False,certificate_engine='batched')
        transactional=[]
        for every in (5,10,20,45,90):
            _,_,z=transactional_updates(E0,R0,old,inc,prot,batches,commit_every=every)
            transactional.append(z)
        print(seed,'online',round(online['new_mrr'],3),'txn',[(z['commit_every'],round(z['new_mrr'],3),z['gate_calls']) for z in transactional],flush=True)
        results.append({'seed':seed,'online_certified':online,'transactional':transactional,'data':meta})
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
