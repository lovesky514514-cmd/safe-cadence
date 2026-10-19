"""Real-source pseudo-stream sparse entity-only optimizer: gate engine A/B.

Same proposals, same pretraining, engine only differs in verifier backend.
No canonical CKGE or end-to-end scalable learning claim.
"""
import argparse,json
from pathlib import Path
import numpy as np
from safecadence.real_graph_v5 import countryinfo_kg
from safecadence.transe import TrainConfig,train_transe
from safecadence.ranking import TrueFacts,exact_rank
from safecadence.diagnostics_v3 import make_batches
from safecadence.sparse_optimizer_v5 import run_certified_sparse_entity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--output',default='outputs/v5_sparse_country.json')
    a=p.parse_args()
    results=[]
    for seed in range(1,a.seeds+1):
        kg,meta=countryinfo_kg(seed)
        rng=np.random.default_rng(seed+110)
        x=rng.permutation(len(kg.train));old=kg.train[x[:int(len(x)*.7)]];incoming=kg.train[x[int(len(x)*.7):]]
        E0,R0,_=train_transe(old,kg.n_entities,kg.n_relations,TrainConfig(dim=24,steps=170,batch_size=96,negatives=2,seed=seed,lr=.035))
        known=np.concatenate((old,incoming));truth=TrueFacts(known)
        eligible=np.asarray([tr for tr in old if any(exact_rank(E0,R0,tr,side,truth)<=10 for side in ('tail','head'))],dtype=np.int64)
        protected=eligible[rng.choice(len(eligible),min(50,len(eligible)),replace=False)]
        batches=make_batches(incoming,known,len(E0),steps=a.steps,batch_size=24,negatives=2,seed=seed*190)
        runs=[]
        for backend in ('batched','overlay'):
            x=run_certified_sparse_entity(E0,R0,old,incoming,protected,batches,engine=backend)
            runs.append(x)
        assert np.allclose(runs[0].E,runs[1].E,atol=1e-10)
        assert runs[0].stats['endpoint_safe']==runs[1].stats['endpoint_safe']
        row={'seed':seed,'data':meta,'dense':runs[0].stats,'sparse_overlay':runs[1].stats,'matched_embeddings':True}
        results.append(row)
        print(seed,'N',len(E0),'modified',runs[1].stats['cumulative_support'],'dense_seconds',round(runs[0].stats['method_wall_seconds'],3),'overlay_seconds',round(runs[1].stats['method_wall_seconds'],3),flush=True)
    path=Path(a.output);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
