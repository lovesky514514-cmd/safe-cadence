"""Small-update atomic commits, direct online-gate replacement under deployed safety."""
import json
from pathlib import Path
import numpy as np
from safecadence.real_graph_v5 import countryinfo_kg
from safecadence.transe import TrainConfig,train_transe
from safecadence.ranking import TrueFacts,exact_rank
from safecadence.diagnostics_v3 import make_batches
from safecadence.transactional_v5 import transactional_updates
rows=[]
for seed in range(1,6):
    kg,_=countryinfo_kg(seed);rng=np.random.default_rng(seed+110)
    x=rng.permutation(len(kg.train));old=kg.train[x[:int(len(x)*.7)]];inc=kg.train[x[int(len(x)*.7):]]
    E0,R0,_=train_transe(old,kg.n_entities,kg.n_relations,TrainConfig(dim=24,steps=170,batch_size=96,negatives=2,seed=seed,lr=.035))
    known=np.concatenate((old,inc));truth=TrueFacts(known)
    eligible=np.asarray([tr for tr in old if any(exact_rank(E0,R0,tr,side,truth)<=10 for side in ('tail','head'))],dtype=np.int64)
    prot=eligible[rng.choice(len(eligible),min(50,len(eligible)),replace=False)]
    batches=make_batches(inc,known,len(E0),steps=90,batch_size=24,negatives=2,seed=seed*190)
    a=[]
    for period in (1,2,3,4):
        _,_,z=transactional_updates(E0,R0,old,inc,prot,batches,commit_every=period)
        a.append(z)
    rows.append({'seed':seed,'transactions':a})
    print('seed',seed,[(t['commit_every'],round(t['new_mrr'],3),round(t['method_seconds'],3)) for t in a],flush=True)
Path('outputs/v5_country_transaction_micro.json').write_text(json.dumps(rows,indent=2))
