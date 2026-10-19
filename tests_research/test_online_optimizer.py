"""Compare certified online SGD and uncertified SGD on stored base checkpoints."""
import json
from pathlib import Path
import numpy as np
from safecadence.data import synthetic_graph
from safecadence.ranking import filtered_metrics
from safecadence.safe_optimizer import certified_minibatch_sgd
from safecadence.certificate_cache import CachedRankCertificate

def main():
    out=[]
    for seed in range(1,6):
        ck=np.load(f'outputs/multiseed_{seed}.npz')
        E,R=ck['old_entities'],ck['old_relations']
        kg=synthetic_graph(seed=seed)
        rng=np.random.default_rng(seed)
        perm=rng.permutation(len(kg.train)); split=int(len(perm)*.75)
        old=kg.train[perm[:split]]; new=kg.train[perm[split:]]
        sample=old[rng.choice(len(old),50,replace=False)]
        rows={'seed':seed}
        for name,certify in (('blind_sgd',False),('online_certified_sgd',True)):
            ee,rr,meta=certified_minibatch_sgd(E,R,new,sample,old,
                                               steps=80,batch_size=32,negatives=2,lr=.25,
                                               k=10,seed=seed*19,certify=certify)
            ch=CachedRankCertificate(E,R,sample,old,k=10)
            actual_safe=ch.verify_exhaustive(ee,rr,added_true=new).safe
            rows[name]={'new_mrr':filtered_metrics(ee,rr,new,kg.all_true,max_eval=100)['mrr'],
                        'old_mrr':filtered_metrics(ee,rr,old,kg.all_true,max_eval=100)['mrr'],
                        'historical_rank_safe':actual_safe,'meta':meta}
        print('seed',seed,'blind',rows['blind_sgd']['new_mrr'],'safe',rows['online_certified_sgd']['new_mrr'],
              'accepted',rows['online_certified_sgd']['meta']['accepted_steps'],'of 80',flush=True)
        out.append(rows)
    Path('outputs/online_sgd.json').write_text(json.dumps({'protocol':'synthetic_sgd_online_safety','rows':out},indent=2),encoding='utf-8')

if __name__=='__main__':main()
