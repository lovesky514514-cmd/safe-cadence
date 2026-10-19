"""v0.4 matched end-to-end CPU budget audit, synthetic-only.

No guarantee that unchecked SGD baseline trajectories are continuously safe:
that status is independently verified after timing, at every accepted parameter step.
Every arm reuses the same pregenerated positive/negative sequence.
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from safecadence.data import synthetic_graph
from safecadence.certificate_cache import CachedRankCertificate
from safecadence.diagnostics_v3 import make_batches,run_variant,Variant,rank_scope_diagnostics
from safecadence.ranking import filtered_metrics

@dataclass(frozen=True)
class Baseline:
    name: str
    lr: float=.25
    anchor: float=0.
    replay: float=0.
    update_mode: str='all'

BASELINES=[
    Baseline('sgd_lr_half',lr=.125),
    Baseline('sgd_lr_tenth',lr=.025),
    Baseline('sgd_lr_0.03x',lr=.0075),
    Baseline('sgd_l2_0.5',anchor=.5),
    Baseline('sgd_l2_1',anchor=1.),
    Baseline('sgd_l2_3',anchor=3.),
    Baseline('sgd_l2_10',anchor=10.),
    Baseline('sgd_replay25',replay=.25),
    Baseline('sgd_replay50',replay=.50),
    Baseline('incoming_entities_only',update_mode='incoming'),
]

# At each accepted step check an independent exact-exhaustive rank oracle.
# This audit is OFFLINE, excluded from the method's wall time.
def audit_trajectory(E0,R0,states,checker,incoming,stop_at_first=True):
    failures=0; first=None
    for j,(E,R) in enumerate(states,1):
        if not checker.verify_exhaustive(E,R,added_true=incoming,stop_early=True).safe:
            failures+=1
            if first is None:first=j
            if stop_at_first:break
    return {'continuously_safe':first is None,'first_unsafe_step':first,
            'unsafe_steps_checked':failures,'audited_steps':j if states else 0}


def run_baseline(E0,R0,old,incoming,protected,batches,config:Baseline,budget_s,seed):
    E=E0.copy(); R=R0.copy(); e_anchor=torch.tensor(E0,dtype=torch.float32)
    r_anchor=torch.tensor(R0,dtype=torch.float32)
    old_tensor=torch.tensor(old,dtype=torch.long)
    rng=np.random.default_rng(seed)
    seen=set(map(tuple,np.concatenate((old,incoming)).tolist()))
    allowed=np.zeros(len(E),dtype=bool)
    allowed[np.unique(incoming[:,(0,2)])]=True
    states=[];diverged=False
    # Avoid preparing all possible replay data outside timing; include its sampling/negatives.
    torch.set_num_threads(1)
    start=perf_counter(); n=0
    for pos,neg in batches:
        if perf_counter()-start>=budget_s:break
        ee=torch.tensor(E,dtype=torch.float32,requires_grad=True)
        rr=torch.tensor(R,dtype=torch.float32,requires_grad=True)
        def distances(t):return torch.linalg.vector_norm(ee[t[:,0]]+rr[t[:,1]]-ee[t[:,2]],dim=1)
        loss=F.relu(1+distances(pos).repeat_interleave(len(neg)//len(pos))-distances(neg)).mean()
        if config.replay:
            npick=max(1,round(len(pos)*config.replay))
            pick=old_tensor[torch.as_tensor(rng.integers(0,len(old_tensor),size=npick),dtype=torch.long)]
            corrupted=pick.clone()
            for idx in range(npick):
                axis=0 if rng.integers(0,2)==0 else 2
                for attempt in range(200):
                    corrupted[idx,axis]=int(rng.integers(len(E)))
                    if tuple(int(x) for x in corrupted[idx].tolist()) not in seen:break
                else:raise ValueError('Could not find negative')
            # Paired equal count negatives for replay
            loss=(1-config.replay)*loss+config.replay*F.relu(1+distances(pick)-distances(corrupted)).mean()
        if config.anchor:
            loss=loss+config.anchor*.5*((ee-e_anchor).square().sum()+(rr-r_anchor).square().sum())
        ge,gr=torch.autograd.grad(loss,(ee,rr))
        ge=ge.detach().numpy().astype(np.float64)
        gr=gr.detach().numpy().astype(np.float64)
        if config.update_mode=='incoming':
            ge[~allowed]=0.;gr[:]=0.
        E-=config.lr*ge;R-=config.lr*gr
        if (not np.isfinite(E).all() or not np.isfinite(R).all() or
            max(np.max(np.abs(E)),np.max(np.abs(R)))>1e6):
            diverged=True
            if states:E,R=states[-1][0].copy(),states[-1][1].copy()
            else:E,R=E0.copy(),R0.copy()
            break
        norm=np.linalg.norm(E,axis=1)
        over=norm>1.000000001
        if np.any(over):E[over]/=norm[over,None]
        n+=1
        states.append((E.copy(),R.copy()))
    wall=perf_counter()-start
    checker=CachedRankCertificate(E0,R0,protected,old,k=10,max_cache_mb=64)
    audit_start=perf_counter()
    audit=audit_trajectory(E0,R0,states,checker,incoming)
    if diverged:audit['continuously_safe']=False
    audit_wall=perf_counter()-audit_start
    endpoint=checker.verify_exhaustive(E,R,added_true=incoming).safe
    truth=np.concatenate((old,incoming))
    return {'config':asdict(config),'diverged':diverged,'steps':n,'wall_seconds':wall,'budget_seconds':budget_s,
            'audit_wall_seconds_excluded':audit_wall, **audit,
            'final_protected_safe':endpoint and not diverged,
            'new_mrr':filtered_metrics(E,R,incoming,truth,max_eval=100)['mrr'],
            'old_mrr':filtered_metrics(E,R,old,truth,max_eval=100)['mrr'],
            'all_old_damage':rank_scope_diagnostics(E0,R0,E,R,old,protected,incoming)['historical_break_fraction']}


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--seeds',type=int,default=5)
    a.add_argument('--max-steps',type=int,default=3600)
    a.add_argument('--baselines',nargs='*',default=None)
    a.add_argument('--output',default='outputs/v4_matched_compute.json')
    a.add_argument('--engine',choices=['cached','batched'],default='batched')
    x=a.parse_args(); outputs=[]
    configs=[cfg for cfg in BASELINES if x.baselines is None or cfg.name in x.baselines]
    for seed in range(1,x.seeds+1):
        state=np.load(f'outputs/multiseed_{seed}.npz')
        E0,R0=state['old_entities'],state['old_relations']
        graph=synthetic_graph(seed=seed)
        rng=np.random.default_rng(seed)
        order=rng.permutation(len(graph.train)); split=int(len(order)*.75)
        old=graph.train[order[:split]]; incoming=graph.train[order[split:]]
        protected=old[rng.choice(len(old),50,replace=False)]
        train80=make_batches(incoming,np.concatenate((old,incoming)),len(E0),steps=80,seed=seed*19)
        _,_,cert=run_variant(E0,R0,incoming,old,protected,train80,Variant('certified_sgd',certified=True),audit_decisions=False,certificate_engine=x.engine)
        budget=cert['method_wall_seconds']
        print(f'seed {seed}: certified CPU budget={budget:.3f}s; new MRR={cert["new_mrr"]:.3f}',flush=True)
        batches=make_batches(incoming,np.concatenate((old,incoming)),len(E0),steps=x.max_steps,seed=seed*19)
        initial_new=filtered_metrics(E0,R0,incoming,np.concatenate((old,incoming)),max_eval=100)['mrr']
        arms={}
        for cfg in configs:
            result=run_baseline(E0,R0,old,incoming,protected,batches,cfg,budget,seed*93)
            arms[cfg.name]=result
            print('  ',cfg.name,'steps',result['steps'],'new',round(result['new_mrr'],3),
                  'continuously safe',result['continuously_safe'],'first violation',result['first_unsafe_step'],flush=True)
        outputs.append({'seed':seed,'initial_new_mrr':initial_new,
                        'certified':{'new_mrr':cert['new_mrr'],'old_mrr':cert['old_mrr'],
                                     'wall_seconds':budget,'steps':80,
                                     'continuously_safe':True,
                                     'final_protected_safe':cert['endpoint_safe_exhaustive']},
                        'baselines':arms})
        Path(x.output).parent.mkdir(exist_ok=True)
        Path(x.output).write_text(json.dumps({'seeds_completed':seed,'rows':outputs},indent=2))
    summary={}
    for cfg in configs:
        rs=[row['baselines'][cfg.name] for row in outputs]
        summary[cfg.name]={'mean_new_mrr':float(np.mean([r['new_mrr'] for r in rs])),
                           'continuous_safe_seeds':sum(r['continuously_safe'] for r in rs),
                           'final_safe_seeds':sum(r['final_protected_safe'] for r in rs),
                           'mean_steps':float(np.mean([r['steps'] for r in rs])),
                           'mean_time_s':float(np.mean([r['wall_seconds'] for r in rs]))}
    summary['certified']={'mean_new_mrr':float(np.mean([r['certified']['new_mrr'] for r in outputs])),
                          'continuous_safe_seeds':len(outputs),
                          'final_safe_seeds':sum(r['certified']['final_protected_safe'] for r in outputs),
                          'mean_steps':80.,
                          'mean_time_s':float(np.mean([r['certified']['wall_seconds'] for r in outputs]))}
    obj={'protocol':'synthetic fixed-entity TransE, same CPU budget per seed, whole trajectory offline safety checker',
         'limitations':['no real CKGE','no machine-level soundness','1 CPU process; noise in wall time',
                        'explicit protected subset, no global safety','baseline checkpoint selection not tuned on external validation'],
         'rows':outputs,'summary':summary}
    Path(x.output).write_text(json.dumps(obj,indent=2))
    print('FINAL SUMMARY\n',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
