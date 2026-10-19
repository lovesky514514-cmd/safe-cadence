"""Reproducible incremental TransE baseline, replay, anchoring, EWC-style and cert.
Run on local official WN18RR splits via --data-dir. No SOTA claims.
Data ingestion never sends validation/test facts into training or certificate.
"""
import argparse,json
from pathlib import Path
from time import perf_counter
import numpy as np
from safecadence.data import synthetic_graph,load_tsv_dataset,audit_splits
from safecadence.transe import TrainConfig,train_transe
from safecadence.ewc import estimate_diagonal_importance
from safecadence.ranking import filtered_metrics
from safecadence.certificate_cache import CachedRankCertificate


def evaluate(model,data,truth,limit):
    if not len(data):return None
    return filtered_metrics(model[0],model[1],data,truth,max_eval=limit)['mrr']


def experiment(args):
    kg=load_tsv_dataset(args.data_dir) if args.data_dir else synthetic_graph(seed=args.seed)
    audit=audit_splits(kg)
    if any(audit[x] for x in ['triple_overlap_train_valid','triple_overlap_train_test','triple_overlap_valid_test']):
        raise ValueError('Overlapping splits; abort')
    rng=np.random.default_rng(args.seed)
    perm=rng.permutation(len(kg.train))
    n_initial=max(2,int(len(perm)*args.initial_fraction))
    initial=kg.train[perm[:n_initial]]
    chunks=[x for x in np.array_split(kg.train[perm[n_initial:]],args.updates) if len(x)]
    t=perf_counter()
    E,R,_=train_transe(initial,kg.n_entities,kg.n_relations,
                       TrainConfig(dim=args.dim,steps=args.base_steps,seed=args.seed,
                                   batch_size=args.batch_size,negatives=args.negatives))
    initial_time=perf_counter()-t
    names=('finetune','replay','relation_anchor','replay_anchor','ewc_style','safe_backtracking')
    models={x:(E.copy(),R.copy()) for x in names}
    past=initial.copy()
    histories=initial[rng.choice(len(initial),min(len(initial),args.protect_queries),replace=False)].copy()
    stages=[]
    for stage,batch in enumerate(chunks,1):
        observed=np.concatenate((past,batch))
        records={}
        frozen=models['safe_backtracking']
        checker=CachedRankCertificate(frozen[0],frozen[1],histories,past,k=args.k,
                                      sides=('head','tail'),max_cache_mb=args.cache_mb)
        t=perf_counter()
        imp=estimate_diagonal_importance(past,*models['ewc_style'],n_batches=args.ewc_batches,
                                         batch_size=min(64,args.batch_size),seed=args.seed+stage)
        fisher_time=perf_counter()-t
        replay_old=past[rng.choice(len(past),len(batch),replace=True)]
        replay_set=np.concatenate((batch,replay_old))
        for name in names:
            update_input=replay_set if name in ('replay','replay_anchor') else batch
            cfg=TrainConfig(dim=args.dim,steps=args.update_steps,batch_size=args.batch_size,
                            negatives=args.negatives,lr=args.update_lr,seed=args.seed*17+stage,
                            relation_anchor_strength=args.anchor if name in ('relation_anchor','replay_anchor') else 0.,
                            ewc_strength=args.ewc_strength if name=='ewc_style' else 0.)
            t=perf_counter()
            next_model=train_transe(update_input,kg.n_entities,kg.n_relations,
                                   cfg,init=models[name],forbidden_positive=observed,
                                   ewc=imp if name=='ewc_style' else None)[:2]
            sec=perf_counter()-t
            if name=='safe_backtracking':
                from_e,from_r=frozen
                de,dr=next_model[0]-from_e,next_model[1]-from_r
                chosen=None; trials=[]
                for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,0.):
                    trial=(from_e+alpha*de,from_r+alpha*dr)
                    verdict=checker.verify(*trial,added_true=batch,stop_early=True)
                    trials.append({'alpha':alpha,'safe':verdict.safe,'elapsed':verdict.verification_seconds,
                                   'checked':verdict.checked_queries,'inspected_fraction':verdict.inspected_candidates/max(1,verdict.full_candidates)})
                    if verdict.safe:
                        chosen=alpha;next_model=trial;break
                if chosen is None: raise AssertionError('Zero update not certified: audit the truth filter')
                sec+=sum(t['elapsed'] for t in trials)+checker.indexing_seconds
                records[name+'_certificate']={'protected':len(checker.protected),
                    'eligible_pool':len(histories)*2,'alpha':chosen,'search':trials,
                    'cache_mb':checker.cache_bytes/1048576,
                    'cache_build_seconds':checker.indexing_seconds}
                # Independent audit: certificate must agree with exact exhaustive.
                assert checker.verify_exhaustive(*next_model,added_true=batch).safe
            models[name]=next_model
            records[name]={'old_mrr':evaluate(next_model,past,kg.all_true,args.max_eval),
                           'new_mrr':evaluate(next_model,batch,kg.all_true,args.max_eval),
                           'test_mrr':evaluate(next_model,kg.test,kg.all_true,args.max_eval),
                           'elapsed_seconds':sec+ (fisher_time if name=='ewc_style' else 0.)}
        stages.append({'stage':stage,'old_facts':len(past),'incoming_facts':len(batch),
                       'certification':records.pop('safe_backtracking_certificate'),
                       'algorithms':records})
        # The next stage additionally attempts to protect newly learned facts.
        new_protection=batch[rng.choice(len(batch),min(args.new_protect_per_stage,len(batch)),replace=False)]
        histories=np.concatenate((histories,new_protection))
        past=observed
        print('stage',stage,'facts',len(past), 'safe_alpha',stages[-1]['certification']['alpha'],
              'new MRR safe',round(records['safe_backtracking']['new_mrr'],4),
              'unconstrained',round(records['finetune']['new_mrr'],4),flush=True)
    return {'protocol':'simulated_order_from_original_training_split_NOT_temporal_benchmark',
            'data_audit':audit,'seed':args.seed,'initial_train_seconds':initial_time,
            'configuration':{**vars(args),'data_dir':str(args.data_dir) if args.data_dir else None, 'out':str(args.out)},
            'first_train_facts':len(initial),'stages':stages,
            'warnings':['Incremental order is random rather than real chronological snapshots',
                        'EWC-style uses squared gradients, not exact Fisher information',
                        'Safety covers selected old top-k queries only, not full graph',
                        'Uniform steps and batch size do not imply identical walltime or data exposure']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=None)
    p.add_argument('--out',type=Path,default=Path('outputs/v2_stream.json'))
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--initial-fraction',type=float,default=.65)
    p.add_argument('--updates',type=int,default=3)
    p.add_argument('--dim',type=int,default=24)
    p.add_argument('--base-steps',type=int,default=240)
    p.add_argument('--update-steps',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--negatives',type=int,default=3)
    p.add_argument('--update-lr',type=float,default=.007)
    p.add_argument('--anchor',type=float,default=.1)
    p.add_argument('--ewc-strength',type=float,default=.4)
    p.add_argument('--ewc-batches',type=int,default=12)
    p.add_argument('--protect-queries',type=int,default=45)
    p.add_argument('--new-protect-per-stage',type=int,default=10)
    p.add_argument('--max-eval',type=int,default=60)
    p.add_argument('--cache-mb',type=float,default=256)
    p.add_argument('--k',type=int,default=10)
    args=p.parse_args()
    if not 0<args.initial_fraction<1: raise ValueError('--initial-fraction must be in (0,1)')
    results=experiment(args)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SAVED',args.out)
if __name__=='__main__':main()
