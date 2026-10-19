"""Offline, intentionally NON-SOTA CKGE loader/training smoke pipeline.

Outputs are identified as SMOKE, not research results: no hyperparam tuning,
no tuned competing implementations, no baseline parity or exhaustive sweep.
"""
from __future__ import annotations
import sys,os,json,platform,time,random
from pathlib import Path
import numpy as np
from .ckge_protocol import load_ckge,save_report

def run_pilot(root,output,*,max_snapshots=2,steps=12,seed=42,
              max_eval=24,dim=24):
    if steps<1 or dim<1 or max_eval<1:raise ValueError('positive steps, dim, max_eval required')
    # Import heavier packages only for training actions.
    from safecadence.transe import TrainConfig,train_transe
    from safecadence.ranking import filtered_metrics
    import torch
    torch.set_num_threads(1)
    stage_data=load_ckge(root,snapshot_count=max_snapshots)
    rng=np.random.default_rng(seed)
    E=R=None;metrics=[]
    for stage in stage_data:
        begun=time.perf_counter()
        init=None
        if E is not None:
            if stage.n_relations!=len(R):raise ValueError('relation expansion unsupported in smoke')
            # Crucial: append new entity embeddings without disturbing stored IDs.
            added=rng.uniform(-0.4,0.4,(stage.n_entities-len(E),dim))
            init=(np.concatenate((E,added),axis=0),R.copy())
        cfg=TrainConfig(dim=dim,steps=steps,batch_size=32,negatives=2,
                        lr=0.025,seed=seed+stage.index)
        # Negative resampling sees the cumulative training facts ONLY.
        E,R,train_meta=train_transe(stage.incoming,stage.n_entities,
                                  stage.n_relations,cfg,init=init,
                                  forbidden_positive=stage.cumulative)
        # Filter with visible stage truth for evaluation only. No future facts.
        eval_true=np.concatenate((stage.cumulative,stage.valid,stage.test),axis=0)
        test=filtered_metrics(E,R,stage.test,eval_true,max_eval=max_eval)
        old=filtered_metrics(E,R,stage.cumulative,eval_true,max_eval=max_eval)
        metrics.append({'stage':stage.index,'entities':stage.n_entities,
                        'relations':stage.n_relations,'incoming_train':len(stage.incoming),
                        'cumulative_train':len(stage.cumulative),'test_mrr':test['mrr'],
                        'old_train_mrr':old['mrr'],'test_eval_queries':test['ranks'],
                        'seconds':round(time.perf_counter()-begun,3),
                        'train_config':vars(cfg),'train_loss_log':train_meta['loss_log']})
    result={'status':'SMOKE_ONLY_NOT_CANONICAL_BENCHMARK',
            'warning':'No SOTA baselines; small training budget; not for a paper table.',
            'seed':seed,'dataset_dir':str(Path(root).resolve()),
            'python':sys.version.split()[0], 'numpy':np.__version__,
            'torch':torch.__version__,'platform':platform.platform(),
            'stages':metrics}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    save_report(result,out)
    print(f'Smoke completed: {len(metrics)} stages; output {out}')
    return result
