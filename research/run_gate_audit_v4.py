"""Exhaustively compare *every* batched gate decision against independent rank checker."""
import json,numpy as np
from pathlib import Path
from safecadence.data import synthetic_graph
from safecadence.diagnostics_v3 import run_variant,Variant,make_batches

def main():
  rows=[]
  for seed in range(1,6):
    state=np.load(f'outputs/multiseed_{seed}.npz');E0,R0=state['old_entities'],state['old_relations']
    g=synthetic_graph(seed=seed);rng=np.random.default_rng(seed);order=rng.permutation(len(g.train))
    old=g.train[order[:300]];incoming=g.train[order[300:]]
    protected=old[rng.choice(len(old),50,replace=False)]
    batches=make_batches(incoming,np.concatenate((old,incoming)),len(E0),steps=80,seed=seed*19)
    _,_,s=run_variant(E0,R0,incoming,old,protected,batches,Variant('certified',certified=True),
                      audit_decisions=True,audit_stride=1,certificate_engine='batched')
    row={'seed':seed,'calls':s['certificate_calls'],'independent_audited_calls':s['independently_audited_gate_calls'],
         'false_accepts':s['gate_false_accepts'],'false_rejects':s['gate_false_rejects'],
         'audit_wall_excluded':s['audit_exhaustive_seconds'],'new_mrr':s['new_mrr']}
    rows.append(row);print(row,flush=True)
  obj={'scope':'every gate proposal, 5 synthetic seeds, fixed fact filtering',
       'rows':rows,'totals':{f:sum(r[f] for r in rows) for f in ['calls','independent_audited_calls','false_accepts','false_rejects']}}
  Path('outputs/v4_gate_full_audit.json').write_text(json.dumps(obj,indent=2))
  print('SUMMARY',obj['totals'])
if __name__=='__main__':main()
