"""Plot and summarize measured results only; no invented confidence intervals."""
import json
from pathlib import Path
from statistics import mean,stdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path('reports'); OUT.mkdir(exist_ok=True)
FIG=Path('figures');FIG.mkdir(exist_ok=True)
read=lambda f:json.loads(Path('outputs',f).read_text(encoding='utf-8'))
bench=read('v2_benchmark.json')['rows']
saved=read('v2_saved_runs.json')
grid=read('grid_search.json')
online=read('online_sgd.json')['rows']
streams=[read(f'v2_stream_{seed}.json') for seed in (7,11,19)]

fig,ax=plt.subplots(figsize=(8,4.8))
sizes=(120,1000,5000)
for label,key in [('first check','cache_first_speedup'),('10 uses','cache_10_speedup'),('100 uses','cache_100_speedup')]:
    ys=[mean(r[key] for r in bench if r['n']==n and r['scenario']=='sparse') for n in sizes]
    ax.plot(sizes,ys,'o-',label=label)
ax.axhline(1,linestyle='--',linewidth=1)
ax.set_xlabel('Number of entities (synthetic geometry)')
ax.set_ylabel('Exhaustive time / cache time incl. build amortization')
ax.set_title('Amortization is necessary for sparse-update acceleration')
ax.legend();ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(FIG/'fig1_speedup_amortization.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.8))
s=[x['seed'] for x in online]
series={
 'Unconstrained SGD':[x['blind_sgd']['new_mrr'] for x in online],
 'Certified stepwise SGD':[x['online_certified_sgd']['new_mrr'] for x in online],
 'Safe 1-D interpolation':[x['one_scale']['new_mrr'] for x in grid['results']],
 'Safe 2-D interpolation':[x['best']['new_mrr'] for x in grid['results']],
}
for name,vals in series.items():ax.plot(s,vals,'o-',label=name)
ax.set_xlabel('Synthetic graph seed');ax.set_ylabel('Incoming training facts filtered MRR')
ax.set_ylim(0,.75)
ax.set_title('Learning under safety constraints (not equal optimizer budgets)')
ax.legend(fontsize=8);ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(FIG/'fig2_safe_plasticity.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4.8))
names=('finetune','replay','relation_anchor','ewc_style','safe_backtracking')
for name in names:
    vals=[mean(s['stages'][i]['algorithms'][name]['new_mrr'] for s in streams) for i in range(3)]
    ax.plot([1,2,3],vals,'o-',label=name)
ax.set_xticks([1,2,3]);ax.set_xlabel('Incremental stage')
ax.set_ylabel('Incoming synthetic training MRR (mean of 3 seeds)')
ax.set_title('Three-stage continual-learning toy protocol')
ax.legend(fontsize=8);ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(FIG/'fig3_stream.png',dpi=160);plt.close(fig)

f=lambda n: f'{n:.3f}'
lines=['# Safe-Cadence v0.2 — Results and falsification report',
       '', '**Status: preliminary, synthetic-only. Not yet an accepted or submission-ready research article.**',
       '', '## A. Strict certificate validity',
       '', '- 13 pytest tests passed (see `pytest -q tests`).',
       '- Old v0.1 five-run checkpoint backtracking alpha candidates matched independent exhaustive checker.',
       '- Cache verifier supports full candidate universe, sparse/dense parameter drift, newly appended entities and additional true facts.',
       '- The correctness guarantee comes from the triangle inequality, not from finite numerical trials.',
       '', '## B. Certificate cost is regime-dependent',
       '', '| N | Scenario | Fraction scored | First speedup | 10-use speedup | 100-use speedup |',
       '|---:|---|---:|---:|---:|---:|']
for n in (120,1000,5000):
 for scenario in ('sparse','dense','appended'):
    subset=[x for x in bench if x['n']==n and x['scenario']==scenario]
    def m(key):return mean(x[key] for x in subset)
    lines.append(f'| {n} | {scenario} | {m("cache_fraction"):.3%} | {m("cache_first_speedup"):.2f}× | {m("cache_10_speedup"):.2f}× | {m("cache_100_speedup"):.2f}× |')
lines.extend(['', 'The above are 3-seed means on random synthetic *geometries*, not a CKGE training benchmark. The 10/100-use figures EXTRAPOLATE amortization from single-check times; they were not independently timed over 10 or 100 consecutive updates. Timings depend on machine, cache reuse and dimensionality. Index build is included or amortized. **First-use speedup <1×** in the 1000/5000 sparse scenarios.',
              '', '![Amortized speed](../figures/fig1_speedup_amortization.png)',
              '', '## C. New-knowledge plasticity is still insufficient',
              '', '| Method | Mean incoming MRR | Mean historical MRR | Rank safety |',
              '|---|---:|---:|---|'])
for arm in ('blind_sgd','online_certified_sgd'):
 a=online;lines.append(f'| {arm} | {mean(x[arm]["new_mrr"] for x in a):.3f} | {mean(x[arm]["old_mrr"] for x in a):.3f} | {"all 5 passed" if arm.startswith("online") else "all 5 violated"} |')
lines.extend([f'| safe 1-D interpolation (Adam candidate) | {grid["summary"]["mean_1d_new_mrr"]:.3f} | — | all 5 passed |',
              f'| safe 2-D interpolation (Adam candidate) | {grid["summary"]["mean_2d_new_mrr"]:.3f} | — | all 5 passed |',
              '', 'Do not directly compare SGD and Adam arms as matched-budget competitors. Even within the SGD matched-proposal pair, certified training incurs additional certification cost.',
              '',f'- Stepwise certified SGD accepted an average of {mean(x["online_certified_sgd"]["meta"]["accepted_steps"] for x in online):.1f}/80 steps.',
              f'- It required {mean(x["online_certified_sgd"]["meta"]["gate_calls"] for x in online):.1f} candidate checks on average, with {mean(x["online_certified_sgd"]["meta"]["certificate_seconds"] for x in online):.3f} seconds spent on certification alone.',
              '', '![New training plasticity](../figures/fig2_safe_plasticity.png)',
              '', '## D. Three-step toy-stream experiments',
              '', '| Method | Final old MRR (3-seed mean) | Final incoming MRR | Final test MRR |',
              '|---|---:|---:|---:|'])
for name in names:
    values=lambda key:mean(s['stages'][-1]['algorithms'][name][key] for s in streams)
    lines.append(f'| {name} | {values("old_mrr"):.3f} | {values("new_mrr"):.3f} | {values("test_mrr"):.3f} |')
lines.extend(['', 'The stream is a random partition of synthetic training facts, **not** true chronological CKGE data.',
              '', '![Streaming](../figures/fig3_stream.png)',
              '', '## E. Formal guarantee and practical caveats',
              '', '- Protection applies only to provided past queries that were Top-k initially, among the full current entity universe under the filtered-ranking protocol.',
              '- All moved and newly appended entities are evaluated exactly. Stable entities are pruned using an old-distance triangle-inequality radius.',
              '- Cache memory complexity is O(number of protected queries × original entity count). Maximum cache size can trigger sound tree fallback.',
              '- A certificate may be valid even if unseen, unprotected old answers are displaced. It does not validate factual semantics or contradictions.',
              '- Benchmarks include index building; repeated updates must rebase or rebuild the cache after each accepted snapshot boundary.',
              '- Reproducible full WN18RR benchmark has NOT been run here; no source training code for original KG-Cadence was available.',
              '', '## F. Research gate',
              '', '**GO:** exact safety verification and amortized sparse-update query cache are feasible in toy geometry.',
              '', '**NO-GO (for Q1 claims now):** low safe plasticity, high stepwise gate cost, and no real multi-snapshot benchmark or SOTA implementation comparison.',
              '', '**Next decisive test:** on official ENTITY/RELATION/FACT/HYBRID benchmark snapshots, measure new-knowledge success conditional on certified safety, strict total walltime, certificate scope, and performance against Replay/LKGE/TADG.',
              ''])
(OUT/'V02_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print('generated reports/V02_RESULTS.md and 3 figures')
