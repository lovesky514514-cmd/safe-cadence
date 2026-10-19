"""Create all v0.3 figures and a results report from the recorded JSON only."""
from pathlib import Path
from statistics import mean
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'outputs/v3_falsification.json').read_text())
A=D['aggregate']
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
REP=ROOT/'reports';REP.mkdir(exist_ok=True)

names=['sgd','sgd_lr_half','sgd_lr_tenth','sgd_strong_l2',
       'sgd_incoming_entities_only','sgd_freeze_entities','certified_sgd',
       'certified_lr_half','certified_lr_tenth','certified_strong_l2',
       'certified_incoming_entities_only']
labels=['SGD','SGD 0.5x','SGD 0.1x','Strong L2',
        'Incoming-only','Freeze entities','Certified SGD',
        'Cert. 0.5x','Cert. 0.1x','Cert. strong L2','Cert. incoming-only']

fig,ax=plt.subplots(figsize=(10,5.5))
x=np.arange(len(names))
y=[A[n]['new_mrr']['mean'] for n in names]
colors=['#A7ADB7' if not n.startswith('certified') else '#477EAF' for n in names]
bars=ax.bar(x,y,color=colors)
ax.axhline(.187582,linestyle='--',color='#BC7440',label='Mean initial incoming MRR')
for i,b in enumerate(bars):
    t=A[names[i]]['safe_endpoints']
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+.009,f'{t}/5',ha='center',fontsize=9)
ax.set_xticks(x);ax.set_xticklabels(labels,rotation=37,ha='right')
ax.set_ylim(0,.56);ax.set_ylabel('Mean incoming train MRR')
ax.set_title('v0.3: Plasticity and final exact-check safety (labels = safe runs / 5)')
ax.legend(loc='upper right');fig.tight_layout()
fig.savefig(FIG/'v3_plasticity_vs_safety.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(9.5,5.5))
for n,l in zip(names,labels):
    v=A[n];w=v['method_wall_seconds']['mean'];u=v['new_mrr']['mean'];safe=v['safe_endpoints']
    ax.scatter(w,u,s=50+30*safe,alpha=.85,color='#477EAF' if n.startswith('certified') else '#A7ADB7')
    if n in ('sgd','sgd_lr_half','sgd_lr_tenth','sgd_strong_l2','certified_sgd','certified_incoming_entities_only'):
        ax.annotate(l,(w,u),xytext=(5,4),textcoords='offset points',fontsize=9)
ax.set_xscale('log');ax.set_xlabel('Mean method walltime per run (s, excludes offline audit)')
ax.set_ylabel('Incoming train MRR')
ax.set_title('v0.3: End-to-end CPU cost vs incoming learning (marker area ~ safe seeds)')
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIG/'v3_walltime_vs_plasticity.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.8,4.8))
cn=[n for n in names if n.startswith('certified')]
cx=np.arange(len(cn));cert=[A[n]['certificate_seconds']['mean'] for n in cn]
train=[A[n]['gradient_seconds']['mean'] for n in cn]
ax.bar(cx,cert,label='Certificate verification')
ax.bar(cx,train,bottom=cert,label='Gradient calculations')
ax.set_xticks(cx);ax.set_xticklabels([l.replace('Cert. ','') for l in labels[-5:]],rotation=24,ha='right')
ax.set_ylabel('Mean measured seconds (CPU)')
ax.set_title('v0.3: Certification dominates the measured compute budget')
ax.legend();fig.tight_layout();fig.savefig(FIG/'v3_cost_breakdown.png',dpi=160);plt.close(fig)

s=lambda name,metric:f'{A[name][metric]["mean"]:.3f}'
rows='\n'.join(f'| {l} | {s(n,"new_mrr")} | {s(n,"old_mrr")} | {s(n,"test_mrr")} | {A[n]["safe_endpoints"]}/5 | {s(n,"method_wall_seconds")} | {s(n,"all_old_break_fraction")} |' for n,l in zip(names,labels))

report=f'''# Safe-Cadence v0.3 — Falsification and ablation report

**Status: completed 5-seed, 80-step synthetic-only negative-control experiment. Not submission-ready.**

### Main answer to the v0.2 review

- Is the certificate independently useful compared to just lowering LR or adding L2? **Conditionally yes on this particular synthetic/strict-Top-10 protocol**: certified SGD has mean incoming MRR {s('certified_sgd','new_mrr')} and strict final safety 5/5, vs a strong L2 baseline {s('sgd_strong_l2','new_mrr')} and 5/5 final safety, and 0.1x LR {s('sgd_lr_tenth','new_mrr')} and 3/5 final safety.
- Is it currently a competitive algorithm? **No.** Certified SGD takes {s('certified_sgd','method_wall_seconds')} seconds per run versus {s('sgd_strong_l2','method_wall_seconds')} seconds for L2 on this machine, and costs roughly {A['certified_sgd']['method_wall_seconds']['mean']/A['sgd_strong_l2']['method_wall_seconds']['mean']:.1f}x more walltime.
- Does final safety imply continuous online safety? **No.** The strong L2 baseline had unsafe intermediate snapshots in seeds 1 and 5, despite a 5/5 safe final snapshot. A certified trajectory passed each checked attempted update under the current floating-point implementation.
- Does it protect the entire historical graph? **No.** Certified SGD still displaced an average of {A['certified_sgd']['all_old_break_fraction']['mean']*100:.2f}% of initially eligible old queries over the **larger, unprotected** historical set. Its only protection scope is the sampled historical subset.

## A. Full matched-proposal table

Five seeds, 80 minibatch proposals per method/seed; batch ID and negative samples are identical per seed. Initial incoming-train MRR = **0.188** mean. All MRR values are *synthetic data*, and the test MRR is only a toy held-out measure. Final strict safety is independently checked on the originally Top-10 old queries among the 50 sampled old triples. The last column uses all old train triples and measures out-of-scope old-query Top-10 damage, not a certificate failure.

| Method | Incoming train MRR | Old train MRR | Toy test MRR | Final subset safety | Wall s/run | All-old damage fraction |
|---|---:|---:|---:|---:|---:|---:|
{rows}

![Plasticity and subset safety](../figures/v3_plasticity_vs_safety.png)

## B. Safety audit and rejection diagnostics

- Certified SGD attempted **{s('certified_sgd','certificate_calls')}** candidate checks per 80-step run (five-seed mean). It accepted **{s('certified_sgd','accepted_steps')}** steps, fully rejected **{s('certified_sgd','rejected_steps')}** steps, and rescued **{s('certified_sgd','rescued_steps')}** otherwise unsafe full steps with smaller step sizes.
- In addition to the final exhaustive check for every method/seed, every **20th gate proposal** is independently exhaustively evaluated. Certified SGD audited {sum(r['methods']['certified_sgd']['independently_audited_gate_calls'] for r in D['rows'])} proposals over 5 seeds. Empirical mismatches: **0 false accepts; 0 false rejects** among that audited subset. **This is not a full numerical soundness proof, and is not a zero-false-reject population estimate.**
- *Gate false reject* means a rejected **particular candidate step** is independently found safe; it is distinct from **fully rejected learning steps** and from an update direction that might admit a tiny safe step outside the eight fixed alpha levels.
- Uncertified SGD had unsafe intermediate snapshots in every seed (counts: {[r['methods']['sgd']['unsafe_intermediate_steps'] for r in D['rows']]} / 80). Strong L2 had counts {[r['methods']['sgd_strong_l2']['unsafe_intermediate_steps'] for r in D['rows']]}; safe final snapshots do not certify all intermediate states.
- Per-step gate auditing is **excluded** from the method runtime and shown separately in the JSON. The online system does not perform the independent exhaustive checker at runtime; it uses its cached verifier.

## C. Cost audit

| Method | Gradient s | Certificate s | Index build s | Overall method wall s | Gate proposals/run |
|---|---:|---:|---:|---:|---:|
'''
for n,l in zip(names,labels):
    report+=f'| {l} | {s(n,"gradient_seconds")} | {s(n,"certificate_seconds")} | {s(n,"index_build_seconds")} | {s(n,"method_wall_seconds")} | {s(n,"certificate_calls")} |\n'
report+='''
The method walltime includes parameter-update and Python-loop overhead, gate checks for certified arms, and index building only when used by the method. It excludes **offline** brute-force decision auditing and final metrics. No extra training steps are given to uncertified baselines to equalize their vastly lower walltime. Therefore these are **matched gradient-proposal**, not matched-compute comparisons.

![Cost breakdown](../figures/v3_cost_breakdown.png)

![Walltime tradeoff](../figures/v3_walltime_vs_plasticity.png)

## D. Limitations and falsification gates

1. **Synthetic only.** Reused the 5 v0.1 saved old-task states; no authoritative CKGE snapshots or original KG-Cadence optimizer have been obtained. The source dataset generator has no newly appearing entities; therefore `incoming_entities_only` means updating **old entities that occur in the incoming minibatch**, with all relation vectors frozen. It must NOT be described as training only genuinely new entities.
2. **Unfair walltime.** Online certification currently loses badly in full compute cost. This version answers the simple-baseline objection under matched *gradient proposal counts* only.
3. **Safety scope.** ~50 sampled old triples (both query directions that were initially Top-10); it does not protect all historical facts or validate truth. At each update the new positive facts are additionally filtered; the initially eligible subset is selected using old-only true facts.
4. **Float soundness.** Original proof uses exact arithmetic; float64/slack implementation lacks directed-rounding guarantee.
5. **Certifier false rejection.** Empirically zero mismatch **on the sampled gate checks only**. This says nothing about skipped alphas or the existence of a different safe descent direction.
6. **No SOTA comparisons yet.** TADG, LKGE, etc. have not been run in this package.

### GO / NO-GO (specific to this round)

- **GO for algorithmic research:** strictly safe accepted updates can meaningfully beat extremely conservative fixed hyperparameter baselines at same nominal gradient proposal count.
- **NO-GO for Q1/industry claims:** certificate costs ~25x strong-L2 baseline, the protected scope is narrow, and real continual-KG evaluation is still absent.
- **Next experiment:** match *total CPU time*, vary protected set fraction and the safety budget, then seek faster gate with sparse row-update and candidate-set indexing. If certified updates cannot outperform a grid of noncertified alternatives at comparable total CPU time under the same strict continuous-safety requirement, stop claiming efficiency.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
python run_falsification_v3.py --seeds 5 --steps 80
python analyze_v3.py
```

Run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for less multithreading noise; hardware and timing still vary. `outputs/v3_falsification.json` records each seed and all intermediate-safety counts.
'''
(REP/'V03_FALSIFICATION.md').write_text(report,encoding='utf-8')
print('Report and figures created from',len(D['rows']),'seeds')
