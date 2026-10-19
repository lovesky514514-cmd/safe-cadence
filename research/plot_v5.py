"""Transparent figures from raw v0.5 JSON without inferential claims."""
import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('figures');p.mkdir(exist_ok=True)
country=json.loads(Path('outputs/v5_country_matched_5seed.json').read_text())
cert=st.mean(s['certified']['new_mrr'] for s in country)
names=['Certified']+[r['config']['name'] for r in country[0]['baselines']]
y=[cert]
safe=[5]
for name in names[1:]:
    rows=[next(x for x in seed['baselines'] if x['config']['name']==name) for seed in country]
    y.append(st.mean(x['new_mrr'] for x in rows))
    safe.append(sum(x['continuously_safe'] for x in rows))
fig,ax=plt.subplots(figsize=(9,4.8));pos=range(len(names));bars=ax.bar(pos,y)
ax.set_xticks(pos,names,rotation=27,ha='right');ax.set_ylabel('New-fact MRR, mean over 5 seeds')
ax.set_title('Real-source country facts, pseudo-temporal, matched walltime')
for i,(bar,v,ok) in enumerate(zip(bars,y,safe)):
    ax.text(i,v+.013,f'{v:.3f} | {ok}/5 safe',ha='center',fontsize=8)
ax.set_ylim(0,max(y)*1.22);fig.tight_layout();fig.savefig(p/'v5_country_matched.png',dpi=200);plt.close(fig)

x=json.loads(Path('outputs/v5_country_transactional.json').read_text());m=json.loads(Path('outputs/v5_country_transaction_micro.json').read_text())
vals={}
vals[0]=(st.mean(r['online_certified']['new_mrr'] for r in x), st.mean(r['online_certified']['method_wall_seconds'] for r in x))
for c in (1,2,3,4):
    rows=[next(r for r in seed['transactions'] if r['commit_every']==c) for seed in m]
    vals[c]=(st.mean(z['new_mrr'] for z in rows),st.mean(z['method_seconds'] for z in rows))
for c in (5,10,20,45,90):
    rows=[next(r for r in seed['transactional'] if r['commit_every']==c) for seed in x]
    vals[c]=(st.mean(z['new_mrr'] for z in rows),st.mean(z['method_seconds'] for z in rows))
fig,ax=plt.subplots(figsize=(7.5,5));
for k,(mrr,t) in vals.items():
    ax.scatter(t,mrr,s=80);ax.annotate('online' if k==0 else f'commit {k}',(t,mrr),xytext=(6,3),textcoords='offset points',fontsize=8)
ax.set_xlabel('Measured CPU walltime, seconds / seed');ax.set_ylabel('New-fact MRR')
ax.set_title('Private staging with atomic safe publication vs per-step gating')
ax.grid(alpha=.2);fig.tight_layout();fig.savefig(p/'v5_transactional_frontier.png',dpi=200);plt.close(fig)

geom=json.loads(Path('outputs/v5_overlay_benchmark.json').read_text())
fig,ax=plt.subplots(figsize=(7.5,4.5))
labels=[f"N={z['N']:,}\nM={z['changed']}" for z in geom]
ax.bar(labels,[z['proposal_speedup'] for z in geom]);ax.axhline(1,linestyle='--',linewidth=1)
ax.set_ylabel('Sparse-overlay / batched exhaustive, per-proposal speedup')
ax.set_title('Sparse-input gate only — synthetic geometry, 100 proposals')
fig.tight_layout();fig.savefig(p/'v5_overlay_synthetic.png',dpi=200);plt.close(fig)
print('Plots generated:',', '.join(str(x) for x in p.glob('v5_*.png')))
