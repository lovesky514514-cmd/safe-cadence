"""Recreate figures and summary tables from logged outputs; does not rerun training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
m=json.loads((root/'outputs/v4_matched_compute.json').read_text())['summary']
s=json.loads((root/'outputs/v4_protection_sweep.json').read_text())['summary']
names=['certified','sgd_lr_tenth','sgd_lr_0.03x','sgd_l2_3','sgd_replay25','incoming_entities_only']
fig,ax=plt.subplots(figsize=(9,5))
for nm in names:
    x=m[nm]['mean_time_s'];y=m[nm]['mean_new_mrr'];saf=m[nm]['continuous_safe_seeds']
    ax.scatter(x,y,s=80 if nm=='certified' else 50,marker='o' if saf==5 else 'x')
    ax.annotate(f'{nm} ({saf}/5 safe)',(x,y),xytext=(7,2),textcoords='offset points',fontsize=8)
ax.set_title('Equal-walltime synthetic results (all 5 seeds)')
ax.set_xlabel('Mean method CPU walltime per seed (s)')
ax.set_ylabel('Mean incoming MRR')
ax.set_xlim(left=0);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(root/'figures/v4_matched_compute.png',dpi=160);plt.close(fig)
xs=np.array([float(x) for x in s]);ys=[s[str(x)]['new_mrr'] for x in xs];ts=[s[str(x)]['method_wall_seconds'] for x in xs]
fig,ax=plt.subplots(figsize=(8,5));ax.plot(xs,ys,'o-',label='New MRR')
ax.set_xlabel('Fraction of old triples selected for certification');ax.set_ylabel('Incoming MRR')
ax2=ax.twinx();ax2.plot(xs,ts,'s--',label='Walltime');ax2.set_ylabel('Method walltime (seconds)')
ax.grid(alpha=.25);fig.tight_layout();fig.savefig(root/'figures/v4_scope_tradeoff.png',dpi=160);plt.close(fig)
print('saved figures',flush=True)
