"""Generate clearly labeled synthetic smoke fixture (never a benchmark)."""
from pathlib import Path
import argparse

def make(root):
    names=['entity_a','entity_b','entity_c']
    for stage in range(5):
        if stage:names.append('new_'+str(stage))
        folder=root/str(stage);folder.mkdir(parents=True,exist_ok=True)
        (folder/'entity2id.txt').write_text(''.join(f'{v}\t{i}\n' for i,v in enumerate(names)),encoding='utf-8')
        (folder/'relation2id.txt').write_text('related\t0\n',encoding='utf-8')
        h,t=(0,1) if stage==0 else (len(names)-1,0)
        (folder/'train_id.txt').write_text(f'{h}\t0\t{t}\n',encoding='utf-8')
        (folder/'train.txt').write_text(f'{names[h]}\trelated\t{names[t]}\n',encoding='utf-8')
        (folder/'valid.txt').write_text(f'{names[1]}\trelated\t{names[2]}\n',encoding='utf-8')
        (folder/'test.txt').write_text(f'{names[2]}\trelated\t{names[1]}\n',encoding='utf-8')
    (root/'SYNTHETIC_NOT_REAL_BENCHMARK.txt').write_text('Synthetic file format fixture only. No research results.\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/synthetic_smoke')
    a=p.parse_args();make(Path(a.output));print('Created synthetic fixture:',a.output)
