"""Download and integrity-audit the original WN18RR three-file release.
Only run this script in an environment with GitHub network access.
Data redistribution not included in the research ZIP.
"""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
BASE='https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr/'
FILES=('train.txt','valid.txt','test.txt')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dir',type=Path,default=Path('data/wn18rr'))
    p.add_argument('--download',action='store_true',help='Download official files if missing')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    a.dir.mkdir(parents=True,exist_ok=True)
    manifest={'source':BASE,'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'files':{}}
    for fn in FILES:
        path=a.dir/fn
        if a.download and (not path.exists() or a.overwrite):
            print('Downloading',fn,flush=True)
            req=urllib.request.Request(BASE+fn,headers={'User-Agent':'SafeCadenceResearch/0.2'})
            with urllib.request.urlopen(req,timeout=120) as resp:
                payload=resp.read()
            if not payload: raise ValueError('Empty download '+fn)
            tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
        if not path.is_file():raise FileNotFoundError(f'{path}: use --download to retrieve from official source')
        raw=path.read_bytes();text=raw.decode('utf-8')
        triples=[ln for ln in text.splitlines() if ln.strip()]
        if any(len(ln.split())!=3 for ln in triples):raise ValueError('Invalid 3-column triples: '+fn)
        manifest['files'][fn]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'triples':len(triples)}
        print(fn,len(triples),'triples',flush=True)
    (a.dir/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from safecadence.data import load_tsv_dataset,audit_splits
    print('AUDIT:',json.dumps(audit_splits(load_tsv_dataset(a.dir)),indent=2))

if __name__=='__main__':main()
