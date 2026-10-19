"""Opt-in, pinned GitHub public-dataset downloader. Do NOT run on import."""
from __future__ import annotations
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
from hashlib import sha256
import json,time

REPO='https://raw.githubusercontent.com/seukgcode/FastKGE'
# Verified repository default-branch commit on 2026-10-14; update deliberately.
PINNED_COMMIT='777b3d88f2adf3a47cc773072e1fc53aba558f38'
FILES=('entity2id.txt','relation2id.txt','train_id.txt','train.txt','valid.txt','test.txt')

def download(dataset:str,output:str|Path,*,count=5):
    if dataset not in ('WN_CKGE', 'FB_CKGE'):
        raise ValueError('unsupported dataset')
    if not isinstance(count, int) or not 1 <= count <= 5:
        raise ValueError('snapshot count must be between 1 and 5')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest={'repository':REPO,'commit':PINNED_COMMIT,'dataset':dataset,
              'snapshot_semantics':'incremental_train_id','files':{}}
    for i in range(count):
        folder=output/str(i);folder.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            url=f'{REPO}/{PINNED_COMMIT}/data/{dataset}/{i}/{name}'
            p=folder/name
            if not p.exists():
                err=None
                for retry in range(3):
                    try:
                        req=Request(url,headers={'User-Agent':'SafeCadence-Research'})
                        with urlopen(req,timeout=45) as r:
                            # Streaming prevents downloading giant files into memory.
                            with (folder/(name+'.partial')).open('wb') as f:
                                while True:
                                    chunk=r.read(1048576)
                                    if not chunk:break
                                    f.write(chunk)
                        (folder/(name+'.partial')).replace(p)
                        err=None;break
                    except (OSError,HTTPError,URLError) as ex:
                        err=ex;(folder/(name+'.partial')).unlink(missing_ok=True)
                        time.sleep(min(2**retry,4))
                if err is not None:raise RuntimeError(f'failed {url}: {err}') from err
            # Existing files are recorded, but provenance is not guaranteed until
            # a separate upstream checksum is supplied and verified.
            digest=sha256(p.read_bytes()).hexdigest()
            manifest['files'][f'{i}/{name}']={'sha256':digest,'bytes':p.stat().st_size,'source_url':url}
            print(f'[{i}] {name}: {p.stat().st_size:,} bytes',flush=True)
    (output/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return manifest
