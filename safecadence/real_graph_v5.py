"""Reproducible real-source facts, pseudo-temporal CKGE stress test.

CountryInfo installed package ships JSON country records with country/capital,
border, region, subregion, currency and language facts. Data facts are real
source-derived (unlike the synthetic translation generator). The stream order
is RANDOM, not real temporal change; this is NOT WN-CKGE / FB-CKGE.

Dataset files are NOT copied into this research package. They are read from a
separately installed package; source provenance and split seeds are recorded.
"""
from __future__ import annotations
from pathlib import Path
import glob,os,json
import numpy as np
from .data import KnowledgeGraph


def countryinfo_kg(seed=7):
    import countryinfo
    folder=Path(countryinfo.__file__).parent/'data'
    facts=set(); names={}
    documents=[]
    for path in sorted(folder.glob('*.json')):
        try:d=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,UnicodeDecodeError):continue
        name=d.get('name')
        if not isinstance(name,str) or not name.strip():continue
        country='country:'+name.strip()
        iso=(d.get('ISO') or {}).get('alpha3') or d.get('alpha3Code')
        if iso: names[str(iso).upper()]=country
        documents.append((country,d))
    for c,d in documents:
        for key,prefix,relation in [
            ('capital','capital','has_capital'),('region','region','in_region'),
            ('subregion','subregion','in_subregion')]:
            val=d.get(key)
            if isinstance(val,str) and val.strip():facts.add((c,relation,prefix+':'+val.strip()))
        for key,prefix,relation in [
            ('currencies','currency','uses_currency'),('languages','language','speaks_language')]:
            vals=d.get(key) or []
            for v in vals:
                if isinstance(v,str) and v.strip():facts.add((c,relation,prefix+':'+v.strip()))
        for iso in d.get('borders',[]) or []:
            tgt=names.get(str(iso).upper())
            if tgt and tgt!=c:facts.add((c,'borders',tgt))
    facts=sorted(facts)
    E=sorted({x for h,r,t in facts for x in (h,t)})
    R=sorted({r for h,r,t in facts})
    e={x:i for i,x in enumerate(E)}; rel={x:i for i,x in enumerate(R)}
    raw=np.asarray([(e[h],rel[r],e[t]) for h,r,t in facts],dtype=np.int64)
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(raw))
    # Hold out 10% valid, 10% test; ONLY remaining 80% form the train stream.
    n_train=int(.80*len(raw));n_valid=int(.10*len(raw))
    train=raw[indices[:n_train]]
    valid=raw[indices[n_train:n_train+n_valid]]
    test=raw[indices[n_train+n_valid:]]
    info={'source':'countryinfo packaged JSON (not CKGE canonical benchmark)',
          'pseudo_temporal':True,'entity_count':len(E),'relation_count':len(R),
          'fact_count':len(raw),'split_seed':seed,'source_files':len(documents),
          'relations':R,'splits':[len(train),len(valid),len(test)]}
    return KnowledgeGraph(train,valid,test,len(E),len(R),'countryinfo_country_kg'),info
