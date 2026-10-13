#!/usr/bin/env python3
"""Build the paper's fixed evaluation datasets from official sources or local inputs."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys
import tarfile
import threading
from urllib.parse import urlsplit, unquote

REPO=Path(__file__).resolve().parents[1]
DEFAULT_SPLITS=['iid','partial_ood','cell_ood','structure_ood','lmc_source_shift','retrieval1000']


def digest(b):return hashlib.sha256(b).hexdigest()


def safe_join(root, rel):
    p=PurePosixPath(rel)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in rel:
        raise ValueError(f'Unsafe relative path: {rel!r}')
    target=Path(root).joinpath(*p.parts)
    if not target.resolve().is_relative_to(Path(root).resolve()):raise ValueError(f'Path escapes root: {rel}')
    return target


def read_jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]


def atomic_bytes(path, b):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.part')
    try:
        tmp.write_bytes(b);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def valid(path, r):
    return path.is_file() and path.stat().st_size==r['image_bytes'] and digest(path.read_bytes())==r['image_sha256']


def load_release(repo, splits):
    cfg=json.loads((repo/'config/release.json').read_text())
    recipes={r['source']+'/'+r['source_path']:r for r in read_jsonl(repo/'config/recipes.jsonl.gz')}
    sets={}
    for split in splits:
        if split not in cfg['splits']:raise ValueError(f'Unknown/non-paper split {split!r}')
        f=repo/'manifests'/f'{split}.jsonl.gz';expected=cfg['splits'][split]
        if digest(f.read_bytes())!=expected['manifest_sha256']:raise ValueError(f'Manifest checksum mismatch: {split}')
        rows=read_jsonl(f)
        if len(rows)!=expected['records'] or not rows:raise ValueError(f'Manifest count mismatch: {split}')
        if len({r['key'] for r in rows})!=len(rows):raise ValueError(f'Duplicate keys: {split}')
        classes=collections.Counter(r['text'] for r in rows) if not split.startswith('retrieval') else {}
        if dict(classes)!=expected['classes']:raise ValueError(f'Class membership mismatch: {split}')
        for r in rows:
            safe_join(Path('/tmp/micropool-path-validation'),r['image_path'])
            safe_join(Path('/tmp/micropool-path-validation'),r['source_path'])
            if r['metadata']['filepath']!=r['image_path']:raise ValueError('Metadata/path mismatch')
            if r['source']+'/'+r['source_path'] not in recipes:raise ValueError('Missing source recipe')
            if split!='lmc_source_shift' and not r['metadata'].get('long_caption','').strip():raise ValueError('Missing detail caption')
            if split.startswith('retrieval') and r['text']!=r['metadata']['long_caption']:raise ValueError('Retrieval caption mismatch')
        sets[split]=rows
    if cfg.get('recipes_sha256') and digest((repo/'config/recipes.jsonl.gz').read_bytes())!=cfg['recipes_sha256']:
        raise ValueError('Recipe checksum mismatch')
    for rel,h in cfg.get('semantic_sha256',{}).items():
        if digest(safe_join(repo,rel).read_bytes())!=h:raise ValueError(f'Semantic table checksum mismatch: {rel}')
    return cfg,sets,recipes


def check_semantics(repo,sets):
    def read(name,key):
        with (repo/'semantic_caption'/name).open() as f:
            return {r[key]:r['semantic_caption'] for r in csv.DictReader(f,delimiter='\t') if r['semantic_caption'].strip()}
    structures=read('A_structures_split.tsv','structure');cells=read('cell_semantic_recaption_manifest.tsv','cell_line')
    for split,rows in sets.items():
        if split.startswith('retrieval') or split=='lmc_source_shift':continue
        for r in rows:
            m=r['metadata'];s=m.get('canonical_organelle','');c=m.get('canonical_cell','')
            if not s or not c or c not in cells or any(p not in structures for p in s.split('_and_')):
                raise ValueError(f'L3 lookup failed: {split}/{r["key"]}')
            if r['text']!='a microscopy photo of '+s+'_of_'+c:
                raise ValueError(f'L1 label mismatch: {split}/{r["key"]}')
    return structures,cells


def unique_images(sets):
    unique={}
    for rows in sets.values():
        for r in rows:
            old=unique.get(r['image_path'])
            if old and old['image_sha256']!=r['image_sha256']:raise ValueError('Conflicting reference images')
            unique[r['image_path']]=r
    return list(unique.values())


def prepared_input(args,r):
    for root in args.prepared_root:
        candidates=[safe_join(root,r['image_path'])]
        if r['image_path'].startswith('fluogen/'):
            candidates.append(safe_join(root,r['image_path'][8:]))
        elif r['source']=='lightmycells':
            candidates.append(safe_join(root,r['image_path'].removeprefix('lightmycells/')))
        for path in candidates:
            if path.is_file():return path
    return None


def raw_input(args,r):
    for root in args.raw_root:
        path=safe_join(root,r['source_path'])
        if path.is_file():return path
        if r['source']=='lightmycells':
            path=safe_join(root,r['source']+'/'+r['source_path'])
            if path.is_file():return path
    return None


def plan(args,sets,recipes):
    rows=unique_images(sets);counts=collections.Counter();blockers=[]
    for r in rows:
        recipe=recipes[r['source']+'/'+r['source_path']]
        if valid(safe_join(args.root/'images',r['image_path']),r):status='cached'
        elif prepared_input(args,r):status='local_prepared'
        elif raw_input(args,r):status='local_raw'
        elif r['source']=='fluo' and 'fluo' not in args.authorized_source:status='permission_required'
        elif recipe['kind']=='manual':status='manual_import'
        else:status='download_'+recipe['kind']
        counts[(r['source'],status)]+=1
        if status in ('manual_import','permission_required'):
            blockers.append(dict(source=r['source'],status=status,source_path=r['source_path'],image_path=r['image_path']))
    print('Paper evaluation samples:',{k:len(v) for k,v in sets.items()},flush=True)
    print(f'{len(rows)} unique images. No resampling; exact image SHA256 required.',flush=True)
    for (source,status),n in sorted(counts.items()):print(f'  {source:22} {status:22} {n:6}',flush=True)
    args.root.mkdir(parents=True,exist_ok=True)
    atomic_bytes(args.root/'plan.json',json.dumps(dict(counts=[dict(source=s,status=t,images=n) for (s,t),n in sorted(counts.items())],blockers=blockers),indent=2).encode())
    if blockers:
        print(f'{len(blockers)} images require local input or permission. See plan.json and docs/SOURCES.md.',flush=True)
    return blockers


def save_checked(args,r,b):
    if len(b)!=r['image_bytes'] or digest(b)!=r['image_sha256']:
        raise ValueError(f'Image differs from frozen evaluation reference: {r["image_path"]}')
    atomic_bytes(safe_join(args.root/'images',r['image_path']),b)


def import_shards(args,sets,cfg):
    """Import only matching image bytes from an existing private author/rebuilder tree."""
    for split,rows in sets.items():
        if not args.from_shards:continue
        source=safe_join(args.from_shards,cfg['splits'][split]['output'])
        if not source.is_file():raise FileNotFoundError(source)
        expected={r['key']+'.'+r['extension']:r for r in rows};found=set()
        with tarfile.open(source,'r:') as t:
            for m in t:
                if m.name in expected:
                    if m.name in found or not m.isfile():raise ValueError('Duplicate or invalid image member')
                    r=expected[m.name];save_checked(args,r,t.extractfile(m).read());found.add(m.name)
        if found!=set(expected):raise ValueError(f'Incomplete input shard: {source}')
        print(f'Imported and verified {split}: {len(found)} images',flush=True)


def fetch(args,sets,recipes):
    from image_transform import convert
    from transport import download,open_zip
    pending=[];errors=[]
    for r in unique_images(sets):
        if not valid(safe_join(args.root/'images',r['image_path']),r):pending.append(r)
    groups=collections.defaultdict(list);singles=[]
    for r in pending:
        rec=recipes[r['source']+'/'+r['source_path']]
        if not prepared_input(args,r) and not raw_input(args,r) and rec['kind'] in ('zip','tar'):
            groups[(rec['kind'],rec['url'])].append((r,rec))
        else:singles.append((r,rec))
    locks=collections.defaultdict(threading.Lock)
    def one(item):
        r,recipe=item;prepared=prepared_input(args,r)
        if prepared:save_checked(args,r,prepared.read_bytes());return
        raw=raw_input(args,r)
        if raw:save_checked(args,r,convert(raw,r['transform']));return
        if args.offline:raise FileNotFoundError(f'Offline input missing: {r["source_path"]}')
        if recipe['kind']=='manual':raise FileNotFoundError(f'Manual source: {r["source"]}/{r["source_path"]}')
        url=recipe['url'];suffix=Path(unquote(urlsplit(url).path)).suffix
        raw=args.root/'cache'/(digest(url.encode())+suffix)
        with locks[url]:download(url,raw,recipe.get('raw_sha256'))
        save_checked(args,r,convert(raw,r['transform'],recipe.get('extract')))
    def archive(task):
        (kind,url),items=task
        if args.offline:raise FileNotFoundError(f'Offline archive inputs missing: {url}')
        if any(r['source']=='fluo' for r,_ in items) and 'fluo' not in args.authorized_source:
            raise PermissionError('CTC permission required; see docs/SOURCES.md (fluo)')
        if kind=='zip':z=open_zip(url,args.archives)
        else:
            local=args.archives/Path(urlsplit(url).path).name
            if not local.is_file():local=download(url,args.root/'cache'/Path(urlsplit(url).path).name)
            z=tarfile.open(local)
        with z:
            names=z.namelist() if kind=='zip' else z.getnames()
            for r,recipe in items:
                try:
                    member=recipe.get('member')
                    if 'member_regex' in recipe:
                        matches=[n for n in names if re.fullmatch(recipe['member_regex'],n)]
                        if len(matches)!=1:raise ValueError(f'Archive member ambiguous/missing: {r["source_path"]}')
                        member=matches[0]
                    if member not in names:raise FileNotFoundError(f'{member} absent in {url}')
                    b=z.read(member) if kind=='zip' else z.extractfile(member).read()
                    raw=args.root/'cache'/(digest((url+'#'+member).encode())+Path(member).suffix)
                    if recipe.get('decode')=='ht29':
                        import zlib,numpy as np,tifffile
                        data=zlib.decompress(b,-15)
                        if len(data)!=52+512*512*2:raise ValueError('Invalid HT29 DIB size')
                        raw=raw.with_suffix('.tif');raw.parent.mkdir(parents=True,exist_ok=True)
                        tifffile.imwrite(raw,np.frombuffer(data,dtype='<u2',offset=52).reshape(512,512))
                    else:atomic_bytes(raw,b)
                    save_checked(args,r,convert(raw,r['transform']))
                except Exception as e:errors.append(dict(image_path=r['image_path'],error=str(e)))
    jobs=[(one,i) for i in singles]+[(archive,i) for i in groups.items()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs={pool.submit(fn,item):item for fn,item in jobs}
        for n,f in enumerate(as_completed(futs),1):
            try:f.result()
            except Exception as e:
                item=futs[f];r=item[0] if isinstance(item[0],dict) else None
                errors.append(dict(image_path=r['image_path'] if r else str(item[0]),error=f'{type(e).__name__}: {e}'))
            if n%100==0 or n==len(futs):print(f'Fetch/import jobs {n}/{len(futs)}; failures {len(errors)}',flush=True)
    atomic_bytes(args.root/'fetch_report.json',json.dumps(dict(jobs=len(jobs),failures=errors),indent=2).encode())
    if errors:raise RuntimeError(f'{len(errors)} fetch/import failures; see {args.root}/fetch_report.json')


def add(t,name,b):
    info=tarfile.TarInfo(name);info.size=len(b);info.mtime=0;info.mode=0o644
    t.addfile(info,io.BytesIO(b))


def build(args,sets,cfg):
    # Validate all inputs BEFORE creating any output shards.
    bad=[r['image_path'] for r in unique_images(sets) if not valid(safe_join(args.root/'images',r['image_path']),r)]
    if bad:raise ValueError(f'{len(bad)} missing/mismatched images; first: {bad[:3]}')
    for split,rows in sets.items():
        target=safe_join(args.root,cfg['splits'][split]['output']);target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.tar.part')
        try:
            with tarfile.open(tmp,'w',format=tarfile.USTAR_FORMAT) as t:
                for r in rows:
                    b=safe_join(args.root/'images',r['image_path']).read_bytes()
                    if digest(b)!=r['image_sha256']:raise ValueError('Image changed during build')
                    add(t,r['key']+'.'+r['extension'],b)
                    add(t,r['key']+'.txt',r['text'].encode())
                    add(t,r['key']+'.json',json.dumps(r['metadata'],ensure_ascii=False,sort_keys=True).encode())
            os.replace(tmp,target)
        finally:tmp.unlink(missing_ok=True)
        tsv=args.root/'tables'/f'{split}.tsv';tsv.parent.mkdir(parents=True,exist_ok=True)
        with tsv.open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['filepath','title','long_caption'])
            for r in rows:w.writerow([str(safe_join(args.root/'images',r['image_path']).resolve()),r['text'],r['metadata'].get('long_caption','')])
        print(f'Built {split}: {len(rows)} records',flush=True)


def verify(args,sets,cfg):
    stats={}
    for split,rows in sets.items():
        path=safe_join(args.root,cfg['splits'][split]['output'])
        if not path.is_file():raise FileNotFoundError(f'Missing shard: {path}')
        expected={}
        for r in rows:
            for ext in [r['extension'],'txt','json']:expected[r['key']+'.'+ext]=r
        found=set()
        with tarfile.open(path,'r:') as t:
            for m in t:
                if m.name not in expected or m.name in found or not m.isfile():raise ValueError(f'Unexpected/duplicate member {m.name}')
                r=expected[m.name];b=t.extractfile(m).read();ext=m.name.rsplit('.',1)[1]
                if ext=='json':ok=json.loads(b)==r['metadata']
                elif ext=='txt':ok=b==r['text'].encode()
                else:ok=len(b)==r['image_bytes'] and digest(b)==r['image_sha256']
                if not ok:raise ValueError(f'Content mismatch: {split}/{m.name}')
                found.add(m.name)
        if found!=set(expected):raise ValueError(f'Missing {len(set(expected)-found)} members: {split}')
        extra=list(path.parent.glob('shard-*.tar'))
        if extra!=[path]:raise ValueError(f'Unexpected extra shards under {path.parent}')
        stats[split]=dict(records=len(rows),verified=True)
        print(f'PASS {split}: every image, text, metadata and member checked',flush=True)
    atomic_bytes(args.root/'verification.json',json.dumps(dict(release=cfg['release'],splits=stats),indent=2).encode())
    write_env(args,sets,cfg)


def write_env(args,sets,cfg):
    variables={'iid':'IID_DATA','partial_ood':'PARTIAL_OOD_DATA','cell_ood':'CELL_OOD_DATA',
       'structure_ood':'STRUCTURE_OOD_DATA','lmc_source_shift':'SOURCE_SHIFT_DATA','retrieval1000':'RETRIEVAL_DATA_1000'}
    lines=['# Generated after exact verification. Use paper evaluation, not legacy train.sh.',
           'export MICRO_DATA_ROOT='+shlex.quote(str(args.root.resolve()))]
    for split in sets:lines.append('export '+variables[split]+'='+shlex.quote(str(safe_join(args.root,cfg['splits'][split]['output']).resolve())))
    atomic_bytes(args.root/'micropool_eval.env',('\n'.join(lines)+'\n').encode())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['plan','fetch','build','verify','all'],default='all')
    p.add_argument('--splits',nargs='+',choices=DEFAULT_SPLITS,default=DEFAULT_SPLITS)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--prepared-root',action='append',type=Path,default=[])
    p.add_argument('--raw-root',action='append',type=Path,default=[])
    p.add_argument('--from-shards',type=Path,help='Private import from an existing evaluation tree; never shipped')
    p.add_argument('--archives',type=Path,help='Directory of officially downloaded ZIP/TGZ archives')
    p.add_argument('--authorized-source',action='append',choices=['fluo'],default=[],help='CTC permission has actually been obtained; this flag grants no rights')
    p.add_argument('--offline',action='store_true')
    a=p.parse_args()
    if a.workers<1:p.error('--workers must be positive')
    a.root=a.root.resolve();a.archives=a.archives or a.root/'archives'
    if a.root==REPO.parent or a.root.is_relative_to(REPO.parent):p.error('Keep rebuilt images outside the MicroCLIP repository')
    cfg,sets,recipes=load_release(REPO,a.splits);check_semantics(REPO,sets)
    if a.stage in ('plan','all'):
        blockers=plan(a,sets,recipes)
        if a.stage=='plan':return 2 if blockers else 0
        if blockers and not a.from_shards:
            raise RuntimeError('Preflight blocked; supply the listed local inputs/permission. Use --stage fetch to fetch available sources first.')
    if a.stage in ('fetch','all'):
        import_shards(a,sets,cfg);fetch(a,sets,recipes)
    if a.stage in ('build','all'):build(a,sets,cfg)
    if a.stage in ('verify','all'):verify(a,sets,cfg)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr);raise SystemExit(2)
