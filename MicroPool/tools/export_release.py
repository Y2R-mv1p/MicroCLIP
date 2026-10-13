#!/usr/bin/env python3
"""Maintainer only: freeze evaluated shards, without copying image bytes.

Inputs are explicit; consumers only need the generated manifests and eval.py.
The shards, rather than resampling scripts or older TSVs, are authoritative.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def sanitize(text):
    text = re.sub(r'\s+', '_', text.strip().replace(' of ', '_of_'))
    return re.sub('_+', '_', re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text)).strip('._ ') or 'unnamed'


def write_gz(path, rows):
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as f:
        for row in rows:
            f.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--author-root', type=Path, required=True)
    ap.add_argument('--old-recipe', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    root = a.author_root
    public_names = {
        'iid': 'iid',
        'partial_ood': 'compositional_shift',
        'cell_ood': 'cell_line_shift',
        'structure_ood': 'organelle_set_shift',
        'lmc_source_shift': 'source_shift',
    }
    specs = {s: root/'evaluation'/public_names[s]/'shard-000000.tar'
             for s in ['iid','partial_ood','cell_ood','structure_ood','lmc_source_shift']}
    specs.update({f'retrieval{n}': root/'evaluation'/f'retrieval_{n}'/'shard-000000.tar' for n in [1000]})
    def extract(item):
        split, path = item
        records = collections.OrderedDict()
        with tarfile.open(path, 'r:') as t:
            for m in t:
                if not m.isfile():
                    raise ValueError(f'Unexpected member {m.name}')
                key, ext = m.name.rsplit('.',1)
                r = records.setdefault(key, {'key':key, 'split':split})
                b = t.extractfile(m).read()
                if ext in ('jpg','png'):
                    r.update(image_sha256=hashlib.sha256(b).hexdigest(), image_bytes=len(b), extension=ext)
                elif ext == 'txt': r['text'] = b.decode()
                elif ext == 'json': r['metadata'] = json.loads(b)
                else: raise ValueError(m.name)
        for r in records.values():
            assert all(k in r for k in ['image_sha256','text','metadata'])
        print(f'Frozen {split}: {len(records)} records', flush=True)
        return split, list(records.values())
    with ThreadPoolExecutor(max_workers=3) as pool:
        sets = dict(pool.map(extract, specs.items()))
    wanted = {Path(r['metadata']['filepath']).name for split,rs in sets.items()
              if split not in ('source_shift','lmc_source_shift') for r in rs}
    origin = {}; counts = collections.Counter()
    srcs = list(csv.DictReader((a.old_recipe/'config/sources.tsv').open(), delimiter='\t'))
    for s in sorted(srcs, key=lambda r:int(r['order'])):
        with gzip.open(a.old_recipe/'manifests/labels'/f"{s['id']}.tsv.gz", 'rt') as f:
            for line in f:
                rel, _, title = line.rstrip('\n').partition('\t')
                folder = sanitize(title); counts[folder] += 1
                bn = f'{folder}_{counts[folder]}.jpg'
                if bn in wanted: origin[bn] = (s['id'], rel, folder)
    if wanted - origin.keys(): raise ValueError(f'Missing provenance: {wanted-origin.keys()}')
    external = {}
    for sid, manifest in [('cpg0046',root/'dataset/cpg0046_/cpg0046_external_manifest.csv'),
                          ('idr0080',root/'dataset/idr0080_external_source_shift/idr0080_external_manifest.csv')]:
        for r in csv.DictReader(manifest.open()):
            rel = str(Path(r['relative_path']).with_suffix('.png'))
            image_rel = rel.replace('images/','images_8bit/',1)
            url = r.get('source_url') or r['s3_uri'].replace('s3://cellpainting-gallery/','https://cellpainting-gallery.s3.amazonaws.com/')
            external[(sid,Path(image_rel).name)] = dict(source=sid, source_path=r['relative_path'],
                image_path=f'{sid}/{image_rel}', transform='percentile_png', url=url)
    lmc = root/'dataset/lightmycells_external_benchmark_builder/lightmycells_external_benchmark/lightmycells_external/final_manifest.csv'
    for r in csv.DictReader(lmc.open()):
        external[('lightmycells',Path(r['local_png']).name)] = dict(source='lightmycells',source_path=r['local_tiff'],
            image_path='lightmycells/'+r['local_png'],transform='percentile_png',url=r['url'],raw_sha256=r['sha256'])
    summary = {}; source_counts = collections.defaultdict(collections.Counter)
    for split, records in sets.items():
        for r in records:
            md = r['metadata']; fp = md['filepath']; bn = Path(fp).name
            if split in ('source_shift','lmc_source_shift'):
                sid = 'lightmycells' if 'lightmycells' in fp else ('cpg0046' if 'cpg0046' in fp else 'idr0080')
                r.update(external[(sid,bn)])
            else:
                sid, rel, folder = origin[bn]
                r.update(source=sid, source_path=rel, image_path=f'fluogen/{folder}/{bn}',transform='minmax_jpeg')
            md['filepath'] = r['image_path']
            source_counts[split][r['source']] += 1
        write_gz(a.out/'manifests'/f'{split}.jsonl.gz',records)
        if split.startswith('retrieval'):
            dest=f'evaluation/retrieval_{split[9:]}/shard-000000.tar'
        else:
            public_names = {
                'iid': 'iid',
                'partial_ood': 'compositional_shift',
                'cell_ood': 'cell_line_shift',
                'structure_ood': 'organelle_set_shift',
                'lmc_source_shift': 'source_shift',
            }
            dest=f'evaluation/{public_names[split]}/shard-000000.tar'
        classes=collections.Counter(r['text'] for r in records) if not split.startswith('retrieval') else {}
        summary[split] = dict(records=len(records),classes=dict(sorted(classes.items())),sources=dict(source_counts[split]),
                              output=dest, manifest_sha256=hashlib.sha256((a.out/'manifests'/f'{split}.jsonl.gz').read_bytes()).hexdigest())
    (a.out/'config/release.json').write_text(json.dumps(dict(schema_version=1,release='eval-v1',
        baseline='Frozen from author evaluation shards; no resampling',splits=summary),ensure_ascii=False,indent=2)+'\n')
    with (a.out/'manifests/source_counts.tsv').open('w') as f:
        f.write('split\tsource\trecords\n')
        for split,c in source_counts.items():
            for sid,n in sorted(c.items()):f.write(f'{split}\t{sid}\t{n}\n')
    print('All images mapped to source; image bytes NOT exported.',flush=True)

if __name__ == '__main__': main()
