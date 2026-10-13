#!/usr/bin/env python3
"""Maintainer: resolve selected records against existing source recipe and official indexes."""
import argparse,csv,gzip,json,re,importlib.util,hashlib
from pathlib import Path
from urllib.parse import quote
from export_release import write_gz

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-recipe',type=Path,required=True);p.add_argument('--research',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);a=p.parse_args()
 repo=Path(__file__).resolve().parents[1]
 spec=importlib.util.spec_from_file_location('old',a.old_recipe/'tools/download.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 plates={}
 for line in (a.research/'idr88_plates').read_text().splitlines():
  parts=line.split('/');folder=parts[-3];plates[folder]='/'.join(parts[4:-1])
 assays={}
 for exp in ['A','B']:
  path=a.raw_root/f'IDR_raw/idr0050-springer-cytoskeletalsystems/20181106-ftp/experiment{exp}/idr0050-experiment{exp}-assays.txt'
  for row in csv.DictReader(path.open(encoding='latin1'),delimiter='\t'):
   assays[(exp,Path(row['Image File']).stem)]=row
 recipes={}
 for f in sorted((repo/'manifests').glob('*.jsonl.gz')):
  for line in gzip.open(f,'rt'):
   r=json.loads(line);sid=r['source'];rel=r['source_path'];key=sid+'/'+rel
   if key in recipes:continue
   x=dict(source=sid,source_path=rel,kind='manual')
   if r.get('url'):x.update(kind='http',url=r['url'],raw_sha256=r.get('raw_sha256'))
   elif sid=='idr088':
    folder=rel.split('/')[2];image=Path(rel).name
    x.update(kind='http',url='https://ftp.ebi.ac.uk/pub/databases/IDR/'+quote(plates[folder]+'/'+image,safe='/'))
   elif sid=='cpjump1':
    suffix=rel.removeprefix('CPJUMP1/images/')
    x.update(kind='http',url='https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/'+quote(suffix,safe='/'))
   elif sid=='idr050':
    exp=rel.split('/')[3][-1];channel=rel.split('/')[4];bn=Path(rel).stem
    if exp=='B':
     base,plane=re.match(r'(.*)_'+re.escape(channel)+r'_channel(\d+)$',bn).groups();plane=int(plane)
    else:base=bn.removesuffix('_'+channel);plane=0
    row=assays[(exp,base)];channels=[c.split(': ',1)[1] for c in row['Channels'].split(', ')]
    remote=f"idr0050-springer-cytoskeletalsystems/20181106-ftp/experiment{exp}/ImagesOfExperiment{exp}/{row['Image File']}"
    x.update(kind='http',url='https://ftp.ebi.ac.uk/pub/databases/IDR/'+quote(remote,safe='/'),
     extract=dict(mode='idr50_'+exp.lower(),channel=channels.index(channel),n_channels=len(channels),plane=plane))
   elif sid=='rxrx1':
    x.update(kind='zip',url='https://storage.googleapis.com/rxrx/rxrx1/rxrx1-images.zip',member='rxrx1/'+rel.removeprefix('RxRx/rxrx1/'))
   elif sid=='bbbc_ht29':
    plate=rel.split('/')[-3];bn=Path(rel).name
    m=re.search(r'_frame(\d+)_channel(\d+)\.tif$',bn);frame,channel=map(int,m.groups())
    well_idx,field=divmod(frame-1,6);well='ABCDEFGHIJKLMNOP'[well_idx//24]+f'{well_idx%24+1:02d}'
    x.update(kind='zip',url=f'https://data.broadinstitute.org/bbbc/BBBC017/BBBC017_v1_images_{plate}.zip',
     member_regex=f'^{re.escape(plate)}/.*_{well}f{field+1:02d}d{channel-1}\\.DIB$',decode='ht29')
   else:
    s=old.SOURCES.get(sid,{})
    if s.get('kind')=='file':
     url=s['rule'](rel)
     if url:x.update(kind='http',url=url)
    for g in s.get('groups',[]):
     prefix=g['member_prefix']
     if rel.startswith(g.get('match',prefix)):
      x.update(kind='tar' if g.get('format')=='tar' else 'zip',url=g['url'],member=g.get('member_root','')+rel[len(prefix):]);break
   recipes[key]=x
 write_gz(repo/'config/recipes.jsonl.gz',recipes.values())
 print('Recipes',len(recipes))
 from collections import Counter
 print(Counter((x['source'],x['kind']) for x in recipes.values()))
if __name__=='__main__':main()
