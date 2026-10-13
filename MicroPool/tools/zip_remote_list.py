#!/usr/bin/env python3
"""Inspect official ZIP members without downloading the entire archive."""
import argparse
import zipfile
from transport import HTTPRangeFile

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('url');p.add_argument('--grep',default='');p.add_argument('--limit',type=int,default=40)
    a=p.parse_args()
    with HTTPRangeFile(a.url) as f, zipfile.ZipFile(f) as z:
        names=[n for n in z.namelist() if a.grep in n]
        print(f'{len(names)} matching members; showing {min(a.limit,len(names))}')
        print('\n'.join(names[:a.limit]))
