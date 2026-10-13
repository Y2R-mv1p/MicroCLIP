"""Bounded, retrying downloads and ZIP range reads. Never disable TLS verification."""
import io
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path

UA = 'MicroPool-eval/1.0 (official-source reconstruction)'


def request(url, headers=None, method=None):
    return urllib.request.Request(url, headers={'User-Agent':UA, **(headers or {})}, method=method)


def download(url, target, expected_sha256=None, retries=3):
    import hashlib
    target = Path(target)
    if target.is_file():
        if expected_sha256 and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
            return target
        if not expected_sha256 and target.stat().st_size:
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name+'.part')
    for attempt in range(retries):
        try:
            h=hashlib.sha256(); size=0
            with urllib.request.urlopen(request(url), timeout=120) as response, tmp.open('wb') as f:
                while True:
                    b=response.read(1024*1024)
                    if not b:break
                    f.write(b);h.update(b);size+=len(b)
                length=response.headers.get('Content-Length')
                if length is not None and size != int(length):raise IOError('Truncated HTTP body')
            if not size:raise IOError('Empty HTTP body')
            if expected_sha256 and h.hexdigest()!=expected_sha256:raise IOError('Source SHA256 mismatch')
            os.replace(tmp,target);return target
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt+1==retries:raise
            time.sleep(2**attempt)


class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP file for zipfile, validating each range before reading its body.

    A server ignoring Range is rejected, rather than accidentally reading 49 GB.
    Blocks are bounded and only the latest block is retained. ETag/Last-Modified
    is used with If-Range so an archive changing during extraction fails closed.
    """
    def __init__(self, url, block_size=8*1024*1024):
        super().__init__();self.url=url;self.pos=0;self.block_size=block_size
        self.start=-1;self.block=b'';self.validator=None
        with urllib.request.urlopen(request(url, method='HEAD'), timeout=120) as r:
            length=r.headers.get('Content-Length')
            if not length:raise IOError('Archive has no Content-Length; provide it locally')
            self.size=int(length);self.validator=r.headers.get('ETag') or r.headers.get('Last-Modified')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self, offset, whence=0):
        pos=offset if whence==0 else (self.pos+offset if whence==1 else self.size+offset)
        if pos<0:raise ValueError('Negative seek')
        self.pos=pos;return pos
    def _block(self, start):
        end=min(self.size-1,start+self.block_size-1)
        headers={'Range':f'bytes={start}-{end}'}
        if self.validator:headers['If-Range']=self.validator
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request(self.url, headers), timeout=120) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                        raise IOError('Range ignored, archive changed, or invalid Content-Range')
                    b=r.read(end-start+2)
                    if len(b)!=end-start+1:raise IOError('Truncated or overlong range')
                self.start=start;self.block=b;return
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    def read(self,n=-1):
        if n is None or n<0:n=max(0,self.size-self.pos)
        n=min(n,max(0,self.size-self.pos));chunks=[]
        while n:
            if not self.start<=self.pos<self.start+len(self.block):self._block(self.pos)
            offset=self.pos-self.start;take=min(n,len(self.block)-offset)
            chunks.append(self.block[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(chunks)


def open_zip(url, archives):
    local=Path(archives)/Path(urllib.request.url2pathname(urllib.parse.urlsplit(url).path)).name
    if local.is_file():return zipfile.ZipFile(local)
    return zipfile.ZipFile(HTTPRangeFile(url))
