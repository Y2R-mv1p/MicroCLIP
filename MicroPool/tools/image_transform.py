"""Versioned author transforms. No resize, crop, randomization or recaptioning."""
from io import BytesIO
from pathlib import Path


def load(path):
    import numpy as np
    import tifffile
    from PIL import Image
    if Path(path).suffix.lower() in ('.tif','.tiff','.lsm'):
        return np.asarray(tifffile.imread(path))
    if Path(path).suffix.lower()=='.czi':
        import czifile
        return czifile.imread(path)
    with Image.open(path) as im:return np.asarray(im)


def grayscale(array):
    import numpy as np
    a=np.squeeze(np.asarray(array))
    if a.ndim==0:return a.reshape(1,1)
    if a.ndim==3 and a.shape[-1] in (3,4):return a[...,:3].mean(axis=-1)
    while a.ndim>2:a=a[0]
    if a.ndim!=2:raise ValueError(f'Unsupported shape {a.shape}')
    return a


def minmax(a):
    import numpy as np
    a=np.asarray(a,dtype=np.float32);finite=np.isfinite(a);out=np.zeros(a.shape,dtype=np.uint8)
    if not finite.any():return out
    values=a[finite];lo,hi=float(values.min()),float(values.max())
    if hi>lo:out[finite]=np.clip((values-lo)/(hi-lo)*255.0,0,255).astype(np.uint8)
    return out


def convert(path, transform, extract=None):
    import numpy as np
    from PIL import Image
    a=load(path)
    if extract:
        mode=extract['mode'];channel=extract['channel'];a=np.asarray(a)
        if mode=='idr50_a':
            a=a[channel,0] if a.ndim==4 else a[channel]
        elif mode=='idr50_b':
            a=a.squeeze();z=extract['plane']
            if Path(path).suffix.lower()!='.lsm' and a.shape[0]==extract['n_channels']:
                a=a[channel] if a.ndim==3 else a[channel,z]
            elif a.ndim==3:
                a=a[channel]
            elif a.ndim==4:a=a[z,channel]
            else:raise ValueError(f'Unsupported IDR0050 shape {a.shape}')
        else:raise ValueError(f'Unknown extraction {mode}')
    if transform=='minmax_jpeg':
        image=Image.fromarray(minmax(grayscale(a)),mode='L');buf=BytesIO();image.save(buf,format='JPEG',quality=95)
    elif transform=='percentile_png':
        a=np.squeeze(a)
        if a.ndim!=2:raise ValueError(f'Expected 2D source, got {a.shape}')
        finite=a[np.isfinite(a)]
        out=np.zeros(a.shape,dtype=np.uint8)
        if finite.size:
            lo,hi=np.percentile(finite,[0.1,99.9])
            if hi>lo:out=(np.clip((a.astype(np.float32)-lo)/(hi-lo),0,1)*255.0+0.5).astype(np.uint8)
        image=Image.fromarray(out,mode='L');buf=BytesIO();image.save(buf,format='PNG',optimize=True)
    else:raise ValueError(f'Unknown transform {transform}')
    return buf.getvalue()
