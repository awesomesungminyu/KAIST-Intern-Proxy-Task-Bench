"""Download official CIFAR-10 and ETT files without changing installed packages."""
import hashlib
import json
import socket
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import pandas as pd
from torchvision.datasets import CIFAR10

# Same canonical archive; torchvision verifies canonical archive and batch MD5s.
# This mirror is also used by the Kornia official ZCA tutorial.
CIFAR10.url='https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz'

ROOT=Path(__file__).resolve().parent
socket.setdefaulttimeout(30)


def main():
    data=ROOT/'data'; data.mkdir(exist_ok=True)
    meta={'started_unix':time.time(),'files':{},'status':'downloading'}
    (ROOT/'prepare_status.json').write_text(json.dumps(meta,indent=2))
    # Small ETT files first, so they remain usable even if CIFAR download fails.
    try:
        req=urllib.request.Request('https://api.github.com/repos/zhouhaoyi/ETDataset/commits/main',headers={'User-Agent':'baseline-pilot'})
        with urllib.request.urlopen(req,timeout=25) as response:
            revision=json.load(response)['sha']
    except Exception:
        revision='main'
    for name,factor in [('ETTh1',1),('ETTm1',4)]:
        url=f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{revision}/ETT-small/{name}.csv'
        p=data/(name+'.csv')
        if not p.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=30) as response:
                        payload=response.read()
                    p.write_bytes(payload)
                    break
                except Exception:
                    if attempt==2: raise
                    time.sleep(2)
        df=pd.read_csv(p)
        assert list(df.columns)==['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
        assert len(df)>=20*30*24*factor and df.iloc[:,1:].notna().all().all()
        dates=pd.to_datetime(df['date'])
        assert dates.is_monotonic_increasing and dates.is_unique
        assert (dates.diff().dropna()==pd.Timedelta(minutes=60//factor)).all()
        meta['files'][name]={'source':url,'revision':revision,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                             'rows':len(df),'columns':list(df.columns),'frequency_minutes':60//factor}
        (ROOT/'prepare_status.json').write_text(json.dumps(meta,indent=2))
        print('ETT_READY',name,len(df),flush=True)
    for attempt in range(3):
        try:
            train=CIFAR10(root=str(data/'cifar'),train=True,download=True)
            test=CIFAR10(root=str(data/'cifar'),train=False,download=False)
            break
        except Exception:
            if attempt==2: raise
            time.sleep(2)
    assert train.data.shape==(50000,32,32,3) and test.data.shape==(10000,32,32,3)
    archive=data/'cifar'/CIFAR10.filename
    meta['files']['cifar10']={'source':CIFAR10.url,'archive_md5_expected':CIFAR10.tgz_md5,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'torchvision_integrity_verified':True,'training_examples':len(train),'test_examples':len(test)}
    meta.update(status='completed',finished_unix=time.time())
    (ROOT/'prepare_status.json').write_text(json.dumps(meta,indent=2))
    print('DATA_READY',json.dumps(meta),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'prepare_failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2))
        raise
