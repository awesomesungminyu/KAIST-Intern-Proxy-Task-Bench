"""Resume only the missing canonical CIFAR archive bytes over verified HTTPS."""
import hashlib
import json
import time
import traceback
import urllib.request
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import extract_archive

ROOT=Path(__file__).resolve().parent
URL='https://huggingface.co/datasets/liangnanying/cifar-10-python/resolve/main/cifar-10-python.tar.gz'
TOTAL=170498071


def main():
    folder=ROOT/'data/cifar';p=folder/'cifar-10-python.tar.gz'
    assert p.exists() and p.stat().st_size<=TOTAL
    while p.stat().st_size<TOTAL:
        start=p.stat().st_size;end=min(start+4*1024*1024,TOTAL)-1
        for attempt in range(3):
            try:
                req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=60) as response:
                    assert response.status==206
                    assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{TOTAL}'
                    chunk=response.read()
                assert len(chunk)==end-start+1
                with p.open('ab') as f:f.write(chunk)
                print('RESUME_BYTES',p.stat().st_size,TOTAL,flush=True)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    md5=hashlib.md5(p.read_bytes()).hexdigest()
    assert md5==CIFAR10.tgz_md5,(md5,CIFAR10.tgz_md5)
    extract_archive(str(p),str(folder))
    train=CIFAR10(str(folder),train=True,download=False);test=CIFAR10(str(folder),train=False,download=False)
    assert len(train)==50000 and len(test)==10000
    meta=json.loads((ROOT/'prepare_status.json').read_text())
    meta['files']['cifar10']=dict(source=URL,canonical_source=CIFAR10.url,archive_md5_expected=CIFAR10.tgz_md5,
        archive_md5_verified=md5,archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        torchvision_integrity_verified=True,training_examples=len(train),test_examples=len(test))
    meta.update(status='completed',finished_unix=time.time())
    tmp=ROOT/'prepare_status.tmp';tmp.write_text(json.dumps(meta,indent=2));tmp.replace(ROOT/'prepare_status.json')
    print('CIFAR_READY',json.dumps(meta['files']['cifar10']),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'resume_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
