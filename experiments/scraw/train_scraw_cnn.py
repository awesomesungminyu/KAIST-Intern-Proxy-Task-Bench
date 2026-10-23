"""One-seed, 12-class SC-Raw M5-style CNN pilot with validation-only selection."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from scraw_bootstrap import LABELS, RawDataset

HERE = Path(__file__).resolve().parent
REFERENCE = 'https://docs.pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


class M5CNN(nn.Module):
    """Tutorial-style M5-fc; 16 kHz input, four internal pooling stages."""
    def __init__(self):
        super().__init__()
        blocks = []
        for cin, cout, kernel, stride in [(1,32,80,16), (32,32,3,1), (32,64,3,1), (64,64,3,1)]:
            blocks.append(nn.Sequential(nn.Conv1d(cin,cout,kernel,stride=stride),
                                       nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(4)))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(64, len(LABELS))

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (1,16000):
            raise ValueError(f'Expected [batch,1,16000], got {tuple(x.shape)}')
        for layer in self.blocks:
            x = layer(x)
        return self.head(x.mean(dim=-1))


def prepare_cache(records, split, cache):
    cache.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256((HERE/'scraw_bootstrap.py').read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps([source_hash, records],sort_keys=True).encode()).hexdigest()
    meta_path = cache / f'{split}.json'
    x_path, y_path = cache/f'{split}_x.npy', cache/f'{split}_y.npy'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        assert meta['record_hash']==digest and meta['shape']==[len(records),16000]
        x=np.load(x_path,mmap_mode='r'); y=np.load(y_path,mmap_mode='r')
        assert x.shape==(len(records),16000) and x.dtype==np.float32
        assert y.shape==(len(records),) and y.dtype==np.int64
        print('CACHE_REUSE',split,len(records),flush=True)
        return x_path,y_path
    print('CACHE_BUILD',split,len(records),flush=True)
    ds=RawDataset(records)
    x=np.lib.format.open_memmap(x_path,mode='w+',dtype=np.float32,shape=(len(ds),16000))
    y=np.empty(len(ds),dtype=np.int64)
    for i in range(len(ds)):
        item,target=ds[i]
        x[i]=item.numpy()[0]
        y[i]=target
        if (i+1)%5000==0: print('CACHE_PROGRESS',split,i+1,flush=True)
    x.flush()
    del x
    np.save(y_path,y)
    write_json(meta_path,dict(record_hash=digest,shape=[len(records),16000],dtype='float32',
                              bootstrap_sha256=source_hash,created_utc=now()))
    return x_path,y_path


class CachedRaw(Dataset):
    def __init__(self, paths):
        self.x=np.load(paths[0],mmap_mode='r')
        self.y=np.load(paths[1],mmap_mode='r')
    def __len__(self): return len(self.y)
    def __getitem__(self,i):
        return torch.from_numpy(np.array(self.x[i],copy=True)).unsqueeze(0), int(self.y[i])


def wait_for_gpu(run, timeout):
    start=time.monotonic()
    consecutive=0
    while True:
        p=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        mem,util=[int(v.strip()) for v in p.stdout.strip().split(',')]
        consecutive=consecutive+1 if mem<=512 and util<=10 else 0
        write_json(run/'progress.json',dict(status='waiting_for_gpu',updated_utc=now(),
                    gpu_memory_used_mib=mem,gpu_utilization_percent=util,
                    waited_seconds=round(time.monotonic()-start),idle_checks=consecutive))
        if consecutive>=2: return
        if time.monotonic()-start>timeout: raise TimeoutError('GPU remained busy beyond queue deadline')
        time.sleep(15)


@torch.inference_mode()
def evaluate(model,loader):
    model.eval()
    loss_sum=0.; n=0
    confusion=torch.zeros(len(LABELS),len(LABELS),dtype=torch.long,device='cuda')
    for x,y in loader:
        x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
        logits=model(x)
        loss=nn.functional.cross_entropy(logits,y,reduction='sum')
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite evaluation loss')
        loss_sum+=loss.item(); n+=len(y)
        confusion+=torch.bincount(y*len(LABELS)+logits.argmax(-1),minlength=len(LABELS)**2).reshape(len(LABELS),len(LABELS))
    matrix=confusion.cpu()
    per_class=matrix.diag().double()/matrix.sum(1).clamp_min(1)
    assert n==len(loader.dataset)
    return dict(loss=loss_sum/n,accuracy=matrix.diag().sum().item()/n,
                balanced_accuracy=per_class.mean().item(),examples=n,
                per_class_accuracy=dict(zip(LABELS,per_class.tolist())),confusion_matrix=matrix.tolist())


def main(args):
    run=args.run
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest_path=HERE/'check-20261016-01/manifest.json'
    audit=json.loads((manifest_path.parent/'data_audit.json').read_text())
    assert audit['status']=='passed' and audit['speech_paths_and_speakers_disjoint']
    manifest=json.loads(manifest_path.read_text())
    assert manifest['labels']==LABELS
    records=manifest['splits']
    config=dict(task='SC-Raw-12',model='M5-style CNN-fc',seed=args.seed,epochs=args.epochs,
        early_stopping_patience=5,batch_size=128,optimizer='Adam',learning_rate=0.001,
        weight_decay=0.0001,lr_schedule='cosine over configured maximum epochs',gradient_clip=1.0,
        precision='FP32',input_shape=[1,16000],sample_rate=16000,augmentation='none',
        waveform_scaling='PCM16 / 32768; no per-example normalization',
        internal_sequence_lengths_after_pool=[249,61,14,3],
        qualification='Small one-seed baseline; internal stride/pooling reduces length. Not evidence for full-length sequence-model rankings.',
        architecture_reference=REFERENCE,reference_differences='16 kHz rather than 8 kHz, 12 classes, raw logits plus cross entropy; independently implemented sequential blocks.',
        data_audit=str(manifest_path.parent/'data_audit.json'),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        bootstrap_sha256=hashlib.sha256((HERE/'scraw_bootstrap.py').read_bytes()).hexdigest(),
        training_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        silence_limitation=audit['silence_policy'],
        test_policy='Only best validation-accuracy checkpoint is evaluated once after training.',
        created_utc=now())
    write_json(run/'config.json',config)
    write_json(run/'progress.json',dict(status='preparing_train_validation_cache',updated_utc=now()))
    cache=HERE/'raw_cache_v1'
    train_paths=prepare_cache(records['training'],'training',cache)
    valid_paths=prepare_cache(records['validation'],'validation',cache)
    if args.prepare_only:
        write_json(run/'progress.json',dict(status='prepared',updated_utc=now()))
        return
    wait_for_gpu(run,args.queue_timeout)
    if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; CPU fallback disabled')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    generator=torch.Generator().manual_seed(args.seed)
    train=DataLoader(CachedRaw(train_paths),batch_size=128,shuffle=True,generator=generator,
                     pin_memory=True,num_workers=0)
    valid=DataLoader(CachedRaw(valid_paths),batch_size=128,shuffle=False,pin_memory=True,num_workers=0)
    model=M5CNN().cuda()
    opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    config.update(parameters=sum(p.numel() for p in model.parameters()),torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(0),cuda_build=torch.version.cuda,training_started_utc=now())
    write_json(run/'config.json',config)
    torch.cuda.reset_peak_memory_stats()
    best=-1.; best_epoch=0; stale=0; history=[]; global_step=0
    start=time.monotonic()
    print('TRAIN_START',json.dumps(config),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train(); epoch_start=time.monotonic(); loss_sum=0.; correct=0; n=0
        lr=opt.param_groups[0]['lr']
        for batch,(x,y) in enumerate(train,1):
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step(); global_step+=1
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(-1)==y).sum().item(); n+=len(y)
            if batch%50==0 or batch==1:
                write_json(run/'progress.json',dict(status='training',updated_utc=now(),epoch=epoch,
                    batch=batch,batches_per_epoch=len(train),global_step=global_step,
                    running_train_loss=loss_sum/n,running_train_accuracy=correct/n))
                print('BATCH',epoch,batch,len(train),'loss',round(loss_sum/n,4),flush=True)
        metrics=evaluate(model,valid)
        row=dict(epoch=epoch,global_step=global_step,train_loss=loss_sum/n,train_accuracy=correct/n,
                 validation_loss=metrics['loss'],validation_accuracy=metrics['accuracy'],
                 validation_balanced_accuracy=metrics['balanced_accuracy'],
                 seconds=time.monotonic()-epoch_start,learning_rate=lr)
        history.append(row)
        with (run/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        schedule.step()
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
                        epoch=epoch,config=config,labels=LABELS,validation=metrics,
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
                        shuffle_rng=generator.get_state())
        if metrics['accuracy']>best:
            best=metrics['accuracy']; best_epoch=epoch; stale=0
            save_checkpoint(run/'best.pt',checkpoint)
        else: stale+=1
        checkpoint.update(best_accuracy=best,best_epoch=best_epoch,stale_epochs=stale)
        save_checkpoint(run/'last.pt',checkpoint)
        print('EPOCH',json.dumps(row),'best_epoch',best_epoch,flush=True)
        write_json(run/'progress.json',dict(status='training',updated_utc=now(),last_epoch=row,
                    best_validation_accuracy=best,best_epoch=best_epoch))
        if stale>=5:
            print('EARLY_STOP',epoch,flush=True)
            break
    selected=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    write_json(run/'progress.json',dict(status='final_test_evaluation',updated_utc=now(),best_epoch=best_epoch))
    test_paths=prepare_cache(records['testing'],'testing',cache)
    test=DataLoader(CachedRaw(test_paths),batch_size=128,shuffle=False,pin_memory=True,num_workers=0)
    test_metrics=evaluate(model,test)
    result=dict(status='completed',completed_utc=now(),config=config,epochs_completed=len(history),
        best_epoch=best_epoch,best_validation=selected['validation'],test=test_metrics,
        training_and_final_evaluation_seconds=time.monotonic()-start,
        peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2,
        best_checkpoint=str(run/'best.pt'),last_checkpoint=str(run/'last.pt'),
        selection='maximum validation accuracy; earliest epoch wins exact ties',test_evaluation_calls=1)
    write_json(run/'result.json',result)
    write_json(run/'progress.json',dict(status='completed',updated_utc=now(),best_epoch=best_epoch,
                validation_accuracy=best,test_accuracy=test_metrics['accuracy']))
    print('COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=456)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--queue-timeout',type=int,default=43200)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.epochs<1: parser.error('epochs must be positive')
    args.run.mkdir(parents=True,exist_ok=False)
    try:
        main(args)
    except Exception as e:
        if args.run.exists():
            write_json(args.run/'failure.json',dict(status='failed',updated_utc=now(),
                        error=repr(e),traceback=traceback.format_exc()))
            write_json(args.run/'progress.json',dict(status='failed',updated_utc=now(),error=repr(e)))
        raise
