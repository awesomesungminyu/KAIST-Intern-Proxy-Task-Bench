"""Time-bounded one-seed sCIFAR and ETT baselines; never tune on test data."""
import argparse
import hashlib
import json
import os
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset

ROOT=Path(__file__).resolve().parent
SEED=456
CHANNELS=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']


def utc(): return datetime.now(timezone.utc).isoformat()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def checkpoint(path,value):
    temp=path.with_suffix('.tmp'); torch.save(value,temp); temp.replace(path)


class Residual1D(nn.Module):
    def __init__(self,dilation,width=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation,bias=False),
            nn.BatchNorm1d(width),nn.ReLU(),nn.Dropout(0.1),
            nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation,bias=False),nn.BatchNorm1d(width))
    def forward(self,x): return F.relu(x+self.net(x))


class SequentialCIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(3,32,1)
        self.blocks=nn.Sequential(*(Residual1D(d) for d in [1,4,16,64]))
        self.head=nn.Linear(64,10)
    def forward(self,x):
        if tuple(x.shape[1:])!=(3,1024): raise ValueError(f'Expected RGB sequence [B,3,1024], got {x.shape}')
        z=self.blocks(self.stem(x))
        return self.head(torch.cat([z.mean(-1),z.amax(-1)],dim=-1))


class DecompositionLinear(nn.Module):
    """Independent shared-channel DLinear implementation, moving-average kernel 25."""
    def __init__(self):
        super().__init__()
        self.seasonal=nn.Linear(96,96)
        self.trend=nn.Linear(96,96)
    def forward(self,x):
        if tuple(x.shape[1:])!=(96,7): raise ValueError('Expected [B,96,7]')
        z=x.transpose(1,2)
        trend=F.avg_pool1d(F.pad(z,(12,12),mode='replicate'),25,stride=1)
        return (self.seasonal(z-trend)+self.trend(trend)).transpose(1,2)


class ForecastWindows(Dataset):
    def __init__(self,values,target_start,target_end):
        self.values=values
        self.first=max(96,target_start)
        self.end=target_end
        assert self.end-self.first>=96
    def __len__(self): return self.end-self.first-96+1
    def __getitem__(self,i):
        t=self.first+i
        return self.values[t-96:t],self.values[t:t+96]


def cifar_data(run):
    from torchvision.datasets import CIFAR10
    data=CIFAR10(str(ROOT/'data/cifar'),train=True,download=False)
    test=CIFAR10(str(ROOT/'data/cifar'),train=False,download=False)
    index=torch.randperm(50000,generator=torch.Generator().manual_seed(42))
    ti,vi=index[:45000],index[45000:]
    assert set(ti.tolist()).isdisjoint(vi.tolist()) and len(torch.unique(index))==50000
    np.savez(run/'split_indices.npz',train=ti.numpy(),validation=vi.numpy())
    arr=np.ascontiguousarray(data.data.transpose(0,3,1,2).reshape(-1,3,1024))
    assert np.array_equal(arr[123].reshape(3,32,32).transpose(1,2,0),data.data[123])
    # Estimate channel statistics from the training subset only, in bounded chunks.
    total=np.zeros(3,dtype=np.float64); squared=total.copy(); count=0
    for ids in ti.split(1000):
        v=arr[ids.numpy()].astype(np.float64)/255
        total+=v.sum(axis=(0,2)); squared+=(v*v).sum(axis=(0,2)); count+=v.shape[0]*1024
    mean=total/count; std=np.sqrt(squared/count-mean**2)
    assert (std>0).all()
    full=TensorDataset(torch.from_numpy(arr),torch.tensor(data.targets))
    test_arr=np.ascontiguousarray(test.data.transpose(0,3,1,2).reshape(-1,3,1024))
    sets=[Subset(full,ti.tolist()),Subset(full,vi.tolist()),TensorDataset(torch.from_numpy(test_arr),torch.tensor(test.targets))]
    metadata=dict(task='sCIFAR10',input_shape=[3,1024],layout='1024 pixels in row-major order; three RGB channels per step',
        train_examples=45000,validation_examples=5000,test_examples=10000,split_seed=42,
        train_only_channel_mean=mean.tolist(),train_only_channel_std=std.tolist(),augmentation='none',
        sequence_length_at_all_convolution_layers=1024,pretrained=False,
        sequence_format_reference='https://github.com/state-spaces/s4/blob/main/example.py',
        qualification='Small 1D residual CNN pilot; not S4 or LRA reproduction, and no architecture-ranking claim.')
    return sets,metadata,(mean,std),SequentialCIFARCNN()


def ett_data(name,run):
    p=ROOT/'data'/(name+'.csv'); df=pd.read_csv(p)
    assert list(df.columns)==['date']+CHANNELS
    values=df[CHANNELS].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    factor=1 if name=='ETTh1' else 4
    boundaries=[12*30*24*factor,16*30*24*factor,20*30*24*factor]
    assert len(values)>=boundaries[-1]
    mean=values[:boundaries[0]].mean(0); std=values[:boundaries[0]].std(0,ddof=0)
    assert (std>0).all()
    scaled=(values-mean)/std
    assert np.abs(scaled[:boundaries[0]].mean(0)).max()<1e-9
    scaled=torch.tensor(scaled,dtype=torch.float32)
    sets=[ForecastWindows(scaled,96,boundaries[0]),ForecastWindows(scaled,boundaries[0],boundaries[1]),
          ForecastWindows(scaled,boundaries[1],boundaries[2])]
    intervals=[]
    for ds in sets:
        first=ds.first; last=ds.first+len(ds)-1
        assert first>=96 and last+96==ds.end
        x,y=ds[0]; assert x.shape==(96,7) and y.shape==(96,7)
        assert torch.equal(x,scaled[first-96:first]) and torch.equal(y,scaled[first:first+96])
        intervals.append([first,last+96])
    assert intervals[0][1]==intervals[1][0] and intervals[1][1]==intervals[2][0]
    metadata=dict(task=name,channels=CHANNELS,input_steps=96,prediction_steps=96,features='M: all 7 channels -> all 7 channels',
        input_hours=96/factor,prediction_hours=96/factor,frequency_minutes=60//factor,
        csv_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),split_row_boundaries=boundaries,
        target_row_intervals_half_open=intervals,train_examples=len(sets[0]),validation_examples=len(sets[1]),test_examples=len(sets[2]),
        scaler_fit_rows=[0,boundaries[0]],scaler_mean=mean.tolist(),scaler_std=std.tolist(),
        scaling='train-only per-channel standardization, ddof=0; metrics in standardized units',
        history_policy='past context may cross the previous split; future targets remain inside their own split',
        split_reference='https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py',
        model_reference='https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py',
        model='DLinear, shared-channel seasonal/trend linear heads, moving-average kernel 25, default Linear initialization',
        qualification='One-seed, 96->96 baseline only; not a full forecasting sweep or proof of negative-control behavior.')
    return sets,metadata,None,DecompositionLinear()


def batch_to_gpu(batch,stats):
    x,y=[z.cuda(non_blocking=True) for z in batch]
    if stats is not None:
        mean,std=stats
        x=(x.float()/255-mean)/std
    return x,y


@torch.inference_mode()
def evaluate(model,loader,classification,stats,with_persistence=False):
    model.eval(); n=0; loss_sum=0.; absolute=0.; denom=0
    matrix=torch.zeros(10,10,dtype=torch.long,device='cuda') if classification else None
    channel_se=torch.zeros(7,dtype=torch.float64,device='cuda')
    channel_ae=channel_se.clone(); baseline_se=channel_se.clone(); baseline_ae=channel_se.clone()
    for batch in loader:
        x,y=batch_to_gpu(batch,stats); pred=model(x)
        if not torch.isfinite(pred).all(): raise FloatingPointError('Nonfinite predictions')
        n+=len(y)
        if classification:
            loss_sum+=F.cross_entropy(pred,y,reduction='sum').item()
            matrix+=torch.bincount(y*10+pred.argmax(-1),minlength=100).reshape(10,10)
        else:
            error=(pred-y).double()
            channel_se+=(error**2).sum((0,1));channel_ae+=error.abs().sum((0,1));denom+=len(y)*96
            if with_persistence:
                naive=(x[:,-1:,:].expand_as(y)-y).double()
                baseline_se+=(naive**2).sum((0,1));baseline_ae+=naive.abs().sum((0,1))
    assert n==len(loader.dataset)
    if classification:
        mat=matrix.cpu();acc=mat.diag().double()/mat.sum(1).clamp_min(1)
        return dict(loss=loss_sum/n,accuracy=mat.diag().sum().item()/n,balanced_accuracy=acc.mean().item(),
                    examples=n,per_class_accuracy=acc.tolist(),confusion_matrix=mat.tolist())
    result=dict(mse=(channel_se/denom).mean().item(),mae=(channel_ae/denom).mean().item(),examples=n,
        scalar_targets=denom*7,per_channel_mse=dict(zip(CHANNELS,(channel_se/denom).tolist())),
        per_channel_mae=dict(zip(CHANNELS,(channel_ae/denom).tolist())))
    if with_persistence:
        result['persistence']=dict(mse=(baseline_se/denom).mean().item(),mae=(baseline_ae/denom).mean().item(),
            per_channel_mse=dict(zip(CHANNELS,(baseline_se/denom).tolist())))
    return result


def run(args):
    root=args.run;root.mkdir(parents=True,exist_ok=False)
    write(root/'progress.json',dict(status='preparing',updated_utc=utc()))
    torch.set_num_threads(2)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    classification=args.task=='scifar'
    sets,metadata,stats,model=cifar_data(root) if classification else ett_data(args.task,root)
    if stats is not None:stats=tuple(torch.tensor(v,dtype=torch.float32,device='cuda').view(1,3,1) for v in stats)
    batch_size=128 if classification else 256
    loaders=[DataLoader(ds,batch_size=batch_size,shuffle=i==0,pin_memory=True,num_workers=0,
                       generator=torch.Generator().manual_seed(SEED)) for i,ds in enumerate(sets)]
    model=model.cuda();torch.cuda.reset_peak_memory_stats()
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01) if classification else torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs) if classification else None
    config=dict(metadata,seed=SEED,epochs_max=args.epochs,batch_size=batch_size,
        optimizer='AdamW' if classification else 'Adam',learning_rate=0.001,weight_decay=0.01 if classification else 0.,
        lr_schedule='cosine' if classification else 'constant',early_stopping_patience=6 if classification else 3,
        gradient_clip=1.,precision='FP32',parameters=sum(p.numel() for p in model.parameters()),
        torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_policy='Final selected checkpoint once; no test-set model selection',
        global_deadline_unix=args.deadline,task_budget_seconds=args.budget,started_utc=utc())
    write(root/'config.json',config)
    started=time.time();budget_end=min(args.deadline-120,started+args.budget)
    best=None;best_epoch=0;stale=0;history=[];fully_completed=0;stop_reason='max_epochs';global_step=0
    for epoch in range(1,args.epochs+1):
        model.train();begin=time.time();loss_sum=0.;n=0;correct=0;cut_short=False
        for bi,batch in enumerate(loaders[0],1):
            if time.time()>=budget_end:
                cut_short=True;stop_reason='time_budget';break
            x,y=batch_to_gpu(batch,stats);opt.zero_grad(set_to_none=True)
            pred=model(x)
            loss=F.cross_entropy(pred,y) if classification else F.mse_loss(pred,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite train loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            global_step+=1;n+=len(y);loss_sum+=loss.item()*len(y)
            if classification:correct+=(pred.argmax(-1)==y).sum().item()
            if bi==1 or bi%100==0:
                write(root/'progress.json',dict(status='training',updated_utc=utc(),epoch=epoch,batch=bi,
                    batches_per_epoch=len(loaders[0]),global_step=global_step,running_loss=loss_sum/n))
        if n==0:
            stop_reason='time_budget';break
        val=evaluate(model,loaders[1],classification,stats)
        score=val['accuracy'] if classification else val['mse']
        improved=best is None or (score>best if classification else score<best)
        row=dict(epoch=epoch,complete_epoch=not cut_short,train_examples_seen=n,global_step=global_step,
            train_loss=loss_sum/n,validation=val,seconds=time.time()-begin,learning_rate=opt.param_groups[0]['lr'])
        if classification:row['train_accuracy']=correct/n
        history.append(row)
        if not cut_short:fully_completed+=1
        with (root/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if scheduler:scheduler.step()
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config=config,validation=val)
        if improved:
            best=score;best_epoch=epoch;stale=0;checkpoint(root/'best.pt',state)
        else:stale+=1
        checkpoint(root/'last.pt',state)
        write(root/'progress.json',dict(status='training',updated_utc=utc(),epoch=epoch,best_epoch=best_epoch,
                                       best_validation_score=best,last_epoch_seconds=row['seconds']))
        print('EPOCH',args.task,json.dumps({k:v for k,v in row.items() if k!='validation'}),'val_score',score,flush=True)
        if cut_short:break
        if stale>=config['early_stopping_patience']:
            stop_reason='early_stopping';break
    if not (root/'best.pt').exists():raise TimeoutError('No trained checkpoint before deadline')
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
    write(root/'progress.json',dict(status='final_test',updated_utc=utc(),best_epoch=best_epoch))
    test=evaluate(model,loaders[2],classification,stats,with_persistence=not classification)
    result=dict(status='completed_time_limited' if stop_reason=='time_budget' else 'completed',config=config,
        stop_reason=stop_reason,epochs_evaluated=len(history),fully_completed_epochs=fully_completed,best_epoch=best_epoch,
        best_validation=saved['validation'],test=test,test_evaluation_calls=1,
        elapsed_training_evaluation_seconds=time.time()-started,peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2,
        completed_utc=utc(),best_checkpoint=str(root/'best.pt'))
    write(root/'result.json',result)
    write(root/'progress.json',dict(status=result['status'],updated_utc=utc(),stop_reason=stop_reason,
        best_epoch=best_epoch,test_accuracy=test.get('accuracy'),test_mse=test.get('mse')))
    print('COMPLETE',args.task,json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['scifar','ETTh1','ETTm1'],required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--budget',type=int,required=True)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();was_existing=a.run.exists()
    try:run(a)
    except Exception as e:
        if a.run.exists() and not was_existing:
            error=dict(status='failed',error=repr(e),traceback=traceback.format_exc(),updated_utc=utc())
            write(a.run/'failure.json',error);write(a.run/'progress.json',error)
        raise
