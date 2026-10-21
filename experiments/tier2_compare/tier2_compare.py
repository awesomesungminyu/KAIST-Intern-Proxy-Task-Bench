"""Preliminary sequential Tier-2 comparisons with separate train/validation clocks."""
import os,time
PROCESS_START=time.perf_counter()
PROCESS_STARTED_UNIX=time.time()
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='6'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,fcntl,hashlib,json,math,random,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils.data import Dataset,DataLoader,TensorDataset,Subset

ROOT=Path(__file__).resolve().parent
DATA=Path('/workspace/exp/2026-10-16-quick-suite/data')
CENTRAL=Path('/workspace/exp/2026-10-12/runs.jsonl')
NOTE='seed 1회, 분산 미추정'
LIMIT_LABEL='예산 제한, 수렴 미도달'
CHANNELS=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']


def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
    q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def sync():torch.cuda.synchronize()


def append(row):
    with CENTRAL.open('a+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        ids={json.loads(x)['run_id'] for x in f if x.strip()}
        if row['run_id'] in ids:raise RuntimeError('Duplicate run_id; record was not appended')
        f.seek(0,2);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        fcntl.flock(f,fcntl.LOCK_UN)


def new_run(base):
    with CENTRAL.open() as f:ids={json.loads(x)['run_id'] for x in f if x.strip()}
    for i in range(1,1000):
        name=base if i==1 else base+f'-r{i}'
        if name in ids:continue
        p=ROOT/'runs'/name
        try:p.mkdir(parents=True,exist_ok=False);return name,p
        except FileExistsError:continue
    raise RuntimeError('Could not allocate unique run ID')


class Windows(Dataset):
    def __init__(self,x,start,end):self.x=x;self.start=max(96,start);self.end=end
    def __len__(self):return self.end-self.start-96+1
    def __getitem__(self,i):
        t=self.start+i;return self.x[t-96:t],self.x[t:t+96]


def data_for(task,run):
    if task!='sCIFAR':
        p=DATA/(task+'.csv');df=pd.read_csv(p);a=df[CHANNELS].to_numpy(np.float64)
        assert list(df.columns)==['date']+CHANNELS and np.isfinite(a).all()
        f=1 if task=='ETTh1' else 4;b=[8640*f,11520*f,14400*f]
        mu=a[:b[0]].mean(0);sd=a[:b[0]].std(0);assert (sd>0).all()
        a=(a-mu)/sd;assert np.abs(a[:b[0]].mean(0)).max()<1e-9
        x=torch.tensor(a,dtype=torch.float32)
        sets=[Windows(x,96,b[0]),Windows(x,b[0],b[1]),Windows(x,b[1],b[2])]
        assert [len(s) for s in sets]==([8449,2785,2785] if f==1 else [34369,11425,11425])
        meta=dict(input_steps=96,prediction_steps=96,features='M: 7 -> 7',channels=CHANNELS,
            split_boundaries=b,scaler_fit_rows=[0,b[0]],scaler_mean=mu.tolist(),scaler_std=sd.tolist(),
            normalization='train-only ddof=0 standardization',metric_units='standardized',
            data_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),history_policy='past context may cross split; future targets do not',
            sample_counts=[len(s) for s in sets])
        return sets,meta,None
    from torchvision.datasets import CIFAR10
    train=CIFAR10(str(DATA/'cifar'),train=True,download=False);test=CIFAR10(str(DATA/'cifar'),train=False,download=False)
    perm=torch.randperm(50000,generator=torch.Generator().manual_seed(42));ti,vi=perm[:45000],perm[45000:]
    assert len(torch.unique(perm))==50000
    np.savez(run/'split_indices.npz',train=ti.numpy(),validation=vi.numpy())
    arr=np.ascontiguousarray(train.data.transpose(0,3,1,2).reshape(-1,3,1024))
    assert np.array_equal(arr[123].reshape(3,32,32).transpose(1,2,0),train.data[123])
    sm=np.zeros(3);sq=np.zeros(3);n=0
    for ix in ti.split(1000):
        z=arr[ix.numpy()].astype(np.float64)/255;sm+=z.sum((0,2));sq+=(z*z).sum((0,2));n+=z.shape[0]*1024
    mu=sm/n;sd=np.sqrt(sq/n-mu**2)
    full=TensorDataset(torch.from_numpy(arr),torch.tensor(train.targets))
    tx=np.ascontiguousarray(test.data.transpose(0,3,1,2).reshape(-1,3,1024))
    sets=[Subset(full,ti.tolist()),Subset(full,vi.tolist()),TensorDataset(torch.from_numpy(tx),torch.tensor(test.targets))]
    meta=dict(input_shape=[1024,3],layout='left to right, top to bottom; RGB channels as three features',
        split_seed=42,sample_counts=[45000,5000,10000],channel_mean=mu.tolist(),channel_std=sd.tolist(),
        classes=train.classes,augmentation='none',purpose='pipeline_validation_only',convergence_label=LIMIT_LABEL,
        comparison_location='appendix_only',normalization='training subset only')
    return sets,meta,(mu,sd)


class Position(nn.Module):
    def __init__(self,length,width=32):
        super().__init__();pos=torch.arange(length).float()[:,None]
        w=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000)/width))
        enc=torch.empty(length,width);enc[:,0::2]=torch.sin(pos*w);enc[:,1::2]=torch.cos(pos*w)
        self.register_buffer('encoding',enc[None],persistent=True)
    def forward(self,x):return x+self.encoding[:,:x.shape[1]]


class Encoder(nn.Module):
    def __init__(self,features,length,layers):
        super().__init__();self.embed=nn.Linear(features,32);self.pos=Position(length)
        layer=nn.TransformerEncoderLayer(32,4,dim_feedforward=64,dropout=0.1,activation='relu',
                                        batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
    def forward(self,x):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return self.encoder(self.pos(self.embed(x)),is_causal=False)


class ForecastTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder(7,96,1);self.project=nn.Linear(32,7);self.time_head=nn.Linear(96,96)
    def forward(self,x):return self.time_head(self.project(self.encoder(x)).transpose(1,2)).transpose(1,2)


class DLinear(nn.Module):
    def __init__(self):super().__init__();self.trend=nn.Linear(96,96);self.remainder=nn.Linear(96,96)
    def forward(self,x):
        z=x.transpose(1,2);trend=F.avg_pool1d(F.pad(z,(12,12),mode='replicate'),25,stride=1)
        return (self.trend(trend)+self.remainder(z-trend)).transpose(1,2)


class Classifier(nn.Module):
    def __init__(self,arch):
        super().__init__();self.arch=arch
        if arch=='transformer':self.mixer=Encoder(3,1024,2);self.head=nn.Linear(32,10)
        else:self.mixer=nn.LSTM(3,64,1,batch_first=True);self.head=nn.Linear(64,10)
        self.drop=nn.Dropout(0.1)
    def forward(self,x):
        if x.shape[1:]!=(1024,3):raise ValueError('Expected [B,1024,3]')
        z=self.mixer(x)
        if self.arch=='lstm':z=z[0]
        return self.head(self.drop(z.mean(1)))


class Naive(nn.Module):
    def forward(self,x):return x[:,-1:,:].expand(-1,96,-1)


def build(task,arch):
    if task=='sCIFAR':return Classifier(arch)
    return {'linear':DLinear,'transformer':ForecastTransformer,'naive':Naive}[arch]()


def batch_gpu(batch,stats):
    x,y=[v.cuda(non_blocking=True) for v in batch]
    if stats is not None:x=((x.float()/255-stats[0])/stats[1]).transpose(1,2).contiguous()
    return x,y


@torch.no_grad()
def evaluate(model,loader,classification,stats):
    model.eval();n=0;loss=0.;correct=0;ae=0.;elements=0
    for batch in loader:
        x,y=batch_gpu(batch,stats);p=model(x)
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite evaluation output')
        n+=len(y)
        if classification:loss+=F.cross_entropy(p,y,reduction='sum').item();correct+=(p.argmax(-1)==y).sum().item()
        else:
            e=(p-y).double();loss+=e.square().sum().item();ae+=e.abs().sum().item();elements+=y.numel()
    assert n==len(loader.dataset)
    return dict(accuracy=correct/n,loss=loss/n,examples=n) if classification else dict(mse=loss/elements,mae=ae/elements,examples=n,scalar_targets=elements)


def configure(seed):
    torch.set_num_threads(6);torch.set_num_interop_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.get_device_capability(0)==(7,5)
    assert len(os.sched_getaffinity(0))<=6


def experiment(args):
    run_id,path=new_run(args.run_id);started=datetime.fromtimestamp(PROCESS_STARTED_UNIX,timezone.utc).isoformat();classification=args.task=='sCIFAR'
    row=dict(run_id=run_id,exp='Tier2-comparison-preliminary',task=args.task,status='running',started_at=started,
        argv=[sys.executable,*sys.argv],n_concurrent=1,t_setup_s=0.,t_train_s=0.,t_val_s=0.,n_val_calls=0,
        t_wall_s=0.,peak_mem_mb=0.,result={'statistical_note':NOTE},env={},statistical_note=NOTE)
    config=dict(arch=args.arch,n_params=None,seed=args.seed,statistical_note=NOTE,preliminary=True,
        epochs=args.epochs,batch_size=64 if classification else 128,learning_rate=0.001,
        optimizer='AdamW' if classification else 'Adam',weight_decay=0.01 if classification else 0.,
        precision='fp32',cpu_core_limit=6,flash_attention=False,sdpa_backend='mem-efficient only; math disabled',
        attention_causal=False,gradient_clip=1.,data_order_seed=args.seed,
        stopping='fixed equal epochs' if classification else 'max 20 epochs, validation patience 3',
        purpose='pipeline_validation_only' if classification else 'negative_control_comparison',
        convergence_label=LIMIT_LABEL if classification else 'preliminary protocol run',
        timing_scope='CUDA-synchronized training-loop wall time, including data transfers; excludes validation/test/checkpoint IO',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_policy='exactly one final selected-checkpoint evaluation; never for tuning')
    row['config']=config
    active_phase=None;section_start=None
    try:
        configure(args.seed)
        row['env']=dict(host='kcvl-gpu3',container='smyu-dev',gpu=torch.cuda.get_device_name(0),sm='75',
            torch=str(torch.__version__),cuda=torch.version.cuda,cpu_affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=torch.get_num_threads(),flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),math_enabled=torch.backends.cuda.math_sdp_enabled(),
            statistical_note=NOTE)
        sets,meta,stats=data_for(args.task,path);config.update(meta)
        # Data construction never decides architecture initialization.
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        if stats is not None:stats=tuple(torch.tensor(v,dtype=torch.float32,device='cuda').view(1,3,1) for v in stats)
        model=build(args.task,args.arch).cuda();config['n_params']=sum(p.numel() for p in model.parameters())
        config['comparison_group']=f"T2C-{args.task}-s{args.seed}"
        if args.arch=='naive':config.update(epochs=0,optimizer=None,learning_rate=None,weight_decay=0.,stopping='deterministic no-training baseline')
        if args.task!='sCIFAR' and args.arch!='naive':assert abs(config['n_params']-18624)/18624<0.03
        generator=torch.Generator().manual_seed(args.seed)
        loaders=[DataLoader(ds,batch_size=config['batch_size'],shuffle=i==0,generator=generator if i==0 else None,
            num_workers=0,pin_memory=True) for i,ds in enumerate(sets)]
        opt=None if args.arch=='naive' else (torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.01)
              if classification else torch.optim.Adam(model.parameters(),lr=0.001))
        sync();row['t_setup_s']=time.perf_counter()-PROCESS_START;torch.cuda.reset_peak_memory_stats()
        write(path/'config.json',config);write(path/'progress.json',dict(status='training',run_id=run_id,statistical_note=NOTE))
        best=None;best_epoch=0;stale=0;completed=0;stop='max_epochs';val=None
        if args.arch!='naive':
            for epoch in range(1,args.epochs+1):
                if time.time()>=args.deadline-45:raise TimeoutError('Global deadline before a complete paired protocol')
                model.train();sync();t=time.perf_counter();active_phase='train';section_start=t;ls=0.;n=0
                for i,batch in enumerate(loaders[0],1):
                    if time.time()>=args.deadline-30:raise TimeoutError('Global deadline during epoch')
                    x,y=batch_gpu(batch,stats);opt.zero_grad(set_to_none=True);pred=model(x)
                    loss=F.cross_entropy(pred,y) if classification else F.mse_loss(pred,y)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                    ls+=loss.item()*len(y);n+=len(y)
                    if i==1 or i%100==0:write(path/'progress.json',dict(status='training',epoch=epoch,batch=i,
                        batches=len(loaders[0]),run_id=run_id,statistical_note=NOTE))
                sync();dt=time.perf_counter()-t;row['t_train_s']+=dt;active_phase=None
                sync();t=time.perf_counter();active_phase='val';section_start=t;val=evaluate(model,loaders[1],classification,stats);sync()
                dv=time.perf_counter()-t;row['t_val_s']+=dv;row['n_val_calls']+=1;completed=epoch;active_phase=None
                score=val['accuracy'] if classification else val['mse']
                improved=best is None or (score>best if classification else score<best)
                if improved:
                    best=score;best_epoch=epoch;stale=0
                    save(path/'best.pt',dict(model=model.state_dict(),epoch=epoch,config=config,validation=val))
                else:stale+=1
                hist=dict(epoch=epoch,t_train_s=dt,t_val_s=dv,train_loss=ls/n,validation=val,statistical_note=NOTE)
                with (path/'history.jsonl').open('a') as f:f.write(json.dumps(hist,ensure_ascii=False)+'\n')
                print('EPOCH',epoch,'TRAIN_S',dt,'VAL_S',dv,'SCORE',score,flush=True)
                if not classification and stale>=3:stop='early_stopping';break
            selected=torch.load(path/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model'])
            val=selected['validation']
        sync();t=time.perf_counter();test=evaluate(model,loaders[2],classification,stats);sync();test_s=time.perf_counter()-t
        row['status']='budget_limited' if classification else 'done'
        row['result']=dict(test=test,best_validation=val,best_epoch=best_epoch,epochs_completed=completed,
            stop_reason='fixed_preliminary_budget' if classification else stop,t_test_s=test_s,test_evaluation_calls=1,
            purpose=config['purpose'],convergence_label=config['convergence_label'],statistical_note=NOTE,
            checkpoint=str(path/'best.pt') if args.arch!='naive' else None)
        row['t_test_s']=test_s
    except Exception as e:
        if active_phase is not None:
            sync();row['t_train_s' if active_phase=='train' else 't_val_s']+=time.perf_counter()-section_start
        row['status']='failed';row['result']=dict(error=repr(e),traceback=traceback.format_exc(),statistical_note=NOTE)
        write(path/'failure.json',row['result'])
    finally:
        row['t_wall_s']=time.perf_counter()-PROCESS_START
        if torch.cuda.is_initialized():row['peak_mem_mb']=torch.cuda.max_memory_allocated()/1024**2
        row['completed_at']=now();write(path/'result.json',row);write(path/'progress.json',dict(status=row['status'],run_id=run_id,statistical_note=NOTE))
        append(row);print('RESULT',json.dumps(row,ensure_ascii=False),flush=True)
    return row['status']!='failed'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['ETTh1','ETTm1','sCIFAR'],required=True)
    p.add_argument('--arch',choices=['linear','transformer','naive','lstm'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--run-id',required=True);p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    with (ROOT/'gpu_experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ok=experiment(a)
    raise SystemExit(0 if ok else 1)
