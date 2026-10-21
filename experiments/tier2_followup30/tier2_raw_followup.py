"""Matched full-length SC-Raw sequence mixer smoke comparison, seed 0."""
import sys
sys.path.insert(0,'/workspace/exp/2026-10-16-tier2-compare')
import tier2_compare as h
from tier2_compare import *
ROOT=Path(__file__).resolve().parent
h.ROOT=ROOT
CACHE=Path('/workspace/exp/2026-10-16-scraw/raw_cache_v1')

class RawDataset(Dataset):
    def __init__(self,split,indices=None):
        self.x=np.load(CACHE/(split+'_x.npy'),mmap_mode='r')
        self.y=np.load(CACHE/(split+'_y.npy'),mmap_mode='r')
        self.indices=np.arange(len(self.y)) if indices is None else indices
        assert self.x.shape==(len(self.y),16000)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i]
        return torch.from_numpy(np.array(self.x[j],copy=True))[:,None],int(self.y[j])

class RawClassifier(nn.Module):
    def __init__(self,arch):
        super().__init__();self.arch=arch
        self.mixer=Encoder(1,16000,1) if arch=='transformer' else nn.LSTM(1,44,1,batch_first=True)
        self.head=nn.Linear(32 if arch=='transformer' else 44,12)
    def forward(self,x):
        assert x.shape[1:]==(16000,1)
        z=self.mixer(x)
        return self.head((z[0] if self.arch=='lstm' else z).mean(1))

def run(a):
    rid,p=new_run(a.run_id)
    row=dict(run_id=rid,exp='Tier2-followup30-preliminary',task='SC-Raw',status='running',
        started_at=datetime.fromtimestamp(PROCESS_STARTED_UNIX,timezone.utc).isoformat(),
        argv=[sys.executable,*sys.argv],n_concurrent=1,t_setup_s=0.,t_train_s=0.,t_val_s=0.,
        n_val_calls=0,t_wall_s=0.,peak_mem_mb=0.,result={'statistical_note':NOTE},env={},statistical_note=NOTE)
    cfg=dict(arch=a.arch,n_params=None,seed=0,statistical_note=NOTE,preliminary=True,
        input_shape=[16000,1],sequence_length=16000,downsampling=False,patching=False,
        batch_size=2,updates_planned=a.steps,learning_rate=.001,optimizer='AdamW',weight_decay=.01,
        gradient_clip=1.,precision='fp32',cpu_core_limit=6,flash_attention=False,
        sdpa_backend='mem-efficient only; math disabled',attention_causal=False,
        purpose='pipeline_validation_only',convergence_label=LIMIT_LABEL,comparison_location='appendix_only',
        stopping='same fixed update/sample budget for both architectures; no full training epoch',
        readout='mean over all 16000 positions, then linear 12-class head',
        model_details=('1 encoder layer, d_model32, heads4, FF64, dropout0.1, sinusoidal position'
                       if a.arch=='transformer' else '1 unidirectional LSTM layer, hidden44'),
        normalization='existing cache: PCM16/32768, no per-example normalization, zero-pad/crop to16000',
        data_split='existing official Speech Commands split and seed456 unknown/silence sampling retained',
        caveat='silence background source recordings shared across split; speech speaker split disjoint',
        test_policy='one full held-out test evaluation after fixed training; never used for tuning',
        comparison_group='T2F30-SC-Raw-seed0',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing_scope='CUDA-synchronized training loop including transfers, excluding validation/test/checkpoint IO')
    row['config']=cfg;phase=None;t=None
    try:
        configure(0)
        row['env']=dict(host='kcvl-gpu3',container='smyu-dev',gpu=torch.cuda.get_device_name(0),sm='75',
            torch=str(torch.__version__),cuda=torch.version.cuda,cpu_affinity=sorted(os.sched_getaffinity(0)),
            torch_threads=torch.get_num_threads(),flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
            mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math_enabled=torch.backends.cuda.math_sdp_enabled(),statistical_note=NOTE)
        indices=torch.randperm(36923,generator=torch.Generator().manual_seed(0))[:a.steps*2].numpy()
        assert len(indices)==a.steps*2 and len(np.unique(indices))==len(indices)
        np.save(p/'train_indices.npy',indices)
        cfg.update(train_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            train_examples_available=36923,train_examples_budget=len(indices),fraction_of_one_epoch=len(indices)/36923,
            sample_counts=[36923,4443,4888],cache_metadata=json.loads((CACHE/'training.json').read_text()))
        sets=[RawDataset('training',indices),RawDataset('validation'),RawDataset('testing')]
        loaders=[DataLoader(ds,batch_size=2,shuffle=False,num_workers=0,pin_memory=True) for ds in sets]
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model=RawClassifier(a.arch).cuda();cfg['n_params']=sum(v.numel() for v in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        write(p/'config.json',cfg);sync();row['t_setup_s']=time.perf_counter()-PROCESS_START
        torch.cuda.reset_peak_memory_stats();model.train();sync();t=time.perf_counter();phase='train';loss_sum=0.;seen=0
        for step,(x,y) in enumerate(loaders[0],1):
            if time.time()>a.deadline-45:raise TimeoutError('30-minute deadline')
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            loss_sum+=loss.item()*len(y);seen+=len(y)
            if step==1 or step%128==0:
                sync();progress=dict(status='training',run_id=rid,step=step,steps=a.steps,
                    train_examples_seen=seen,elapsed_train_s=time.perf_counter()-t,train_loss=loss_sum/seen,
                    statistical_note=NOTE,convergence_label=LIMIT_LABEL)
                write(p/'progress.json',progress)
                with (p/'history.jsonl').open('a') as f:f.write(json.dumps(progress,ensure_ascii=False)+'\n')
                print('PROGRESS',json.dumps(progress,ensure_ascii=False),flush=True)
        sync();row['t_train_s']=time.perf_counter()-t;phase=None
        save(p/'final.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),config=cfg,
            step=step,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        sync();t=time.perf_counter();phase='val'
        write(p/'progress.json',dict(status='validation',run_id=rid,statistical_note=NOTE))
        val=evaluate(model,loaders[1],True,None);sync();row['t_val_s']=time.perf_counter()-t;row['n_val_calls']=1;phase=None
        sync();t=time.perf_counter();phase='test'
        write(p/'progress.json',dict(status='test',run_id=rid,validation=val,statistical_note=NOTE))
        test=evaluate(model,loaders[2],True,None);sync();row['t_test_s']=time.perf_counter()-t;phase=None
        row['status']='budget_limited'
        row['result']=dict(test=test,validation=val,updates_completed=step,train_examples_seen=seen,
            full_epochs_completed=0,train_loss=loss_sum/seen,stop_reason='fixed_equal_update_budget',
            purpose=cfg['purpose'],convergence_label=LIMIT_LABEL,statistical_note=NOTE,
            checkpoint=str(p/'final.pt'),test_evaluation_calls=1)
    except Exception as e:
        if phase:
            sync();key={'train':'t_train_s','val':'t_val_s','test':'t_test_s'}[phase]
            row[key]=row.get(key,0.)+time.perf_counter()-t
        row['status']='failed';row['result']=dict(error=repr(e),traceback=traceback.format_exc(),statistical_note=NOTE)
    finally:
        row['t_wall_s']=time.perf_counter()-PROCESS_START;row['completed_at']=now()
        if torch.cuda.is_initialized():row['peak_mem_mb']=torch.cuda.max_memory_allocated()/1024**2
        write(p/'result.json',row);write(p/'progress.json',dict(status=row['status'],run_id=rid,statistical_note=NOTE))
        append(row);print('RESULT',json.dumps(row,ensure_ascii=False),flush=True)
    return row['status']!='failed'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arch',choices=['transformer','lstm'],required=True)
    ap.add_argument('--steps',type=int,default=2048);ap.add_argument('--run-id',required=True)
    ap.add_argument('--deadline',type=float,required=True);a=ap.parse_args()
    with Path('/workspace/exp/2026-10-16-tier2-compare/gpu_experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ok=run(a)
    raise SystemExit(0 if ok else 1)
