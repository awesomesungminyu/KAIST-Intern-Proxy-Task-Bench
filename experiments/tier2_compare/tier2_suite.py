"""Complete ETT comparisons first; only admit an entire matched sCIFAR pair."""
import os,time,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
ROOT=Path(__file__).resolve().parent
START=1792139915.0
DEADLINE=START+3600
NOTE='seed 1회, 분산 미추정'
STATE=dict(status='running',request_started_unix=START,deadline_unix=DEADLINE,
    statistical_note=NOTE,results=[],failures=[],deferred=[],seed_extension_triggered=False)


def record():
    STATE['updated_at']=datetime.now(timezone.utc).isoformat()
    p=ROOT/'suite_status.tmp';p.write_text(json.dumps(STATE,ensure_ascii=False,indent=2));p.replace(ROOT/'suite_status.json')


def launch(task,arch,seed,epochs):
    base=f"T2C-{task}-{arch}-s{seed}-"+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    cmd=[sys.executable,'-u',str(ROOT/'tier2_compare.py'),'--task',task,'--arch',arch,
         '--seed',str(seed),'--epochs',str(epochs),'--run-id',base,'--deadline',str(DEADLINE-45)]
    STATE.update(current_task=task,current_arch=arch,current_seed=seed,current_run_id=base);record()
    with (ROOT/(base+'.log')).open('x') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',NUMEXPR_NUM_THREADS='6'))
        STATE['current_pid']=p.pid;record();code=p.wait()
    candidates=sorted((ROOT/'runs').glob(base+'*/result.json'))
    row=json.loads(candidates[-1].read_text()) if candidates else dict(run_id=base,task=task,
            config=dict(arch=arch,seed=seed),status='failed',result={'error':'child exited without result','exit_code':code,'statistical_note':NOTE})
    STATE['results' if row['status'] in {'done','budget_limited'} else 'failures'].append(row);record()
    return row


def main():
    record()
    for task in ['ETTh1','ETTm1']:
        for arch in ['linear','transformer','naive']:
            row=launch(task,arch,0,20)
            if row['status']=='failed':
                STATE['status']='ETT_incomplete';STATE['deferred'].append({'task':'sCIFAR','reason':'ETT comparison not complete','statistical_note':NOTE});record();return
    STATE['ett_seed0_completed_unix']=time.time();STATE['ett_seed0_minutes']=(time.time()-START)/60;record()
    if time.time()-START<=1200 and DEADLINE-time.time()>600:
        STATE['seed_extension_triggered']=True;record()
        for seed in [1,2]:
            for task in ['ETTh1','ETTm1']:
                for arch in ['linear','transformer']:
                    launch(task,arch,seed,20)
    # Admission is based on sequential synthetic timing preflight, before either sCIFAR run.
    pre=json.loads((ROOT/'preflight.json').read_text())
    epoch_est={a:v['train_batch_s']*704+v['eval_batch_s']*79 for a,v in pre['scifar_timing'].items()}
    epochs=3
    required=sum(epoch_est.values())*epochs*1.75+180
    remaining=DEADLINE-time.time()
    STATE['scifar_admission']=dict(equal_epochs=epochs,estimated_pair_seconds=required,remaining_seconds=remaining,
        batch_size=64,statistical_note=NOTE);record()
    if required<remaining-120 and not STATE['failures']:
        for arch in ['transformer','lstm']:launch('sCIFAR',arch,0,epochs)
    else:
        STATE['deferred'].append(dict(task='sCIFAR',reason='whole matched pair not safely admissible within deadline',
                                      statistical_note=NOTE))
    STATE.update(status='completed' if not STATE['failures'] else 'finished_with_failures',
        current_task=None,current_arch=None,current_pid=None,finished_unix=time.time(),within_one_hour=time.time()<=DEADLINE)
    record();print('SUITE_DONE',STATE['status'],flush=True)


if __name__=='__main__':main()
