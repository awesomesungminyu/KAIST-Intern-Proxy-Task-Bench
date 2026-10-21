"""Sequential experiment supervisor with a single absolute one-hour deadline."""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REQUEST_STARTED=1792138361.6705713
DEADLINE=REQUEST_STARTED+3600


def write(value):
    value['updated_utc']=datetime.now(timezone.utc).isoformat()
    temp=ROOT/'suite_summary.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(ROOT/'suite_summary.json')


def main():
    state=dict(status='preparing_data',request_started_unix=REQUEST_STARTED,deadline_unix=DEADLINE,
        deadline_kst='2026-10-16 18:12:41 KST',order=['ETTh1','ETTm1','sCIFAR'],results={},failures={})
    sc=Path('/workspace/exp/2026-10-16-scraw/runs/cnn-s456-20261016T080408Z/result.json')
    if sc.exists():state['results']['SC-Raw']=json.loads(sc.read_text())
    write(state)
    specs=[('ETTh1',20,240),('ETTm1',20,240),('scifar',30,900)]
    for task,epochs,budget in specs:
        if time.time()>DEADLINE-120:
            state['failures'][task]={'status':'skipped_deadline'};write(state);continue
        if task=='scifar':
            ready=False;prep_end=min(time.time()+480,DEADLINE-1050)
            state.update(status='waiting_for_cifar_data',current_task='scifar');write(state)
            while time.time()<prep_end:
                status=ROOT/'prepare_status.json'
                if status.exists():
                    try:ready=json.loads(status.read_text()).get('status')=='completed'
                    except json.JSONDecodeError:ready=False
                if ready or (ROOT/'prepare_failure.json').exists():break
                time.sleep(5)
            if not ready:
                state['failures'][task]={'status':'data_unavailable','detail':'CIFAR preparation did not complete within its time budget'}
                write(state);continue
        if task!='scifar' and not (ROOT/'data'/(task+'.csv')).exists():
            state['failures'][task]={'status':'data_unavailable'};write(state);continue
        run=ROOT/'runs'/task
        state.update(status='running',current_task=task);write(state)
        cmd=[sys.executable,'-u',str(ROOT/'quick_bench.py'),'--task',task,'--run',str(run),
            '--epochs',str(epochs),'--budget',str(budget),'--deadline',str(DEADLINE)]
        log_path=ROOT/(task+'.log')
        with log_path.open('x') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                  start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            state.update(current_pid=proc.pid,current_run=str(run),current_log=str(log_path));write(state)
            try:code=proc.wait(timeout=max(1,min(budget+180,DEADLINE-time.time()-60)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code='watchdog_timeout'
        result=run/'result.json'
        if code==0 and result.exists():state['results'][task]=json.loads(result.read_text())
        else:
            detail=json.loads((run/'failure.json').read_text()) if (run/'failure.json').exists() else {}
            state['failures'][task]=dict(status='failed',exit_code=code,detail=detail,log=str(log_path))
        write(state)
    state.update(status='completed' if not state['failures'] else 'finished_with_failures',
        current_task=None,current_pid=None,finished_unix=time.time(),within_one_hour=time.time()<=DEADLINE)
    write(state)
    print('SUITE_FINISHED',json.dumps({'status':state['status'],'within_one_hour':state['within_one_hour'],
                                      'completed':list(state['results']),'failures':state['failures']}),flush=True)


if __name__=='__main__':main()
