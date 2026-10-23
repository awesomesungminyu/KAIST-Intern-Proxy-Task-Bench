import os,time,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEADLINE=1792144940
NOTE='seed 1회, 분산 미추정'
def write(x):
    p=ROOT/'suite_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def main():
    s=dict(request_started_unix=1792143140,deadline_unix=DEADLINE,started_unix=time.time(),
        statistical_note=NOTE,status='running',runs=[])
    baseline=Path('/workspace/exp/2026-10-12/runs.jsonl').read_text()
    (ROOT/'central_before.jsonl').write_text(baseline)
    write(s)
    for task in ['SC-Raw','sCIFAR']:
        if task=='sCIFAR':
            # Includes setup, validation/test and safety margin for the whole pair.
            epochs=min(10,int((DEADLINE-time.time()-75)/60))
            if epochs<4:
                s['sCIFAR_deferred']='insufficient time for a paired improvement over prior3epochs';break
            s['sCIFAR_epochs']=epochs;write(s)
        for arch in ['transformer','lstm']:
            rid=f'T2F30-{task}-{arch}-s0-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
            script=ROOT/'tier2_raw_followup.py' if task=='SC-Raw' else Path('/workspace/exp/2026-10-16-tier2-compare/tier2_compare.py')
            cmd=[sys.executable,'-u',str(script),'--arch',arch,'--run-id',rid,'--deadline',str(DEADLINE)]
            cmd+=['--steps','2048'] if task=='SC-Raw' else ['--task','sCIFAR','--seed','0','--epochs',str(epochs)]
            rec=dict(task=task,arch=arch,run_id=rid,started_unix=time.time(),argv=cmd,status='running')
            s['runs'].append(rec);write(s)
            with (ROOT/(rid+'.log')).open('w') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            rec.update(returncode=result.returncode,ended_unix=time.time(),status='finished' if result.returncode==0 else 'failed');write(s)
            if result.returncode:
                s['status']='failed';s['ended_unix']=time.time();write(s);return
    s['status']='complete';s['ended_unix']=time.time();write(s)
if __name__=='__main__':main()
