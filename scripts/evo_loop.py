"""Bounded candidate generation -> precheck -> suite -> report -> feedback loop.

Default auto uses the local-only EvoScientist adapter when configured; otherwise
an explicit deterministic mock. Mock success is wiring evidence, not AI discovery.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validation.runner import load_protocol
from validation.util import write_json
from scripts.make_feedback import make_feedback

IDEAS=['AdaptiveLocalGlobal','TemporalSpecialization','SSMLinearHybrid','DynamicCausalGraph','RelationalPrefixMemory']


def run(command, log, timeout=7200):
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            return 124


def initial_candidates(folder,count):
    folder.mkdir(parents=True,exist_ok=True)
    for i,cls in enumerate(IDEAS[:count]):
        p=folder/f'{i+1}_{cls}.py'
        if not p.exists():p.write_text((ROOT/'examples/research_mixers.py').read_text()+f'\nCandidateMixer = {cls}\n')
    return sorted(folder.glob('*.py'))[:count]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--candidate-root',type=Path)
    p.add_argument('--generations',type=int,default=2);p.add_argument('--candidates',type=int,default=5)
    p.add_argument('--profile',choices=['smoke','full'],default='smoke')
    p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--max-llm-calls',type=int,default=30);p.add_argument('--backend',choices=['auto','mock','evoscientist'],default='mock')
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--tfl-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--timeout',type=float,default=240)
    p.add_argument('--stop-starting-at',default=os.environ.get('EVO_TRAINING_CUTOFF'))
    a=p.parse_args(argv)
    if not 1<=a.generations<=10 or not 1<=a.candidates<=5:raise ValueError('Generations 1..10, gen0 candidates 1..5; later generations <=3')
    if not 0<=a.max_llm_calls<=30:raise ValueError('LLM call cap must be 0..30')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    a.candidate_root=(a.candidate_root or a.out/'candidates').resolve()
    identity=dict(profile=a.profile,seeds=a.seeds,device=a.device,jobs=a.jobs,candidates=a.candidates,
                  backend=a.backend,max_llm_calls=a.max_llm_calls,data_root=str(a.data_root.resolve()))
    state_path=a.out/'loop.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('configuration')!=identity:
        raise ValueError('Loop configuration changed; use a new --out')
    state=dict(schema_version=1,status='running',configuration=identity,generations=[],human_intervention=False,
               source_note='gen0 is a sourced substitute set; requested historical EvoScientist top-5 was not found')
    write_json(state_path,state)
    previous_feedback=None
    for generation in range(a.generations):
        work=a.out/f'gen{generation}';work.mkdir(parents=True,exist_ok=True)
        folder=a.candidate_root/f'gen{generation}'
        if generation==0:
            files=initial_candidates(folder,a.candidates);origin='sourced_substitute'
        else:
            from integrations.evoscientist.adapter import generate
            generated=generate(previous_feedback,folder,min(3,a.candidates),a.backend,a.max_llm_calls,a.out/'llm_budget.json',work)
            files=[Path(x) for x in generated['files']];origin=generated['backend']
        results=[];reports=[]
        for candidate in files:
            out=work/candidate.stem;out.mkdir(parents=True,exist_ok=True)
            ref=str(candidate)+':CandidateMixer'
            pre=run([sys.executable,'-m','validation','precheck','--mixer',ref,'--data-root',str(a.data_root),'--out',str(out/'precheck.json')],out/'precheck.log',240)
            if pre==0:
                command=[sys.executable,'-m','validation','suite','--mixer',ref,'--profile',a.profile,'--seeds',*[str(s) for s in a.seeds],
                    '--out',str(out),'--data-root',str(a.data_root),'--tfl-root',str(a.tfl_root),'--device',a.device,'--jobs',str(a.jobs),'--timeout',str(a.timeout)]
                if a.stop_starting_at:command+=['--stop-starting-at',a.stop_starting_at]
                code=run(command,out/'suite.log')
            else:
                check=json.loads((out/'precheck.json').read_text()) if (out/'precheck.json').exists() else {}
                write_json(out/'suite.json',dict(status='blocked',profile=a.profile,precheck='failed',reason='precheck failed',
                    candidate=check.get('candidate'),tasks=list(load_protocol()['tasks']),seeds=a.seeds,runs=[]))
                code=None
            report=run([sys.executable,'-m','validation','report','--suite',str(out)],out/'report.log',60)
            if report==0:reports.append(out/'report.json')
            suite=json.loads((out/'suite.json').read_text()) if (out/'suite.json').exists() else {}
            results.append(dict(file=str(candidate),precheck_exit=pre,suite_exit=code,report_exit=report,
                passed_cells=sum(r['status']=='done' for r in suite.get('runs',[])),requested_cells=len(load_protocol()['tasks'])*len(a.seeds)))
            write_json(work/'candidates.json',results)
        previous_feedback=make_feedback(reports,work/'feedback.md')
        state['generations'].append(dict(index=generation,origin=origin,candidates=results,
            validity_rate=sum(r['precheck_exit']==0 for r in results)/len(results) if results else 0,
            feedback=str(previous_feedback)))
        write_json(state_path,state)
    state['status']='completed' if all(r['report_exit']==0 for g in state['generations'] for r in g['candidates']) else 'incomplete'
    state['all_smoke_passed']=all(r['precheck_exit']==r['suite_exit']==r['report_exit']==0 for g in state['generations'] for r in g['candidates'])
    write_json(state_path,state)
    print(json.dumps(dict(status=state['status'],all_smoke_passed=state['all_smoke_passed'],generations=len(state['generations']))))
    return int(state['status']!='completed' or not state['all_smoke_passed'])
if __name__=='__main__':raise SystemExit(main())
