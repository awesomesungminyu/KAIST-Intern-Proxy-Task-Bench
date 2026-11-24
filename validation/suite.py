"""Candidate suite orchestration. One immutable identity per resumable output folder."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .util import canonical_hash, digest, write_json


def add_commands(actions):
    p=actions.add_parser('precheck',help='CPU candidate contract checks (no training)')
    p.add_argument('--mixer',required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--out',type=Path,default=Path('precheck.json'))
    p.add_argument('--gpu-benchmark',action='store_true',help='Opt in to maximum-length GPU probe')
    p=actions.add_parser('suite',help='Precheck, then resumable Tier 1 -> Tier 2 execution')
    p.add_argument('--mixer',required=True);p.add_argument('--profile',choices=['smoke','full'],default='smoke')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--tfl-root',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--cascade',action='store_true');p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('--tasks',nargs='+');p.add_argument('--python',default=sys.executable)
    p.add_argument('--allow-busy-gpu',action='store_true',help='Explicit opt-in to shared GPU use; never stops other jobs')
    p.add_argument('--stop-starting-at',help='ISO timestamp including timezone, or EVO_TRAINING_CUTOFF')
    p=actions.add_parser('report',help='Create machine/human reports; smoke never receives scores')
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--baselines',type=Path,default=Path(__file__).parent/'baselines/final16_v4.json')


def _load(path):
    return json.loads(Path(path).read_text())


def _cell(a, task, spec, seed, identity):
    from . import runner
    cell=a.out/'cells'/f'{task}-s{seed}';cell.mkdir(parents=True,exist_ok=True)
    result=cell/'result.json'
    if result.exists():
        previous=_load(result)
        if previous.get('suite_identity')==identity and previous.get('status')=='done':
            # Evidence must still match its manifest and native output; do not trust status alone.
            native=Path(previous['native_result'])
            if native.is_file() and digest(native)==previous.get('native_result_sha256'):
                return previous
    cutoff=a.stop_starting_at or os.environ.get('EVO_TRAINING_CUTOFF')
    if cutoff and datetime.now(timezone.utc)>=datetime.fromisoformat(cutoff).astimezone(timezone.utc):
        row=dict(task=task,seed=seed,status='not_run',reason='training cutoff reached',suite_identity=identity)
        write_json(result,row);return row
    attempt=1
    while (cell/f'attempt{attempt}').exists():attempt+=1
    args=argparse.Namespace(task=task,model=a.model_id,python=a.python,data_root=a.data_root,
        tfl_root=a.tfl_root,profile=a.profile,seeds=[seed],timeout=a.timeout,out=cell/f'attempt{attempt}',
        mixer=a.mixer,device=a.device,one_step=a.profile=='smoke',allow_busy_gpu=a.allow_busy_gpu)
    runner.run(args,spec)
    child=list(args.out.glob('*/result.json'))
    if not child:raise RuntimeError('Runner did not write a result')
    row=_load(child[0]);row['suite_identity']=identity
    row['result_path']=str(child[0].resolve());row['native_result']=str((child[0].parent/'native'/'result.json').resolve())
    # Full native/formal trainers emit different native artifacts; fingerprint the runner result as fallback.
    if not Path(row['native_result']).is_file():row['native_result']=row['result_path']
    row['native_result_sha256']=digest(row['native_result'])
    write_json(result,row);return row


def run_suite(a):
    from .runner import load_protocol, PROTOCOL, ROOT
    from .precheck import run_precheck
    from proxy_tasks.common_set.plugins import describe
    if a.jobs<1 or a.jobs>32:raise ValueError('jobs must be between 1 and 32')
    if a.device=='cuda' and a.jobs!=1:raise ValueError('Shared GPU suite currently requires --jobs 1')
    if a.profile=='full' and a.device!='cuda':raise ValueError('Full frozen training requires CUDA; use CPU smoke')
    if not a.seeds or len(set(a.seeds))!=len(a.seeds) or min(a.seeds)<0:raise ValueError('Seeds must be unique nonnegative integers')
    if not 0<a.timeout<float('inf'):raise ValueError('Timeout must be positive and finite')
    a.out=a.out.resolve();a.data_root=a.data_root.resolve()
    a.tfl_root=(a.tfl_root or ROOT/'third_party/tfl').resolve()
    protocol=load_protocol(); tasks=a.tasks or list(protocol['tasks'])
    if len(set(tasks))!=len(tasks) or any(t not in protocol['tasks'] for t in tasks):raise ValueError('Unknown or duplicate tasks')
    tasks=sorted(tasks,key=lambda t:protocol['tasks'][t]['tier'])
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'suite.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another process owns this suite')
        check=run_precheck(a.mixer,a.out/'precheck.json',a.data_root)
        if check['status']!='passed':
            write_json(a.out/'suite.json',dict(schema_version=1,status='blocked',profile=a.profile,
                reason='precheck failed',candidate=check['candidate'],runs=[],tasks=tasks,seeds=a.seeds))
            return 1
        candidate=describe(a.mixer);a.model_id=candidate['model_id'];a.mixer=candidate['reference']
        identity_doc=dict(candidate_sha256=candidate['sha256'],protocol_sha256=digest(PROTOCOL),
            harness_sha256=check['harness_sha256'],profile=a.profile,seeds=a.seeds,tasks=tasks,
            data_root=str(a.data_root),tfl_root=str(a.tfl_root),device=a.device,python=str(Path(a.python).resolve()),cascade=a.cascade)
        # Data contents are fingerprinted once per suite identity and again by each runner cell.
        assets={f'{task}/{name}':a.data_root/task/name for task in tasks for name in protocol['tasks'][task]['assets']}
        from .util import fingerprint
        identity_doc['data_sha256']=fingerprint(assets)['sha256']
        identity=canonical_hash(identity_doc)
        path=a.out/'suite.json'
        if path.exists() and _load(path).get('identity') not in (None,identity):
            raise ValueError('Suite identity changed; use a new --out (candidate/protocol/source/data/settings differ)')
        document=dict(schema_version=1,status='running',identity=identity,configuration=identity_doc,
            candidate=candidate,profile=a.profile,tasks=tasks,seeds=a.seeds,runs=[],precheck='passed')
        write_json(path,document);failed_tier1=False
        for tier in [1,2]:
            selected=[t for t in tasks if protocol['tasks'][t]['tier']==tier]
            if tier==2 and a.cascade and failed_tier1:
                document['runs'] += [dict(task=t,seed=s,status='not_run',reason='cascade: Tier 1 failure/timeout') for t in selected for s in a.seeds]
                write_json(path,document);break
            with ThreadPoolExecutor(max_workers=a.jobs) as pool:
                futures={pool.submit(_cell,a,t,protocol['tasks'][t],s,identity):(t,s) for t in selected for s in a.seeds}
                for future in as_completed(futures):
                    task,seed=futures[future]
                    try: row=future.result()
                    except Exception as exc:row=dict(task=task,seed=seed,status='failed',error=f'{type(exc).__name__}: {exc}')
                    document['runs'].append(row)
                    if tier==1 and row['status']!='done':failed_tier1=True
                    write_json(path,document)
        document['status']='done' if all(r['status']=='done' for r in document['runs']) else 'incomplete'
        write_json(path,document)
        return int(document['status']!='done')


def dispatch(a):
    if a.action=='precheck':
        from .precheck import run_precheck
        d=run_precheck(a.mixer,a.out,a.data_root,gpu=a.gpu_benchmark)
        print(json.dumps(dict(status=d['status'],out=str(a.out),candidate=d['candidate'])))
        return int(d['status']!='passed')
    if a.action=='suite':return run_suite(a)
    from .report import make_report
    d=make_report(a.suite,a.baselines)
    print(json.dumps(dict(status=d['status'],performance_comparable=d['performance_comparable'],out=str(a.suite/'report.json'))))
    return 0
