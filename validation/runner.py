"""Execute only the final sixteen tasks. Listing requires the Python standard library."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from proxy_tasks.common.paths import data_root, output_root

from .util import (canonical_hash, command_output, compare, digest, execute, finite,
                   fingerprint, gpu_lock, require_idle_gpu, write_json)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'validation/protocol.json'


def load_protocol(task=None):
    protocol = json.loads(PROTOCOL.read_text())
    return protocol if task is None else (protocol, protocol['tasks'][task])


def data_base(args, spec):
    root = args.data_root if args.data_root else data_root()
    return Path(root).expanduser().resolve() / args.task


def asset_paths(args, spec):
    return {name: data_base(args, spec) / name for name in spec['assets']}


def identities(args, spec):
    source = {str(p.relative_to(ROOT)): p for folder in ['proxy_tasks', 'validation']
              for p in (ROOT / folder).rglob('*.py') if '__pycache__' not in p.parts}
    if spec['adapter'] == 'formal':
        source['external_tfl'] = args.tfl_root / 'src'
    evaluation = {k: v for k, v in source.items()
                  if 'common_set/' not in k and not k.endswith('/models.py')}
    evaluation['protocol'] = PROTOCOL
    if spec['adapter'] == 'formal':
        evaluation.pop('external_tfl')
        evaluation.update({str(p.relative_to(args.tfl_root)): p for p in (args.tfl_root / 'src').rglob('*.py')
                           if 'components' not in p.parts})
    return fingerprint(source), fingerprint(evaluation)['sha256']


def preflight(args, spec):
    if args.model not in spec['models']:
        raise ValueError(f'Unsupported model: {args.model}')
    assets = asset_paths(args, spec)
    missing = [p.name for p in assets.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{args.task}: missing {', '.join(missing)} in {data_base(args, spec)}. See docs/DATA.md and run python -m validation prepare --source <extracted-folder> --task {args.task}")
    data = fingerprint(assets)
    if spec['adapter'] == 'formal':
        expected = (ROOT / 'third_party/tfl.commit').read_text().strip().split()[-1]
        actual = command_output(['git', 'rev-parse', 'HEAD'], args.tfl_root)
        if expected != actual:
            raise ValueError('TFL source commit does not match third_party/tfl.commit')
        if 'cs_mixer' not in (args.tfl_root / 'src/args.py').read_text():
            raise ValueError('TFL common-set patch is missing; run setup.sh parity')
    environment = json.loads(command_output([args.python, '-c',
        "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))))"]))
    if environment['torch'].split('+')[0] != spec['runtime']['torch']:
        raise ValueError('This configuration requires torch ' + spec['runtime']['torch'])
    if args.model == 'cs_retnet':
        command_output([args.python, '-c',
            'from proxy_tasks.common_set.retention import fla_chunk_retention; fla_chunk_retention()'], ROOT)
    source, evaluation = identities(args, spec)
    return dict(ready=True, data=data, environment=environment, source=source, evaluation_sha256=evaluation)


def build_command(args, spec, seed, destination):
    native = destination / 'native'
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(seed), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if spec['adapter'] == 'part':
        env['PROXY_TASK_DATA_DIR'] = str(data_base(args, spec))
        return [args.python, '-m', 'proxy_tasks.final16.train', '--task', args.task,
                '--model', args.model, '--seed', str(seed), '--profile', args.profile,
                '--out', str(native)], ROOT, env
    options = spec.get('common', []) + spec[args.profile]
    if spec['adapter'] == 'native':
        return [args.python, '-m', args.task.replace('-', '_') + '.train', '--model', args.model,
                '--seed', str(seed), '--out_dir', str(native), *options], ROOT / 'proxy_tasks', env
    stage = native / 'tfl'
    stage.mkdir(parents=True)
    (stage / 'src').symlink_to(args.tfl_root / 'src', target_is_directory=True)
    (stage / 'data').mkdir()
    (stage / 'data' / spec['dataset']).symlink_to(data_base(args, spec), target_is_directory=True)
    env.update(EXP_DIR=str(native), EXP_ID='proxy-final16', N_CONCURRENT='1')
    mixer = args.model.removeprefix('cs_')
    return [args.python, str(ROOT / 'proxy_tasks/formal_languages/runner.py'), '-mode', 'train',
            '-dataset', spec['dataset'], '-opt', 'rmsprop', '-batch_size', '32', '-gpu', '0',
            '-seed', str(seed), '-run_name', destination.name, '-model_type', 'CS', '-cs_mixer', mixer,
            '-d_model', '32', '-depth', '2', '-heads', '4', '-lr', '0.001', *options], stage, env


def extract_metrics(args, spec, native, seed):
    if spec['adapter'] == 'part':
        result = json.loads((native / 'result.json').read_text())
        expected = min(3, spec['steps']) if args.profile == 'smoke' else spec['steps']
        if (result['status'], result['task'], result['model'], result['seed'], result['profile'], result['steps']) != (
                'done', args.task, args.model, seed, args.profile, expected):
            raise ValueError('Incomplete or mismatched task result')
        return {} if args.profile == 'smoke' else {spec['metric']: finite(result['eval']['score'], spec['metric'])}
    if spec['adapter'] == 'native':
        result = json.loads((native / f'{args.model}_seed{seed}' / 'history.json').read_text())
        history = result['history']
        if not history or history[-1]['step'] != spec['budget_rule'][args.profile]:
            raise ValueError('Incomplete RegBench budget')
        return {spec['metric']: finite(history[-1][spec['metric']], spec['metric'])}
    records = [json.loads(line) for line in (native / 'runs.jsonl').read_text().splitlines() if line.strip()]
    if len(records) != 1 or records[0]['status'] != 'done':
        raise ValueError('Incomplete Parity result')
    result = records[0]
    if result['config'].get('seed') != seed or result['config'].get('epochs') != (2 if args.profile == 'smoke' else 100):
        raise ValueError('Parity seed or budget mismatch')
    return {spec['metric']: statistics.mean(finite(x, 'bin_accuracy') for x in result['result']['bins_max_acc'])}


def run(args, spec):
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    if args.out is None:
        args.out = output_root() / args.task / args.model / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        destination = args.out / f'{args.task}-{args.model}-s{seed}-{uuid.uuid4().hex[:8]}'
        destination.mkdir()
        row = dict(schema_version=1, task=args.task, model=args.model, seed=seed, profile=args.profile,
                   protocol_version=load_protocol()['version'], protocol_sha256=digest(PROTOCOL), status='running',
                   started_at=datetime.now(timezone.utc).isoformat(), budget=dict(profile=args.profile, configuration=spec),
                   metric=dict(name=spec['metric'], direction=spec['direction'], split=spec['split']),
                   selection=spec['selection'], limitations=spec.get('limitations', []))
        started = time.monotonic()
        try:
            with gpu_lock():
                require_idle_gpu()
                row['provenance'] = preflight(args, spec)
                command, cwd, env = build_command(args, spec, seed, destination)
                write_json(destination / 'manifest.json', dict(command=command, cwd=str(cwd), **row))
                execute(command, cwd, env, destination / 'train.log', args.timeout)
                row['metrics'] = extract_metrics(args, spec, destination / 'native', seed)
                if identities(args, spec)[0]['sha256'] != row['provenance']['source']['sha256']:
                    raise ValueError('Source changed during execution')
                if fingerprint(asset_paths(args, spec))['sha256'] != row['provenance']['data']['sha256']:
                    raise ValueError('Input data changed during execution')
                row.update(status='done', evidence='smoke_only' if args.profile == 'smoke' else 'measured')
        except (Exception, KeyboardInterrupt) as exc:
            row.update(status='timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'failed',
                       error=f'{type(exc).__name__}: {exc}')
        row['wall_seconds'] = time.monotonic() - started
        write_json(destination / 'result.json', row)
        rows.append(row)
        done = len(rows) == len(args.seeds) and all(r['status'] == 'done' for r in rows)
        write_json(args.out / 'summary.json', dict(schema_version=1, runs=rows, requested_seeds=args.seeds,
                                                  status='done' if done else 'failed' if row['status'] != 'done' else 'running'))
        print(json.dumps(dict(task=args.task, model=args.model, seed=seed, status=row['status'], error=row.get('error'))))
        if row['status'] != 'done':
            return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list')
    command = actions.add_parser('prepare', help='Prepare supplied local files without downloading or training')
    command.add_argument('--source', required=True, type=lambda p: Path(p).expanduser().resolve())
    command.add_argument('--task', choices=['all', *load_protocol()['tasks']], default='all')
    command.add_argument('--data-root', type=lambda p: Path(p).expanduser().resolve())
    command = actions.add_parser('data', help='Check data availability without CUDA')
    command.add_argument('--task', choices=['all', *load_protocol()['tasks']], default='all')
    command.add_argument('--data-root', type=lambda p: Path(p).expanduser().resolve())
    for action in ['check', 'run']:
        command = actions.add_parser(action)
        command.add_argument('--task', required=True, choices=list(load_protocol()['tasks']))
        command.add_argument('--model', required=True, choices=load_protocol()['models'])
        command.add_argument('--python', default=sys.executable)
        command.add_argument('--data-root', type=lambda p: Path(p).expanduser().resolve())
        command.add_argument('--tfl-root', type=lambda p: Path(p).expanduser().resolve(), default=ROOT / 'third_party/tfl')
        if action == 'run':
            command.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
            command.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
            command.add_argument('--timeout', type=float, default=3600)
            command.add_argument('--out', type=lambda p: Path(p).expanduser().resolve())
    command = actions.add_parser('compare')
    command.add_argument('--baseline', required=True, type=Path)
    command.add_argument('--candidate', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == 'list':
            print(PROTOCOL.read_text())
        elif args.action == 'prepare':
            from .prepare import prepare
            return prepare(args)
        elif args.action == 'data':
            tasks = load_protocol()['tasks']
            rows = []
            for cid, spec in tasks.items():
                if args.task != 'all' and args.task != cid:
                    continue
                task_args = argparse.Namespace(task=cid, data_root=args.data_root)
                missing = [str(p) for p in asset_paths(task_args, spec).values() if not p.is_file()]
                rows.append(dict(task=cid, ready=not missing, missing=missing))
            print(json.dumps(rows, indent=2))
            return int(any(not row['ready'] for row in rows))
        elif args.action == 'compare':
            print(json.dumps(compare(args.baseline, args.candidate), indent=2, allow_nan=False))
        else:
            spec = load_protocol(args.task)[1]
            if args.action == 'check':
                print(json.dumps(preflight(args, spec), indent=2))
            else:
                if not 0 < args.timeout < float('inf'):
                    raise ValueError('Timeout must be positive and finite')
                return run(args, spec)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(dict(status='failed', error=f'{type(exc).__name__}: {exc}')), file=sys.stderr)
        return 2
