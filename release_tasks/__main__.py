"""Add-on task data and candidate cells for the release integration."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

from validation.runner import PROTOCOL, load_protocol
from validation.util import canonical_hash, digest, gpu_lock, require_idle_gpu, write_json

SPEC = Path(__file__).with_name('protocol.json')


def tasks():
    return json.loads(SPEC.read_text())['tasks']


def data(args):
    from validation.prepare import prepare_task

    failed = False
    for task, spec in tasks().items():
        destination = args.data_root / task
        try:
            if args.source:
                prepare_task(args.source, destination, task, spec)
            missing = [name for name in spec['assets'] if not (destination/name).is_file()]
            row = {'task':task, 'ready':not missing, 'missing':missing}
        except (OSError, ValueError) as error:
            row = {'task':task, 'ready':False, 'error':str(error)}
        print(json.dumps(row))
        failed |= not row['ready']
    return int(failed)


def check(args):
    from proxy_tasks.common_set.plugins import load
    from validation.precheck import check_configuration

    mixer = load(args.mixer)
    shapes = {}
    for task, spec in tasks().items():
        shape = dict(d_model=spec['d_model'], n_layers=spec['layers'],
                     num_heads=spec['heads'], l_max=spec['length'])
        shapes[task] = dict(configuration=shape, **check_configuration(mixer, shape, gpu=False))
    write_json(args.out, shapes)
    return int(not all(row['passed'] for row in shapes.values()))


def extended_cell(args, spec, candidate):
    import torch
    from proxy_tasks.common.gpu_guard import require_cuda
    from proxy_tasks.final16.training import fit
    from release_tasks.adapters import ADAPTERS

    smoke = args.profile == 'smoke'
    with nullcontext() if smoke else gpu_lock():
        if not smoke:
            require_idle_gpu()
        device = torch.device('cpu') if smoke else require_cuda(args.task)
        os.environ['PROXY_CANDIDATE_MIXER'] = args.mixer
        part = ADAPTERS[args.task](spec, args.data_root/args.task)
        part.load(device)
        result, model = fit(part, candidate['model_id'], 0, device, 1 if smoke else part.steps)
        row = dict(status=result['status'], metrics={},
                   parameters=dict(total=sum(p.numel() for p in model.parameters()),
                                   block=sum(p.numel() for b in model.backbone.blocks for p in b.parameters()),
                                   mixer=sum(p.numel() for b in model.backbone.blocks for p in b.mixer.parameters())))
        if row['status'] == 'done' and not smoke:
            model.eval()
            row['metrics'][spec['metric']] = part.evaluate(model)['score']
        if smoke and result['history']:
            row['smoke_loss'] = result['history'][-1]['train_loss']
        return row


def cell(args):
    from proxy_tasks.common_set.plugins import describe
    from validation.suite import _cell

    candidate = describe(args.mixer)
    official = load_protocol()['tasks']
    if args.task in official:
        identity = canonical_hash({'candidate':candidate['sha256'], 'task':args.task,
                                   'profile':args.profile,'protocol':digest(PROTOCOL)})
        args.seeds, args.model_id, args.python = [0], candidate['model_id'], sys.executable
        args.stop_starting_at, args.allow_busy_gpu = None, False
        args.device = 'cpu' if args.profile == 'smoke' else 'cuda'
        args.timeout = 240 if args.profile == 'smoke' else 3600
        row = _cell(args, args.task, official[args.task], 0, identity)
    else:
        started = time.monotonic()
        spec = tasks()[args.task]
        try:
            row = extended_cell(args, spec, candidate)
        except (RuntimeError, ValueError, OSError) as error:
            row = dict(status='failed', error=str(error))
        row.update(task=args.task, seed=0, profile=args.profile,
                   protocol_sha256=digest(SPEC), wall_seconds=time.monotonic()-started,
                   evidence='smoke_only' if args.profile=='smoke' else 'measured',
                   budget=dict(profile=args.profile, configuration=spec))
    write_json(args.out/'result.json',row)
    return int(row['status']!='done')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest='action',required=True)
    d=actions.add_parser('data')
    d.add_argument('--data-root',type=Path,required=True)
    d.add_argument('--source',type=Path)
    c=actions.add_parser('check')
    c.add_argument('--mixer',required=True)
    c.add_argument('--out',type=Path,required=True)
    run=actions.add_parser('cell')
    run.add_argument('--task',required=True,choices=[*load_protocol()['tasks'],*tasks()])
    run.add_argument('--out',required=True,type=Path)
    run.add_argument('--mixer',required=True)
    run.add_argument('--data-root',required=True,type=Path)
    run.add_argument('--tfl-root',required=True,type=Path)
    run.add_argument('--profile',required=True,choices=['smoke','full'])
    args=parser.parse_args()
    return {'data':data,'check':check,'cell':cell}[args.action](args)


if __name__ == '__main__':
    raise SystemExit(main())
