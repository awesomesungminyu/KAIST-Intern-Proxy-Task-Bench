import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.paths import FFLM_ASSETS, OUTPUT_ROOT  # noqa: E402
from fflm.models import MODELS, model_for  # noqa: E402

DATA_DIR = Path(os.environ.get('FFLM_DATA_DIR', FFLM_ASSETS['data'])).resolve()


def fresh_batch(rng, batch=16, length=512, p_ignore=.8):
    pairs = length // 2
    ops = rng.choice(3, size=(batch, pairs), p=[(1-p_ignore)/2, (1-p_ignore)/2, p_ignore])
    ops[:, 0] = TOKENS['w']
    ops[:, -1] = TOKENS['r']
    bits = rng.integers(0, 2, size=(batch, pairs), dtype=np.int64)
    latest = np.maximum.accumulate(np.where(ops == TOKENS['w'], np.arange(pairs), 0), axis=1)
    state = np.take_along_axis(bits, latest, axis=1)
    bits = np.where(ops == TOKENS['r'], state, bits)
    seq = np.empty((batch, length), dtype=np.int64)
    seq[:, 0::2], seq[:, 1::2] = ops, bits + 3
    return seq


def check_oracle(seq):
    assert seq.shape[1] % 2 == 0
    assert np.all(seq[:, 0] == 0) and np.all(seq[:, -2] == 1)
    assert np.isin(seq[:, 0::2], [0, 1, 2]).all()
    assert np.isin(seq[:, 1::2], [3, 4]).all()
    state = np.zeros(len(seq), dtype=np.int64)
    for j in range(0, seq.shape[1], 2):
        op, bit = seq[:, j], seq[:, j+1]
        state = np.where(op == 0, bit, state)
        assert np.all(bit[op == 1] == state[op == 1])


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def lr_at(step):
    # Original LSTM retains the same 10001 endpoint as the Transformer.
    return 3e-4 * (step / 50 if step <= 50 else (10001-step) / (10001-50))


def objective(model, array, device):
    seq = torch.as_tensor(array, dtype=torch.long, device=device)
    x, y = seq[:, :-1], seq[:, 1:]
    mask = x == TOKENS['r']
    logits = model(x)
    return F.cross_entropy(logits[mask], y[mask])


@torch.no_grad()
def evaluate(model, data, device, batch_size=16, sample_limit=None):
    model.eval()
    total_reads = errors = bad_sequences = 0
    examples = []
    count = len(data) if sample_limit is None else min(sample_limit, len(data))
    sync(device); started = time.perf_counter()
    for start in range(0, count, batch_size):
        seq = torch.as_tensor(np.array(data[start:min(start+batch_size, count)], copy=True),
                              dtype=torch.long, device=device)
        x, y = seq[:, :-1], seq[:, 1:]
        pred = model(x).argmax(-1)
        mask = x == 1
        wrong = (pred != y) & mask
        total_reads += int(mask.sum())
        errors += int(wrong.sum())
        bad_sequences += int(wrong.any(-1).sum())
        if len(examples) < 20:
            for row, pos in wrong.nonzero().tolist()[:20-len(examples)]:
                examples.append({'sequence_index': start+row, 'read_position_zero_based': pos,
                                 'expected_token': int(y[row, pos]), 'predicted_token': int(pred[row, pos])})
    sync(device)
    return {'sequences': count, 'reads': total_reads, 'read_errors': errors,
            'read_error_rate': errors/total_reads, 'bad_sequences': bad_sequences,
            'exact_sequence_accuracy': (count-bad_sequences)/count,
            'elapsed_s': time.perf_counter()-started, 'first_errors': examples}


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def run(args, arch):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU run was not silently moved to CPU.')
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = model_for(arch, args.smoke, not args.no_flash).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.999),
                                 eps=1e-8, weight_decay=.1)
    rng = np.random.default_rng(args.data_seed)
    steps = args.steps or (500 if arch == 'lstm' else 10000)
    if args.mode == 'benchmark':
        steps = args.steps or 100
    elif args.smoke:
        steps = args.steps or 3
    cfg = {'arch': arch, 'steps': steps, 'mode': args.mode, 'smoke': args.smoke,
           'model_seed': args.seed, 'data_seed': args.data_seed, 'batch': args.batch,
           'eval_batch': args.eval_batch, 'precision': 'fp32', 'p_ignore_train': .8,
           'length': 512, 'tokens': TOKENS, 'flash_sdpa': not args.no_flash,
           'evaluation_splits': args.evaluation_splits,
           'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
           'weight_decay_policy': 'all model parameters', 'lstm_schedule_endpoint': 10001,
           'model_reconstruction': 'x-transformers pinned explicit pre-norm GELU learned absolute PE; unpublished author details may differ',
           'training_samples': 'fresh procedural samples; same distribution, not published train prefix rows',
           'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'eval_manifest': json.loads((DATA_DIR/'manifest.json').read_text())}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_id = 'E3-' + arch + '-' + args.mode + '-s' + str(args.seed) + '-' + stamp
    dest = args.out / run_id
    dest.mkdir(parents=True)
    (dest/'config.json').write_text(json.dumps(cfg, indent=2))
    env = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
           'device': str(device), 'python_threads': torch.get_num_threads(),
           'gpu': torch.cuda.get_device_name(device) if device.type=='cuda' else None,
           'x_transformers': importlib.metadata.version('x-transformers') if arch=='transformer' else None}
    (dest/'env.json').write_text(json.dumps(env, indent=2))
    started_utc = datetime.now(timezone.utc).isoformat()
    setup_end = time.perf_counter()
    train_s = online_eval_s = 0.
    timings = []
    status = 'running'
    print(run_id, 'parameters=',cfg['model_parameters'],flush=True)
    try:
        for step in range(1, steps+1):
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step)
            sync(device); t0 = time.perf_counter()
            loss = objective(model, fresh_batch(rng, args.batch), device)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            sync(device); elapsed = time.perf_counter()-t0
            train_s += elapsed
            if step > 10:
                timings.append(elapsed)
            if step == 1 or step % 100 == 0 or step == steps:
                row = {'step': step, 'loss': float(loss.detach()), 'lr': lr_at(step), 'train_s': train_s}
                append(dest/'history.jsonl', row)
                print(run_id, row,flush=True)
            if args.mode == 'train' and not args.smoke and step % 1000 == 0:
                data = np.load(DATA_DIR/'id.npy', mmap_mode='r')
                val = evaluate(model, data, device, args.eval_batch, 1000)
                online_eval_s += val['elapsed_s']
                append(dest/'history.jsonl', {'step': step, 'id_monitor': val})
                # Monitoring does not select a checkpoint or stop learning.
                checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'step': step, 'config': cfg, 'numpy_rng': rng.bit_generator.state,
                              'torch_rng': torch.get_rng_state(),
                              'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
                torch.save(checkpoint, dest/'checkpoint_latest.pt')
        tests = {}
        limit = 8 if args.smoke else None
        if args.mode == 'benchmark':
            data = np.load(DATA_DIR/'sparse.npy', mmap_mode='r')
            forward = evaluate(model, data, device, args.eval_batch, 160)
            mean = sum(timings)/len(timings) if timings else train_s/steps
            estimate = {'train_step_mean_s_after_warmup': mean,
                        'estimated_full_train_s': mean*(500 if arch=='lstm' else 10000),
                        'eval_seconds_per_sequence': forward['elapsed_s']/forward['sequences'],
                        'estimated_full_final_evaluation_s': forward['elapsed_s']/forward['sequences']*104000,
                        'note': 'Estimate excludes setup, save time and periodic ID evaluation.'}
            (dest/'benchmark.json').write_text(json.dumps(estimate,indent=2))
            print('BUDGET',arch,json.dumps(estimate),flush=True)
        else:
            torch.save({'model': model.state_dict(), 'config': cfg, 'step': steps}, dest/'final.pt')
            for name in args.evaluation_splits:
                data = np.load(DATA_DIR/ (name+'.npy'), mmap_mode='r')
                tests[name] = evaluate(model, data, device, args.eval_batch, limit)
                if limit is None:
                    assert tests[name]['reads'] == cfg['eval_manifest']['splits'][name]['reads']
                print('FINAL',arch,name,json.dumps(tests[name]),flush=True)
        result = {'run_id':run_id, 'exp':'E3-FFLM', 'task':'FFLM', 'status':'done',
                  'started_at':started_utc, 'config':cfg, 'env':env,
                  't_train_s':train_s, 't_online_eval_s':online_eval_s,
                  't_final_eval_s':sum(t['elapsed_s'] for t in tests.values()),
                  't_wall_s_after_setup':time.perf_counter()-setup_end,
                  'peak_gpu_mem_mb':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                  'result':tests,
                  'joint_ood_zero':all(tests[n]['read_errors']==0 for n in ['sparse','dense']) if all(n in tests for n in ['sparse','dense']) else None}
        (dest/'result.json').write_text(json.dumps(result,indent=2))
        append(args.out/'runs.jsonl', result)
    except BaseException as exc:
        result = {'run_id':run_id,'exp':'E3-FFLM','status':'failed', 'error':repr(exc), 'config':cfg}
        (dest/'failure.json').write_text(json.dumps(result,indent=2))
        append(args.out/'runs.jsonl',result)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arch',choices=[*MODELS.names(),'both'],default='both')
    p.add_argument('--mode',choices=['benchmark','train'],default='train')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--data-seed',type=int,default=1000)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval-batch',type=int,default=16)
    p.add_argument('--steps',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-flash',action='store_true')
    p.add_argument('--evaluation-splits', nargs='+', choices=['id','sparse','dense'], default=['id','sparse','dense'])
    p.add_argument('--out',type=Path,default=OUTPUT_ROOT/'fflm'/'runs')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for arch in (['lstm','transformer'] if args.arch=='both' else [args.arch]):
        run(args,arch)


if __name__=='__main__':
    main()
