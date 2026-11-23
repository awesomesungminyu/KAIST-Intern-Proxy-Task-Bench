"""CPU-only executable candidate contract checks; no optimizer or datasets changed."""
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from .util import canonical_hash, digest, write_json


def configurations(data_root):
    from .runner import load_protocol
    from proxy_tasks.final16.catalog import build_part
    result = {}
    for task, spec in load_protocol()['tasks'].items():
        if spec['adapter'] == 'part':
            part = build_part(task)
            if data_root:
                part.data_dir_override = Path(data_root) / task
            result[task] = dict(d_model=part.d_model, n_layers=part.n_layers,
                                num_heads=part.n_heads, l_max=part.l_max)
        else:
            result[task] = (dict(d_model=128, n_layers=8, num_heads=2, l_max=256)
                            if spec['adapter'] == 'native' else
                            dict(d_model=32, n_layers=2, num_heads=4, l_max=128))
    return result


def check_configuration(cls, config, gpu=False):
    import torch
    from proxy_tasks.common_set.backbone import Block
    from proxy_tasks.common_set.mixers import CausalSelfAttention
    torch.manual_seed(0)
    kwargs = {k: config[k] for k in ['d_model', 'num_heads', 'l_max']}
    mixer = cls(**kwargs).cpu()
    d, h = kwargs['d_model'], kwargs['num_heads']
    count = lambda m: sum(p.numel() for p in m.parameters())
    reference = CausalSelfAttention(d, h)
    raw_count, raw_ref = count(mixer), count(reference)
    # Frozen repository budget is Block, not mixer-only (see PLAN.md).
    block_count, block_ref = count(Block(d, mixer)), count(Block(d, reference))
    checks = dict(construction=True, parameter_budget=abs(block_count / block_ref - 1) <= .10)
    detail = dict(mixer_parameters=raw_count, transformer_mixer_parameters=raw_ref,
                  mixer_ratio=raw_count / raw_ref, block_parameters=block_count,
                  transformer_block_parameters=block_ref, block_ratio=block_count / block_ref)
    for mode in ['eval', 'train']:
        getattr(mixer, mode)()
        for length in sorted({min(17, kwargs['l_max']), min(41, kwargs['l_max'])}):
            torch.manual_seed(length)
            x = torch.randn(2, length, d, requires_grad=True)
            y = mixer(x)
            checks[f'{mode}_{length}_shape'] = tuple(y.shape) == tuple(x.shape)
            checks[f'{mode}_{length}_finite'] = bool(torch.isfinite(y).all())
            if not checks[f'{mode}_{length}_shape'] or not checks[f'{mode}_{length}_finite']:
                continue
            mixer.zero_grad(set_to_none=True)
            y.square().mean().backward()
            grads = [p.grad for p in mixer.parameters() if p.requires_grad and p.grad is not None]
            checks[f'{mode}_{length}_backward'] = bool(grads and x.grad is not None and
                torch.isfinite(x.grad).all() and all(torch.isfinite(g).all() for g in grads))
            # Re-seed each call so dropout does not masquerade as future leakage.
            for cut in [1, length // 2, length - 1]:
                changed = x.detach().clone(); changed[:, cut:] = torch.randn_like(changed[:, cut:]) * 7
                with torch.no_grad():
                    torch.manual_seed(123); a = mixer(x.detach())[:, :cut]
                    torch.manual_seed(123); b = mixer(changed)[:, :cut]
                checks[f'{mode}_{length}_causal_{cut}'] = bool(torch.allclose(a, b, atol=1e-5, rtol=1e-5))
    gpu_result = dict(status='unmeasured', reason='CUDA unavailable or disabled')
    if gpu:
        mixer.eval().cuda(); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, kwargs['l_max'], d, device='cuda', requires_grad=True)
        start = time.monotonic(); y = mixer(x); y.square().mean().backward(); torch.cuda.synchronize()
        checks['gpu_max_length_finite'] = bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
        gpu_result = dict(status='measured', length=kwargs['l_max'], seconds=time.monotonic()-start,
                          peak_bytes=torch.cuda.max_memory_allocated())
        mixer.cpu(); torch.cuda.empty_cache()
    return dict(passed=all(checks.values()), checks=checks, **detail, gpu=gpu_result)


def run_precheck(reference, output, data_root=None, gpu=False):
    import torch
    from proxy_tasks.common_set.plugins import describe, load
    from .runner import PROTOCOL, ROOT
    torch.set_num_threads(max(1, min(2, (os.cpu_count() or 2)//2)))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    document = dict(schema_version=1, status='failed', candidate=None, tasks={},
        parameter_rule='Block (mixer + identical RMSNorm/SwiGLU) within 10%; raw mixer ratio is diagnostic',
        protocol_sha256=digest(PROTOCOL), checked_at=datetime.now(timezone.utc).isoformat())
    # Implementation identity makes stale prechecks invalid after harness changes.
    document['harness_sha256'] = canonical_hash({str(p.relative_to(ROOT)):digest(p)
        for folder in ['validation','proxy_tasks'] for p in (ROOT/folder).rglob('*.py')})
    try:
        document['candidate'] = describe(reference)
        cls = load(reference)
        configs = configurations(data_root)
        cache = {}
        for task, config in configs.items():
            key=canonical_hash(config)
            if key not in cache:
                try: cache[key] = check_configuration(cls, config, gpu=gpu)
                except Exception as exc: cache[key] = dict(passed=False,error=f'{type(exc).__name__}: {exc}')
            document['tasks'][task] = dict(configuration=config, **cache[key])
        document['status'] = 'passed' if all(r['passed'] for r in document['tasks'].values()) else 'failed'
    except Exception as exc:
        document['error'] = f'{type(exc).__name__}: {exc}'
    document['wall_seconds'] = time.monotonic()-started
    write_json(output, document)
    return document
