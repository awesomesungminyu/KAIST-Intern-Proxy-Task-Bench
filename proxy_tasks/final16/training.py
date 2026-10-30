import math
import time
import torch
from ..common.seed import set_seed

def fit(part, mixer, seed, device, steps, sanity=False):
    set_seed(seed)
    model = part.build(mixer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=part.lr, weight_decay=part.wd)
    warm = max(1, int(part.warmup_frac * steps))
    g = torch.Generator(device='cpu' if part.gen_on_cpu else device).manual_seed(seed * 7919 + 13)
    res = {'part': part.name, 'n_params': model.num_params(), 'steps': steps, 'status': 'done'}
    losses, hist, times = ([], [], [])
    every = max(1, steps // 20)
    if sanity:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    last_progress = t0
    for step in range(steps):
        factor = (step + 1) / warm if step < warm else (steps - step) / max(1, steps - warm)
        for grp in opt.param_groups:
            grp['lr'] = part.lr * factor
        if sanity:
            torch.cuda.synchronize()
            ts = time.time()
        batch = part.sample(g)
        x, mask = part.inputs(batch)
        out = model(x, mask)
        loss = part.loss(out, batch)
        if not math.isfinite(loss.item()):
            res.update(status='nan', error=f'non-finite loss at step {step}')
            return (res, model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), part.clip)
        opt.step()
        losses.append(loss.item())
        if sanity:
            torch.cuda.synchronize()
            times.append(time.time() - ts)
            if step == 0:
                res['out_shape'] = list(out.shape)
        progress_now = time.time()
        if step == 0 or (step + 1) % every == 0 or progress_now - last_progress >= 30:
            elapsed = progress_now - t0
            print(f'PROGRESS part={part.name} mixer={mixer} seed={seed} step={step + 1}/{steps} elapsed_s={elapsed:.3f} mean_sec_per_step={elapsed / (step + 1):.6f}', flush=True)
            last_progress = progress_now
        if (step + 1) % every == 0:
            hist.append({'step': step + 1, 'train_loss': sum(losses[-every:]) / every})
    res['train_wall_s'] = time.time() - t0
    res['history'] = hist
    if sanity:
        k = max(1, min(10, len(losses) // 2))
        res.update(loss_first=sum(losses[:k]) / k, loss_last=sum(losses[-k:]) / k, sec_per_step=sorted(times[10:] or times)[len(times[10:] or times) // 2], peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        if mixer == 'retnet':
            from ..common_set import retention
            res['fla_used'] = retention.fla_chunk_retention.cache_info().currsize > 0
    return (res, model)
