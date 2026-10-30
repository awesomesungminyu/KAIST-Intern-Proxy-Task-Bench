"""One optimizer step per selected task using supplied data; no evaluation or checkpoints."""
import argparse
import gc
import json
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from validation.runner import load_protocol
from proxy_tasks.final16.catalog import build_part


def one_step(task, spec, data_root, tfl_root, device):
    torch.manual_seed(0)
    if spec['adapter'] == 'part':
        part = build_part(task)
        part.data_dir_override = data_root / task
        part.batch_size = 2
        if hasattr(part, 'n_eval'):
            part.n_eval = 8
        part.load(device)
        model = part.build('transformer').to(device)
        generator = torch.Generator(device='cpu' if part.gen_on_cpu else device).manual_seed(13)
        batch = part.sample(generator)
        loss = part.loss(model(*part.inputs(batch)), batch)
    elif spec['adapter'] == 'native':
        sys.path.insert(0, str(ROOT / 'proxy_tasks'))
        from T1_R0_07.data import RegBenchDataConfig, RegBenchDataset, collate_fn
        from T1_R0_07.models import build_model
        from T1_R0_07.pfa import PFAConfig, vocab_size_with_specials
        config = PFAConfig.repo_default()
        dataset = RegBenchDataset(RegBenchDataConfig(config, max_total_len=64, virtual_epoch_size=2), 0, 'train')
        batch = collate_fn([dataset[0], dataset[1]])
        model = build_model('cs_transformer', vocab_size_with_specials(config), 64).to(device)
        output = model(batch['input_ids'].to(device))
        loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), batch['labels'].to(device).reshape(-1))
    else:
        sys.path.insert(0, str(tfl_root))
        from src.utils.helper import Voc
        from src.dataloader import Sampler
        from proxy_tasks.common_set.tfl import CommonSetTFLModel
        with (data_root / task / 'train_corpus.pk').open('rb') as stream:
            corpus = pickle.load(stream)
        vocabulary = Voc()
        for character in sorted(set(''.join(corpus.source))):
            vocabulary.add_word(character)
        source, target, lengths = Sampler(corpus, vocabulary, 2).get_batch(0)
        model = CommonSetTFLModel(vocabulary.nwords, corpus.noutputs, 'transformer', 32, 4, 2).to(device)
        output = model(source.to(device))
        mask = (torch.arange(source.shape[0])[:, None] < lengths[None, :]).to(device)
        loss = ((output - target.to(device)).square().mean(-1) * mask).sum() / mask.sum()
    if not torch.isfinite(loss):
        raise ValueError('Non-finite loss')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise ValueError('Missing or non-finite gradients')
    before = next(p.detach().clone() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    after = next(p.detach() for p in model.parameters() if p.grad is not None)
    if torch.equal(before, after):
        raise ValueError('Optimizer did not update parameters')
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=lambda p: Path(p).resolve(), default=ROOT / 'data')
    parser.add_argument('--tfl-root', type=lambda p: Path(p).resolve(), default=ROOT / 'third_party/tfl')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--task', default='all', choices=['all', *load_protocol()['tasks']])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == 'cuda':
        from validation.util import gpu_lock, require_idle_gpu
        context = gpu_lock()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    rows = []
    with context:
        if args.device == 'cuda':
            require_idle_gpu()
        for task, spec in load_protocol()['tasks'].items():
            if args.task != 'all' and task != args.task:
                continue
            started = time.monotonic()
            try:
                loss = one_step(task, spec, args.data_root, args.tfl_root, args.device)
                row = dict(task=task, status='passed', steps=1, loss=loss)
            except Exception as exc:
                row = dict(task=task, status='failed', error=f'{type(exc).__name__}: {exc}')
            row['seconds'] = round(time.monotonic() - started, 3)
            rows.append(row)
            print(json.dumps(row), flush=True)
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
    print(json.dumps(dict(passed=sum(r['status'] == 'passed' for r in rows), total=len(rows))), flush=True)
    return int(any(row['status'] != 'passed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
