"""SC-Raw data audit and GPU plumbing check; not a benchmark training run."""
import argparse
import ast
import hashlib
import json
import random
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

BASE = Path('/data/mqar')
DATA = BASE / 'data/SpeechCommands/speech_commands_v0.02'
WANTED = sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])
LABELS = WANTED + ['unknown', 'silence']
SPLITS = ('training', 'validation', 'testing')


def pcm(path):
    with wave.open(str(path), 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError(f'Expected 16 kHz mono PCM16: {path}')
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float32) / 32768.0


def make_records():
    valid = set((DATA / 'validation_list.txt').read_text().splitlines())
    test = set((DATA / 'testing_list.txt').read_text().splitlines())
    all_files = {str(p.relative_to(DATA)) for p in DATA.glob('*/*_nohash_*.wav')}
    assert not valid & test
    assert valid <= all_files and test <= all_files
    walkers = {'training': all_files - valid - test, 'validation': valid, 'testing': test}
    noises = sorted((DATA / '_background_noise_').glob('*.wav'))
    assert noises
    noise_lengths = {}
    for p in noises:
        with wave.open(str(p), 'rb') as f:
            assert f.getframerate() == 16000 and f.getnchannels() == 1
            assert f.getsampwidth() == 2 and f.getnframes() >= 16000
            noise_lengths[p] = f.getnframes()
    records = {}
    for split_id, split in enumerate(SPLITS):
        walker = walkers[split]
        known = sorted(p for p in walker if Path(p).parent.name in WANTED)
        unknown = sorted(p for p in walker if Path(p).parent.name not in WANTED)
        rng = random.Random(456 + split_id)
        selected = rng.sample(unknown, min(len(unknown), round(len(known) * 0.1)))
        rows = [dict(path=p, label=Path(p).parent.name, kind='speech') for p in known]
        rows += [dict(path=p, label='unknown', kind='speech') for p in selected]
        # Training/validation replicate the existing baseline's seed-456 silence schedule.
        # Testing uses an explicit new seed; testing is audited but never trained/evaluated here.
        rng = random.Random(456 + 1000 * (split_id + 1))
        for i in range(round(len(known) * 0.1)):
            p = noises[i % len(noises)]
            offset = rng.randint(0, noise_lengths[p] - 16000)
            amplitude = rng.uniform(0.02, 0.15)
            rows.append(dict(path=str(p.relative_to(DATA)), label='silence', kind='noise_crop',
                             offset=offset, amplitude=amplitude))
        records[split] = rows
    return records


class RawDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.noises = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if row['kind'] == 'noise_crop':
            if row['path'] not in self.noises:
                self.noises[row['path']] = pcm(DATA / row['path'])
            start = row['offset']
            x = self.noises[row['path']][start:start + 16000] * row['amplitude']
        else:
            x = pcm(DATA / row['path'])[:16000]
            x = np.pad(x, (0, 16000 - len(x)))
        assert x.shape == (16000,) and np.isfinite(x).all()
        return torch.from_numpy(x.copy()).unsqueeze(0), LABELS.index(row['label'])


def audit(records, out):
    # Extract only the existing dataset class; avoid running scripts with training or writes.
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
    from scipy.io import wavfile
    src = BASE / 'validate_speechcommands12.py'
    tree = ast.parse(src.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'SC12':
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'ROOT', 'DATA_ROOT', 'WANTED', 'SAMPLE_RATE', 'SEED'} for t in node.targets):
            selected.append(node)
    env = dict(Path=Path, random=random, torch=torch, SPEECHCOMMANDS=SPEECHCOMMANDS)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(src), 'exec'), env)
    result = dict(status='passed', purpose='data validation, not model quality', dataset='speech_commands_v0.02',
                  input_shape=[1, 16000], labels=LABELS, model_seed=456, selection_seed=456,
                  reference_source=str(src), reference_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  input='unfiltered PCM16 / 32768; end zero-pad or crop to 16000; no MFCC/log-Mel',
                  silence_policy='Replicates existing seed-456 baseline noise crops for training/validation; shared background recordings across splits may overlap; no leakage-free claim.',
                  test_policy='Data integrity checks only; test accuracy is not computed.', splits={})
    speech_paths = {}
    speakers = {}
    for split in SPLITS:
        rows = records[split]
        speech = [r for r in rows if r['kind'] == 'speech']
        reference = env['SC12'](split)
        paths = [str(DATA / r['path']) for r in speech]
        assert paths == reference._walker, f'{split}: original dataset path/order mismatch'
        cache = torch.load(BASE / f'data/speechcommands12_cache/{split}.pt', map_location='cpu', weights_only=True)
        labels = [LABELS.index(r['label']) for r in speech]
        assert cache['labels'] == LABELS
        assert labels == cache['targets'].tolist(), f'{split}: original cached targets mismatch'
        dataset = RawDataset(rows)
        indices = sorted(set(np.linspace(0, len(rows)-1, 48).astype(int).tolist() +
                         [next(i for i, r in enumerate(rows) if r['label'] == label) for label in LABELS]))
        for i in indices:
            x, y = dataset[i]
            row = rows[i]
            sr, raw = wavfile.read(DATA / row['path'])
            assert sr == 16000 and raw.dtype == np.int16
            independent = raw.astype(np.float32) / 32768.0
            if row['kind'] == 'noise_crop':
                independent = independent[row['offset']:row['offset']+16000] * row['amplitude']
            else:
                independent = np.pad(independent[:16000], (0, max(0, 16000-len(independent))))
            assert np.array_equal(x.numpy()[0], independent)
            assert y == LABELS.index(row['label'])
        speech_paths[split] = set(paths)
        speakers[split] = {Path(p).stem.split('_nohash_')[0] for p in paths}
        result['splits'][split] = dict(examples=len(rows), speech_examples=len(speech),
            class_counts=dict(Counter(r['label'] for r in rows)), independent_wav_checks=len(indices),
            original_speech_paths_and_order_match=True, original_cached_targets_match=True)
        print('AUDIT', split, json.dumps(result['splits'][split]), flush=True)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            assert not speech_paths[a] & speech_paths[b], f'shared speech paths {a}/{b}'
            assert not speakers[a] & speakers[b], f'shared speech speakers {a}/{b}'
    result['speech_paths_and_speakers_disjoint'] = True
    result['torch_version'] = torch.__version__
    result['torchaudio_version'] = torchaudio.__version__
    (out / 'data_audit.json').write_text(json.dumps(result, indent=2))
    (out / 'manifest.json').write_text(json.dumps(dict(labels=LABELS, data_root=str(DATA), splits=records)))
    return result


def smoke(records, out):
    assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
    torch.set_num_threads(2)
    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    rows = records['training']
    # Exactly one example per class; same batch reused to check forward/backward plumbing.
    indices = [next(i for i, r in enumerate(rows) if r['label'] == label) for label in LABELS]
    dataset = RawDataset(rows)
    batch = [dataset[i] for i in indices]
    x = torch.stack([p[0] for p in batch]).cuda()
    y = torch.tensor([p[1] for p in batch], device='cuda')
    model = nn.Sequential(nn.Conv1d(1, 16, 9, padding=4), nn.GELU(),
                          nn.Conv1d(16, 16, 9, dilation=4, padding=16), nn.GELU(),
                          nn.Conv1d(16, 16, 9, dilation=16, padding=64), nn.GELU(),
                          nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, len(LABELS))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.003)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    history = []
    for step in range(30):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        assert torch.isfinite(loss), 'non-finite loss'
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
        history.append(loss.item())
    torch.cuda.synchronize()
    assert history[-1] < history[0], 'loss did not decrease on the fixed diagnostic batch'
    result = dict(status='passed', purpose='GPU plumbing only; not a baseline or benchmark result',
        model='tiny stride-1 Conv1d diagnostic; global mean only after sequence layers',
        input_shape=list(x.shape), sequence_length_at_convolution_layers=16000,
        training_updates=30, train_examples=12, validation_used=False, test_used=False,
        first_loss=history[0], final_loss=history[-1], loss_history=history,
        parameters=sum(p.numel() for p in model.parameters()),
        elapsed_seconds=time.perf_counter()-start,
        peak_vram_mib=torch.cuda.max_memory_allocated()/1024**2,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    (out / 'gpu_smoke.json').write_text(json.dumps(result, indent=2))
    print('GPU_SMOKE', json.dumps({k:v for k,v in result.items() if k != 'loss_history'}), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        records = make_records()
        audit(records, args.out)
        if args.smoke:
            smoke(records, args.out)
    except Exception as e:
        (args.out / 'failure.json').write_text(json.dumps(dict(type=type(e).__name__, error=str(e)), indent=2))
        raise


if __name__ == '__main__':
    main()
