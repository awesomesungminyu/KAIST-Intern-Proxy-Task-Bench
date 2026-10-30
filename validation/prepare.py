"""Prepare local, supplied datasets into data/<task ID> without downloads or training."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from proxy_tasks.common.paths import data_root
from .runner import load_protocol

# These are layouts inside the supplied distribution archives, never server paths.
SOURCES = {
    'T1-R1-05': ['mad_tsc/t-C_vs-16_sl-32_ntr-12800_nte-1280_km-1_vm-1_mq-1_fn-0#0_nvs-0_ntc-0_s-12345/train'],
    'T1-R0-04': ['parity', 'Parity'],
    'T2-R1-11': ['t2k_mnist'],
    'T2-R3-05': ['round4/R10', 'round4/T2-R3-05'],
    'T2-R4-02': ['round4_diversity/T2-R4-02', 'speech_commands_v0.02'],
    'T2-R2-07': ['round2/n10_beijing/x', 'round2/n10_beijing'],
    'T2-R1-04': ['t2d_protein/stability'],
    'T2-R1-09': ['t2i_enwik8'],
    'T2-R1-08': ['t2h_imdb'],
    'T2-R2-02': ['round2/t2m_lipo'],
}


def locate(source, task):
    for suffix in [task, *SOURCES.get(task, [])]:
        direct = source / suffix
        if direct.is_dir():
            return direct
        matches = sorted(p for p in source.rglob(Path(suffix).name)
                         if p.is_dir() and p.as_posix().endswith('/' + suffix))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f'{task}: multiple source folders; pass their common dataset folder directly')
    return source


def speech_cache(source, target):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from concurrent.futures import ThreadPoolExecutor
    words = sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith('_'))
    if len(words) != 35:
        raise ValueError('Speech Commands requires the supplied 35-class raw waveform folder')
    validation = set((source / 'validation_list.txt').read_text().split())
    testing = set((source / 'testing_list.txt').read_text().split())
    rows = dict(train=[], val=[])
    for label, word in enumerate(words):
        for path in sorted((source / word).glob('*.wav')):
            name = word + '/' + path.name
            if name not in testing:
                rows['val' if name in validation else 'train'].append((path, label))
    def read(row):
        rate, signal = wavfile.read(row[0])
        if rate != 16000 or signal.ndim != 1:
            raise ValueError('Expected mono 16kHz source audio')
        waveform = np.zeros(16000, np.float32)
        signal = signal[:16000].astype(np.float32) / 32768
        waveform[:len(signal)] = signal
        return resample_poly(waveform, 1, 8).astype(np.float32)
    values = {}
    with ThreadPoolExecutor(8) as executor:
        for split in rows:
            values[split] = np.stack(list(executor.map(read, rows[split])))
    mean, std = float(values['train'].mean()), float(values['train'].std())
    for split in rows:
        np.save(target / f'{split}_x.npy', ((values[split] - mean) / std)[..., None])
        np.save(target / f'{split}_y.npy', np.array([row[1] for row in rows[split]]))


def prepare_task(source, destination, task, spec):
    assets = spec['assets']
    if not assets:
        return 'generated at runtime'
    if all((destination / name).is_file() for name in assets):
        return 'already prepared'
    if destination.exists():
        raise FileExistsError(f'{task}: incomplete destination exists; use a new --data-root or move that folder before retrying')
    source = locate(source, task)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prepare-', dir=destination.parent) as tmp:
        stage = Path(tmp) / task
        stage.mkdir()
        if all((source / name).is_file() for name in assets):
            for name in assets:
                shutil.copy2(source / name, stage / name)
        elif task in {'T2-R3-05', 'T2-R1-08', 'T2-R2-02'}:
            raw = {'T2-R3-05': ['imgs.npz', 'labels.npz'], 'T2-R1-08': ['train.parquet'],
                   'T2-R2-02': ['Lipophilicity.csv']}[task]
            for name in raw:
                shutil.copy2(source / name, stage / name)
            from proxy_tasks.final16.catalog import build_part
            part = build_part(task)
            part.data_dir_override = stage
            part.prepare()
        elif task == 'T2-R2-07':
            name = assets[0]
            with zipfile.ZipFile(source / 'beijing_pm25.zip') as archive:
                matches = [p for p in archive.namelist() if Path(p).name == name]
                if len(matches) != 1:
                    raise ValueError('Expected one Beijing PM2.5 CSV in supplied archive')
                with archive.open(matches[0]) as stream, (stage / name).open('wb') as out:
                    shutil.copyfileobj(stream, out)
        elif task == 'T2-R4-02':
            speech_cache(source, stage)
        else:
            missing = [name for name in assets if not (source / name).is_file()]
            raise FileNotFoundError(f'{task}: supplied source is missing {missing}. See docs/DATA.md')
        if not all((stage / name).is_file() for name in assets):
            raise ValueError(f'{task}: preprocessing did not produce the required files')
        stage.rename(destination)
    return 'prepared'


def prepare(args):
    if not args.source.is_dir():
        raise NotADirectoryError('Extract the supplied archives first and pass that directory with --source')
    root = args.data_root or data_root()
    failed = False
    for task, spec in load_protocol()['tasks'].items():
        if args.task != 'all' and task != args.task:
            continue
        try:
            status = prepare_task(args.source, root / task, task, spec)
            row = dict(task=task, status=status)
        except Exception as exc:
            failed = True
            row = dict(task=task, status='failed', error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(row), flush=True)
    return int(failed)
