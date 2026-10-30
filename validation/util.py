from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def command_output(command, cwd=None, timeout=30):
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or f'Command failed: {command}')
    return proc.stdout.strip()

def fingerprint(paths):
    """Logical names omit machine-specific absolute paths; contents define identity."""
    records = []
    for label, path in sorted(paths.items()):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else sorted((p for p in path.rglob('*') if p.is_file()))
        for f in files:
            if any((x in {'__pycache__', '.git'} for x in f.relative_to(path.parent).parts)):
                continue
            rel = f.name if path.is_file() else str(f.relative_to(path))
            records.append({'path': f'{label}/{rel}', 'sha256': digest(f)})
    return {'sha256': canonical_hash(records), 'files': records}

def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)):
        raise ValueError(f'Invalid metric {name}: {value!r}')
    return value

@contextmanager
def gpu_lock():
    import fcntl
    lock_path = Path(tempfile.gettempdir()) / f'proxy-task-bench-gpu-{os.getuid()}.lock'
    with lock_path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another proxy validation run holds the GPU lock') from exc
        yield

def require_idle_gpu():
    text = command_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
    for line in text.splitlines():
        memory, utilization = [int(x.strip()) for x in line.split(',')]
        if memory > 1024 or utilization > 10:
            raise RuntimeError('GPU is busy; retry later without interrupting the existing job')

def execute(command, cwd, env, log, timeout):
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'Trainer exited with {code}; inspect {log}')

def load_runs(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'summary.json'
    doc = json.loads(path.read_text())
    if 'runs' in doc:
        if doc.get('status') != 'done':
            raise ValueError('Experiment summary is not complete')
        if 'requested_seeds' in doc and sorted((r['seed'] for r in doc['runs'])) != sorted(doc['requested_seeds']):
            raise ValueError('Experiment summary is missing requested seeds')
    rows = doc.get('runs', [doc])
    if not rows:
        raise ValueError('No runs to compare')
    return rows

def compare(baseline, candidate):
    left, right = (load_runs(baseline), load_runs(candidate))
    for rows in [left, right]:
        if any((r.get('status') != 'done' or r.get('profile') != 'full' or r.get('evidence') != 'measured' for r in rows)):
            raise ValueError('Only completed full measured runs can be compared; smoke/failure is not performance evidence')
        if len({r['seed'] for r in rows}) != len(rows):
            raise ValueError('Duplicate seeds')
        if len({r['model'] for r in rows}) != 1 or len({r['provenance']['source']['sha256'] for r in rows}) != 1:
            raise ValueError('One side mixes models or source revisions')
    if {r['seed'] for r in left} != {r['seed'] for r in right}:
        raise ValueError('Baseline and candidate seed sets differ')
    reference = left[0]
    for r in left + right:
        for key in ['task', 'protocol_sha256', 'budget', 'metric', 'selection']:
            if r[key] != reference[key]:
                raise ValueError(f'Incompatible {key}')
        for key in ['data', 'environment', 'evaluation_sha256']:
            if r['provenance'][key] != reference['provenance'][key]:
                raise ValueError(f'Incompatible {key}')
    if reference['metric']['name'] == 'multi_metric':
        definitions = reference['metric_definitions']
        if any((r.get('metric_definitions') != definitions for r in left + right)):
            raise ValueError('Incompatible metric definitions')
        descriptions = {}
        for metric in definitions:
            name = metric['name']
            a = [finite(r['metrics'][name], name) for r in left]
            b = [finite(r['metrics'][name], name) for r in right]
            descriptions[name] = {**metric, 'baseline_mean': statistics.mean(a), 'baseline_std_population': statistics.pstdev(a), 'candidate_mean': statistics.mean(b), 'candidate_std_population': statistics.pstdev(b)}
        return {'schema_version': 1, 'task': reference['task'], 'seeds': sorted((r['seed'] for r in left)), 'metrics': descriptions, 'limitations': reference['limitations'], 'interpretation': 'Descriptive per-metric statistics only; no aggregate improvement or cross-tier prediction.'}
    name = reference['metric']['name']
    a = {r['seed']: finite(r['metrics'][name], name) for r in left}
    b = {r['seed']: finite(r['metrics'][name], name) for r in right}
    deltas = [b[s] - a[s] for s in sorted(a)]
    delta = statistics.mean(deltas)
    improvement = delta if reference['metric']['direction'] == 'maximize' else -delta
    return {'schema_version': 1, 'task': reference['task'], 'metric': reference['metric'], 'seeds': sorted(a), 'baseline_mean': statistics.mean(a.values()), 'baseline_std_population': statistics.pstdev(a.values()), 'candidate_mean': statistics.mean(b.values()), 'candidate_std_population': statistics.pstdev(b.values()), 'paired_candidate_minus_baseline': deltas, 'mean_improvement': improvement, 'evidence': 'single_seed_pilot' if len(a) < 3 else 'repeated_seed_measurement', 'limitations': reference['limitations'], 'interpretation': 'Descriptive matched-budget comparison only; no statistical significance or generalization claim is inferred.'}
