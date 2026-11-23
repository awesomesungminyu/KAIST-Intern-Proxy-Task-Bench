"""Launch an agent-led EvoScientist session with a local-only counted LLM gateway.

This runner does not generate or evaluate candidates itself. EvoScientist must
read workspace skills, write candidates, call ./evaluate.sh, and inspect reports.

All bounds are run arguments (no fixed calendar cutoff): wall-clock minutes from
the first start of an output folder, generations, candidates per generation and
counted LLM requests. The start time and the call counter persist in the output
folder, so a restart or --resume cannot reset either bound.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrations.evoscientist.adapter import LOCAL, limited_gateway

KST = timezone(timedelta(hours=9))
ROOT = Path(os.environ.get('EVO_WORK_ROOT', Path(__file__).resolve().parents[2] / '.evo-work')).resolve()
ENV_FILE = ROOT / '.env'
# EvoScientist features that issue model requests outside the visible agent turn.
# They are switched off so the counted budget is spent on the agent loop itself.
QUIET_CONFIG = (('memory_workers_enabled', 'false'), ('memory_observations_enabled', 'false'),
                ('memory_skill_synthesis_enabled', 'false'), ('memory_profile_enabled', 'false'),
                ('enable_scheduler', 'false'), ('enable_async_subagents', 'false'))


def read_local_key():
    """Read only the named local .env; never import or source arbitrary settings."""
    if not ENV_FILE.is_file():
        raise FileNotFoundError(f'Missing required {ENV_FILE}; agent run skipped')
    if ENV_FILE.stat().st_mode & 0o077:
        raise PermissionError(f'{ENV_FILE} must be chmod 600')
    fields = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'(CUSTOM_OPENAI_API_KEY|CUSTOM_OPENAI_BASE_URL)=(.*)', line)
        if match:
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            fields[match.group(1)] = value
    key = fields.get('CUSTOM_OPENAI_API_KEY', '')
    if not key or key.startswith('<'):
        raise ValueError(f'{ENV_FILE} has no usable CUSTOM_OPENAI_API_KEY')
    endpoint = fields.get('CUSTOM_OPENAI_BASE_URL', LOCAL).rstrip('/')
    from urllib.parse import urlsplit
    target = urlsplit(endpoint)
    if (target.scheme != 'https' or not target.hostname or target.path != '/v1'
            or target.username or target.password or target.query or target.fragment):
        raise ValueError('Configure an explicit HTTPS OpenAI-compatible /v1 endpoint')
    if LOCAL and endpoint != LOCAL:
        raise ValueError('Configured gateway endpoint differs from the allowed endpoint')
    return endpoint, key


def counts(workspace):
    candidates = list((workspace / 'candidates').rglob('*.py'))
    by_generation = {}
    for path in candidates:
        parent = path.parent.name
        if re.fullmatch(r'gen\d+', parent):
            by_generation[parent] = by_generation.get(parent, 0) + 1
    reports = list((workspace / 'runs').rglob('report.json'))
    return by_generation, len(reports)


def write_heartbeat(out, budget, workspace, phase):
    calls = json.loads(budget.read_text()).get('calls', 0) if budget.exists() else 0
    by_generation, completed = counts(workspace)
    row = dict(time=datetime.now(KST).isoformat(), phase=phase,
               generations=sorted(by_generation), candidate_counts=by_generation,
               llm_calls=calls, evaluations_completed=completed)
    with (out / 'heartbeat.log').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def run_window(out, max_minutes):
    """Return (start, deadline). The first launch of an output folder fixes the start."""
    record = out / 'run_window.json'
    if record.exists():
        start = datetime.fromisoformat(json.loads(record.read_text())['start'])
    else:
        start = datetime.now(KST)
        record.write_text(json.dumps(dict(start=start.isoformat(), max_minutes=max_minutes)))
    return start, start + timedelta(minutes=max_minutes)


def stop(process):
    # EvoScientist's run_in_background starts a new process group. Capture only
    # this agent's descendants before stopping the parent so training cannot
    # continue after the run deadline or call budget is reached.
    import psutil
    groups = {process.pid}
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            try:
                groups.add(os.getpgid(child.pid))
            except ProcessLookupError:
                pass
    except psutil.NoSuchProcess:
        pass
    groups.discard(os.getpgrp())
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help=f'Output folder under {ROOT}')
    parser.add_argument('--prompt', type=Path, required=True, help='Run prompt file inside <out>/workspace')
    parser.add_argument('--profile', choices=['smoke', 'full'], default='smoke')
    parser.add_argument('--max-minutes', type=float, default=90, help='Hard wall-clock limit from the first start of --out')
    parser.add_argument('--max-generations', type=int, default=2)
    parser.add_argument('--max-candidates', type=int, default=2, help='Candidate files per generation')
    parser.add_argument('--llm-cap', type=int, default=30, help='Counted model requests, including retries and subagents')
    parser.add_argument('--model', default='deepseek-v4-flash', help='Model id on the laboratory endpoint')
    parser.add_argument('--evosci', type=Path, default=ROOT / 'venv/bin/EvoSci',
                        help='EvoSci executable; its folder is put first on PATH. Must be inside ' + str(ROOT))
    parser.add_argument('--no-force-model', action='store_true',
                        help='Do not rewrite the request model id in the gateway (default: rewrite to --model)')
    parser.add_argument('--resume', help='EvoScientist thread ID from the prior run, if restarting')
    parser.add_argument('--compact-agent', action='store_true',
                        help='Use the four permitted EvoScientist tools with a short local-model system message')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if ROOT.resolve() not in out.parents:
        raise ValueError(f'--out must be inside {ROOT}')
    if min(args.max_minutes, args.max_generations, args.max_candidates, args.llm_cap) <= 0:
        raise ValueError('All limits must be positive')
    workspace = out / 'workspace'
    prompt = args.prompt.resolve()
    if workspace not in prompt.parents:
        raise ValueError('--prompt must be inside <out>/workspace')
    executable = args.evosci.absolute()
    if ROOT.resolve() not in executable.resolve().parents:
        raise ValueError(f'--evosci must be inside {ROOT}')
    for path in (workspace / 'skills/proxy-task-bench/SKILL.md', workspace / 'evaluate.sh', prompt):
        if not path.is_file():
            raise FileNotFoundError(f'Missing prepared workspace file: {path}')
    endpoint, key = read_local_key()
    if not executable.is_file():
        raise FileNotFoundError(f'EvoScientist not installed in allowed venv: {executable}')
    if args.profile == 'full':
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('GPU full requested but nvidia-smi failed')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU full requested but PyTorch CUDA is unavailable')
    for directory in ('home', 'config', 'cache', 'tmp', 'state'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    start, deadline = run_window(out, args.max_minutes)
    if datetime.now(KST) >= deadline:
        raise RuntimeError(f'Run window of this output folder ended at {deadline.isoformat()}')
    budget = out / 'llm_budget.json'
    environment = dict(PATH=f'{executable.parent}:/usr/local/bin:/usr/bin:/bin',
                       HOME=str(out / 'home'), XDG_CONFIG_HOME=str(out / 'config'),
                       XDG_CACHE_HOME=str(out / 'cache'), TMPDIR=str(out / 'tmp'),
                       PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                       EVOSCIENTIST_DATA_DIR=str(out / 'state'),
                       EVOSCIENTIST_WORKSPACE_DIR=str(workspace),
                       EVO_EVAL_PROFILE=args.profile, EVO_ML_PYTHON=sys.executable,
                       EVO_TRAINING_CUTOFF=deadline.isoformat(),
                       LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S='240',
                       PYTHONDONTWRITEBYTECODE='1', LANG='C.UTF-8')
    if args.profile == 'smoke':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    limits = dict(start=start.isoformat(), deadline=deadline.isoformat(), max_minutes=args.max_minutes,
                  max_generations=args.max_generations, max_candidates=args.max_candidates,
                  llm_cap=args.llm_cap, model=args.model, profile=args.profile, prompt=str(prompt),
                  evosci=str(executable), force_model=not args.no_force_model)
    (out / 'limits.json').write_text(json.dumps(limits, indent=2))
    with limited_gateway(endpoint, key, budget, args.llm_cap, transcript=out / 'gateway',
                         force_model=None if args.no_force_model else args.model,
                         compact_agent=args.compact_agent) as gateway:
        environment['CUSTOM_OPENAI_BASE_URL'] = gateway
        environment['CUSTOM_OPENAI_API_KEY'] = 'local-gateway'
        for name, value in (('provider', 'custom-openai'), ('model', args.model)) + QUIET_CONFIG:
            subprocess.run([str(executable), 'config', 'set', name, value], cwd=ROOT / 'evoscientist_work',
                           env=environment, check=True, capture_output=True, text=True, timeout=60)
        # stream-json: stdout is one JSON event per line (tool calls, tool results, text).
        command = [str(executable), '--workdir', str(workspace), '--ui', 'cli', '--auto-mode',
                   '--output-format', 'stream-json']
        if args.resume:
            command += ['--resume', args.resume]
        command += ['-p', prompt.read_text()]
        write_heartbeat(out, budget, workspace, 'starting')
        with (out / 'events.jsonl').open('a') as events, (out / 'evoscientist.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT / 'evoscientist_work', env=environment,
                                       stdout=events, stderr=log, start_new_session=True)
            last_heartbeat = 0.0
            for number in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(number, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
            try:
                while process.poll() is None:
                    if datetime.now(KST) >= deadline:
                        stop(process)
                        write_heartbeat(out, budget, workspace, 'cutoff')
                        return 124
                    by_generation, _ = counts(workspace)
                    if len(by_generation) > args.max_generations or any(n > args.max_candidates for n in by_generation.values()):
                        stop(process)
                        write_heartbeat(out, budget, workspace, 'candidate_limit')
                        return 125
                    refusals = out / 'gateway/errors.log'
                    if refusals.exists() and 'cap exhausted' in refusals.read_text(errors='replace'):
                        # The gateway refused a request: nothing more can happen, so do not let
                        # EvoScientist sit in its retry loop until the wall-clock limit.
                        stop(process)
                        write_heartbeat(out, budget, workspace, 'llm_cap')
                        return 126
                    if time.monotonic() - last_heartbeat >= 60:
                        write_heartbeat(out, budget, workspace, 'running')
                        last_heartbeat = time.monotonic()
                    time.sleep(5)
            finally:
                if process.poll() is None:
                    stop(process)
            write_heartbeat(out, budget, workspace, 'done' if process.returncode == 0 else 'failed')
            return process.returncode or 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
