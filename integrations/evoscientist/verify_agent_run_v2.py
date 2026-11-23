"""verify_agent_run v2: stricter evidence of agent-initiated harness use.

Derived from integrations/evoscientist/verify_agent_run.py (untracked at 036e3c8).
Reads a run folder strictly read-only and writes its evidence file somewhere else
(--evidence-out, refused if it lies inside the run folder). No model, GPU or
evaluation is used.

Changes against v1
  (a) "next candidate" = write_file/edit_file on candidates/gen1/*.py after the first
      successful report read. Edits of candidates/gen0/*.py after an evaluation are a
      separate, informational item (error_fix_applied_to_gen0).
  (b) "report cited" = at least --min-pairs distinct (field, value) pairs of the gen0
      report.json files ON DISK found in proposals/gen1.md (or, if that file is absent,
      in the agent text and write bodies after the first report read). A bare field name
      never counts. Pairs of common-word fields (status, profile, ...) and pairs already
      spelled out in the run prompt / skills are "weak" and not counted.
  (c) every evaluate.sh call records tool_call line, tool_result line, subagent, and the
      mtime of runs/<dir>/precheck.json and report.json, and checks that those mtimes lie
      inside the wall-clock window of that call. events.jsonl has no timestamps, so the
      window is taken from the gateway transcript: start = mtime of the request whose
      response issued the tool call, end = mtime of the first later request that carries
      the tool result. The report's candidate.path/sha256 are compared with the command.
      The exit status is read from the full tool result in the gateway request, because
      events.jsonl truncates long results and drops the 'exit code 0' trailer (with v1 a
      successful 16-task evaluation therefore always counted as "not exited 0").
  (d) verdict: PASS / NOT PASSED (capability or failure) / NOT REACHED (LLM budget or
      wall-clock exhausted before the criteria were met). While the run has no terminal
      heartbeat the verdict is PENDING with a provisional class (use --final to force).
  Also: v1 only looked at type == 'tool_call'. EvoScientist emits sub-agent tool use as
  'subagent_tool_call' / 'subagent_tool_result'; v2 reads both.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

WRITE_TOOLS = ('write_file', 'edit_file')
SHELL_TOOLS = ('execute', 'run_in_background')
CALL_TYPES = ('tool_call', 'subagent_tool_call')
RESULT_TYPES = ('tool_result', 'subagent_tool_result')
TEXT_TYPES = ('text', 'subagent_text')
# Field names that are ordinary words: a (field, value) hit on them proves nothing.
WEAK_FIELDS = frozenset(('status', 'profile', 'task', 'name', 'tier', 'mean', 'rank', 'path', 'clip',
                         'source', 'formula', 'ideal', 'schema_version', 'class_name', 'reference'))
TERMINAL_PHASES = ('done', 'failed', 'cutoff', 'candidate_limit')
SLACK_SECONDS = 2.0
NUMBER = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])')


def load_events(path):
    events = []
    if not path.exists():
        return events
    for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            event['_line'] = number
            events.append(event)
    return events


def iso(timestamp):
    return None if timestamp is None else datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def workspace_relative(target, workspace):
    """File tools are rooted at the workspace ('/x' == <workspace>/x); shells use real paths."""
    target = str(target).strip().strip('\'"')
    prefix = str(workspace)
    if target.startswith(prefix + '/'):
        target = target[len(prefix) + 1:]
    while target.startswith('./'):
        target = target[2:]
    return target.lstrip('/')


# ---------------------------------------------------------------- (b) report pairs
def flatten(value, leaf=None):
    """Yield (leaf field name, scalar or list-of-scalars) for every leaf of a report."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, key)
    elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        for item in value:
            yield from flatten(item, leaf)
    elif leaf is not None:
        yield leaf, value


def value_in(window, value):
    if isinstance(value, bool):
        match = re.search(r'(?<![A-Za-z])(true|false)(?![A-Za-z])', window, re.I)
        return bool(match) and (match.group(1).lower() == 'true') is value
    if value is None:
        return bool(re.search(r'(?<![A-Za-z])(null|none)(?![A-Za-z])', window[:40], re.I))
    if isinstance(value, (int, float)):
        match = NUMBER.search(window)
        if not match:
            return False
        token = match.group(0)
        if isinstance(value, int) or float(value).is_integer():
            return float(token) == float(value)
        decimals = len(token.split('.')[1].split('e')[0].split('E')[0]) if '.' in token else 0
        return decimals >= 1 and abs(round(value, decimals) - float(token)) < 1e-9
    if isinstance(value, str):
        return 2 <= len(value) <= 80 and value.lower() in window.lower()
    if isinstance(value, list):
        if not value:
            return bool(re.search(r'\[\s*\]|(?<![\w.])0(?![\w.])|(?<![A-Za-z])(none|empty)(?![A-Za-z])|없음', window[:40], re.I))
        return all(str(item) in window for item in value)
    return False


def pair_in(text, field, value):
    """True if `field` occurs as a token and `value` follows it on the same line."""
    width = 200 if isinstance(value, (str, list)) else 60
    for match in re.finditer(r'(?<![A-Za-z0-9_])' + re.escape(field) + r'(?![A-Za-z0-9_])', text):
        if value_in(text[match.end():match.end() + width].split('\n')[0], value):
            return True
    return False


def pair_key(field, value):
    return field, json.dumps(value, ensure_ascii=False, sort_keys=True)


def cited_pairs(corpus, reports, known_text):
    """Return (strong, weak, names_only) for the pairs of the on-disk reports found in corpus."""
    strong, weak, seen = [], [], set()
    names = set()
    for report_path, report in reports:
        for field, value in flatten(report):
            names.add(field)
            key = pair_key(field, value)
            if key in seen or not pair_in(corpus, field, value):
                continue
            seen.add(key)
            row = dict(field=field, value=value, report=report_path)
            if field in WEAK_FIELDS:
                weak.append(dict(row, weak_reason='common-word field'))
            elif pair_in(known_text, field, value):
                weak.append(dict(row, weak_reason='pair already written in run prompt/skills'))
            else:
                strong.append(row)
    matched = {row['field'] for row in strong + weak}
    names_only = sorted(field for field in names - matched
                        if re.search(r'(?<![A-Za-z0-9_])' + re.escape(field) + r'(?![A-Za-z0-9_])', corpus))
    return strong, weak, names_only


# ---------------------------------------------------------------- (c) call windows
def gateway_index(folder):
    """Per counted model call: request mtime, response mtime, response text, tool results carried."""
    rows = []
    for request in sorted(folder.glob('call_*_request.json')) if folder.is_dir() else []:
        number = int(request.name.split('_')[1])
        response = folder / f'call_{number:03d}_response.txt'
        carried = {}
        try:
            for message in json.loads(request.read_text(errors='replace')).get('messages', []):
                if message.get('role') == 'tool' and message.get('tool_call_id'):
                    carried[message['tool_call_id']] = str(message.get('content', ''))
        except (json.JSONDecodeError, AttributeError):
            pass
        rows.append(dict(number=number, request_mtime=request.stat().st_mtime,
                         response_mtime=response.stat().st_mtime if response.exists() else None,
                         response_text=response.read_text(errors='replace') if response.exists() else '',
                         carried=carried))
    return rows


def call_window(tool_id, gateway, events_mtime, has_result):
    """Wall-clock window in which the tool call with `tool_id` can have run."""
    issued = next((row for row in gateway if tool_id and tool_id in row['response_text']), None)
    returned = next((row for row in gateway if tool_id in row['carried']
                     and (issued is None or row['number'] > issued['number'])), None)
    window = dict(issued_by_llm_call=issued and issued['number'], returned_in_llm_call=returned and returned['number'],
                  start=None, end=None, start_source=None, end_source=None, response_complete=None)
    if issued:
        window.update(start=issued['request_mtime'], start_source=f"mtime call_{issued['number']:03d}_request.json",
                      response_complete=issued['response_mtime'])
    if returned:
        window.update(end=returned['request_mtime'], end_source=f"mtime call_{returned['number']:03d}_request.json")
    elif has_result and events_mtime is not None:
        # The result line is in events.jsonl, so the call ended before the file's last write (upper bound only).
        window.update(end=events_mtime, end_source='mtime events.jsonl (upper bound; result never sent to the model)')
    return window


def inside(mtime, window):
    if mtime is None or window['start'] is None or window['end'] is None:
        return None
    return window['start'] - SLACK_SECONDS <= mtime <= window['end'] + SLACK_SECONDS


def evaluate_arguments(command):
    match = re.search(r'evaluate\.sh["\']?\s+(\S+)\s+(\S+)', command)
    if not match:
        return None, None
    return tuple(part.strip('\'";&|') for part in match.groups())


def candidate_sha256(path, class_name):
    """Same identity as proxy_tasks/common_set/plugins.describe: sha256(file bytes + NUL + class name)."""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes() + b'\0' + str(class_name).encode()).hexdigest()


def full_result_text(tool_id, event_content, gateway):
    """events.jsonl truncates long tool results (~2000 chars, '... (truncated)'), which cuts off the
    '[Command succeeded with exit code 0]' trailer. The request sent to the model has the full text."""
    for row in gateway:
        if tool_id in row['carried']:
            return row['carried'][tool_id], 'gateway request'
    return event_content, 'events.jsonl'


# ---------------------------------------------------------------- (d) exhaustion
def run_state(run, now):
    limits = json.loads((run / 'limits.json').read_text()) if (run / 'limits.json').exists() else {}
    budget = json.loads((run / 'llm_budget.json').read_text()) if (run / 'llm_budget.json').exists() else {}
    beats = []
    if (run / 'heartbeat.log').exists():
        for line in (run / 'heartbeat.log').read_text().splitlines():
            try:
                beats.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    last_phase = beats[-1].get('phase') if beats else None
    errors = (run / 'gateway/errors.log').read_text(errors='replace') if (run / 'gateway/errors.log').exists() else ''
    calls, cap = budget.get('calls'), limits.get('llm_cap')
    deadline = datetime.fromisoformat(limits['deadline']) if limits.get('deadline') else None
    budget_exhausted = calls is not None and cap is not None and calls >= cap
    time_exhausted = last_phase == 'cutoff' or (deadline is not None and now >= deadline and last_phase not in ('done', 'failed', 'candidate_limit'))
    return dict(llm_calls=calls, llm_cap=cap, budget_exhausted=budget_exhausted,
                cap_refusals_logged=errors.count('cap exhausted'),
                deadline=limits.get('deadline'), time_exhausted=time_exhausted, last_heartbeat_phase=last_phase,
                last_heartbeat_time=beats[-1].get('time') if beats else None,
                terminal=last_phase in TERMINAL_PHASES or time_exhausted,
                candidate_limit_hit=last_phase == 'candidate_limit')


def analyse(run, now=None, min_pairs=2, final=False):
    run = Path(run).resolve()
    workspace = run / 'workspace'
    now = now or datetime.now(timezone.utc)
    events_path = run / 'events.jsonl'
    events = load_events(events_path)
    events_mtime = events_path.stat().st_mtime if events_path.exists() else None
    gateway = gateway_index(run / 'gateway')
    results = {e['id']: e for e in events if e.get('type') in RESULT_TYPES and e.get('id')}
    calls = [e for e in events if e.get('type') in CALL_TYPES]

    gen0_writes, gen1_writes, proposal_writes, evaluations, report_reads = [], [], [], [], []
    for call in calls:
        arguments = call.get('args') or {}
        result = results.get(call.get('id'), {})
        content = str(result.get('content', ''))
        target = workspace_relative(arguments.get('file_path') or arguments.get('path') or '', workspace)
        command = str(arguments.get('command') or '')
        subagent = call.get('subagent') or ('(unnamed)' if call['type'] == 'subagent_tool_call' else '')
        row = dict(line=call['_line'], result_line=result.get('_line'), tool=call.get('name'), subagent=subagent,
                   is_subagent=bool(subagent))
        name = call.get('name')
        if name in WRITE_TOOLS:
            generation = re.fullmatch(r'candidates/gen(\d+)/[^/]+\.py', target)
            entry = dict(row, file=target, on_disk=(workspace / target).is_file(),
                         tool_error=content.lower().startswith('error') or not result,
                         result_head=content[:160])
            entry['accepted'] = entry['on_disk'] and not entry['tool_error']
            if generation and generation.group(1) == '0':
                gen0_writes.append(entry)
            elif generation and generation.group(1) == '1':
                gen1_writes.append(entry)
            elif target == 'proposals/gen1.md':
                proposal_writes.append(entry)
        if name in SHELL_TOOLS and 'evaluate.sh' in command:
            candidate_arg, out_arg = evaluate_arguments(command)
            out_dir = workspace / workspace_relative(out_arg, workspace) if out_arg else None
            window = call_window(call.get('id'), gateway, events_mtime, bool(result))
            full, full_source = full_result_text(call.get('id'), content, gateway)
            exit_0 = 'succeeded with exit code 0' in full
            exit_failed = bool(re.search(r'Command failed with exit code|Exit code: [1-9]|timed out', full))
            suite_path = out_dir / 'suite.json' if out_dir else None
            suite_status = None
            if suite_path and suite_path.is_file():
                try:
                    suite_status = json.loads(suite_path.read_text()).get('status')
                except json.JSONDecodeError:
                    pass
            entry = dict(row, command=command, candidate_arg=candidate_arg, out_dir=str(out_dir) if out_dir else None,
                         background=name == 'run_in_background',
                         exit_0=exit_0, exit_failed=exit_failed, exit_text_source=full_source,
                         events_result_truncated='(truncated)' in content[-40:],
                         # Trailer lost (truncated in events and never sent to the model): fall back to suite.json.
                         exit_0_by_disk=(not exit_0 and not exit_failed and suite_status == 'done'),
                         suite_status_on_disk=suite_status, result_tail=full[-400:],
                         window={k: (iso(v) if k in ('start', 'end', 'response_complete') else v) for k, v in window.items()})
            for label in ('precheck', 'report'):
                path = out_dir / f'{label}.json' if out_dir else None
                mtime = path.stat().st_mtime if path and path.is_file() else None
                entry[f'{label}_mtime'] = iso(mtime)
                entry[f'{label}_mtime_in_window'] = inside(mtime, window)
            report_path = out_dir / 'report.json' if out_dir else None
            entry['report_candidate_matches_command'] = None
            entry['report_candidate_sha_matches_disk'] = None
            if report_path and report_path.is_file() and candidate_arg:
                try:
                    candidate = json.loads(report_path.read_text()).get('candidate', {})
                    expected = workspace / workspace_relative(candidate_arg, workspace)
                    entry['report_candidate_matches_command'] = Path(candidate.get('path', '')) == expected
                    entry['report_candidate_sha_matches_disk'] = candidate.get('sha256') == candidate_sha256(
                        expected, candidate.get('class_name', 'CandidateMixer'))
                except json.JSONDecodeError:
                    entry['report_candidate_matches_command'] = False
            # A background launch returns before the evaluation finishes. Its report
            # cannot lie inside the launch tool's short call window. The agent's
            # requested command writes a sibling .exit file after evaluate.sh ends.
            # Attribute the report only when that marker says success and both
            # files were created after this launch, with a matching candidate.
            if name == 'run_in_background' and out_dir:
                marker = Path(str(out_dir) + '.exit')
                entry['background_exit_marker'] = str(marker)
                entry['background_exit_code'] = marker.read_text().strip() if marker.is_file() else None
                report_mtime = report_path.stat().st_mtime if report_path and report_path.is_file() else None
                marker_mtime = marker.stat().st_mtime if marker.is_file() else None
                started = window['start']
                entry['background_report_after_launch'] = bool(
                    started is not None and report_mtime is not None and marker_mtime is not None
                    and started - SLACK_SECONDS <= report_mtime <= marker_mtime + SLACK_SECONDS)
                entry['attributed_to_this_call'] = bool(
                    entry['background_exit_code'] == '0' and entry['background_report_after_launch']
                    and entry['report_candidate_matches_command'] and entry['report_candidate_sha_matches_disk']
                    and suite_status == 'done')
            else:
                entry['attributed_to_this_call'] = bool((entry['exit_0'] or entry['exit_0_by_disk']) and entry['report_mtime_in_window']
                                                        and entry['report_candidate_matches_command'])
            evaluations.append(entry)
        reads_file = name == 'read_file' and re.fullmatch(r'runs/[^/]+/report\.json', target)
        reads_shell = name in SHELL_TOOLS and 'report.json' in command and 'evaluate.sh' not in command
        if reads_file or reads_shell:
            full, _ = full_result_text(call.get('id'), content, gateway)
            report_reads.append(dict(row, target=target or command,
                                     returned_report='performance_comparable' in full, result_head=content[:300]))

    first_read = min((r['result_line'] for r in report_reads if r['returned_report'] and r['result_line']), default=None)
    first_evaluation_result = min((e['result_line'] for e in evaluations if e['result_line']), default=None)
    next_candidates = [w for w in gen1_writes if w['accepted'] and first_read is not None and w['line'] > first_read]
    gen0_fixes = [w for w in gen0_writes if first_evaluation_result is not None and w['line'] > first_evaluation_result]

    # (b) corpus and on-disk gen0 reports
    reports = []
    for path in sorted((workspace / 'runs').glob('gen0*/report.json')):
        try:
            reports.append((str(path.relative_to(workspace)), json.loads(path.read_text())))
        except json.JSONDecodeError:
            pass
    proposal = workspace / 'proposals/gen1.md'
    if proposal.is_file():
        corpus, corpus_source = proposal.read_text(errors='replace'), 'proposals/gen1.md'
    elif first_read is not None:
        parts = []
        for event in events:
            if event['_line'] <= first_read:
                continue
            if event.get('type') in TEXT_TYPES:
                parts.append(str(event.get('content', '')))
            elif event.get('type') in CALL_TYPES and event.get('name') in WRITE_TOOLS:
                arguments = event.get('args') or {}
                parts.append('\n' + str(arguments.get('content') or arguments.get('new_string') or '') + '\n')
        corpus, corpus_source = ''.join(parts), f'agent text and write bodies after events.jsonl line {first_read}'
    else:
        corpus, corpus_source = '', 'none (no proposals/gen1.md and no successful report read)'
    known = []
    for path in [workspace / 'evaluate.sh', *sorted((workspace / 'prompts').glob('*.md')), *sorted((workspace / 'skills').glob('*/SKILL.md'))]:
        if path.is_file():
            known.append(path.read_text(errors='replace'))
    strong, weak, names_only = cited_pairs(corpus, reports, '\n'.join(known))

    state = run_state(run, now)
    criteria = {
        'agent_wrote_gen0_candidate': any(w['accepted'] for w in gen0_writes),
        'agent_shell_called_evaluate_sh': bool(evaluations),
        'agent_evaluation_exit0_report_in_call_window': any(e['attributed_to_this_call'] for e in evaluations),
        'agent_read_report_json': first_read is not None,
        'gen1_candidate_written_after_report_read': bool(next_candidates),
        'report_field_value_pairs_cited': len(strong) >= min_pairs,
    }
    informational = {
        'error_fix_applied_to_gen0': bool(gen0_fixes),
        'proposals_gen1_md_written_by_agent_tool': any(w['accepted'] for w in proposal_writes),
        'evaluate_called_by_subagent': any(e['is_subagent'] for e in evaluations),
        'gen1_candidate_evaluated': any(e['attributed_to_this_call'] and '/runs/gen1' in (e['out_dir'] or '') for e in evaluations),
        'reports_on_disk_not_attributed_to_any_agent_call': sorted(
            str(p.relative_to(workspace)) for p in (workspace / 'runs').rglob('report.json')
            if str(p.parent) not in {e['out_dir'] for e in evaluations if e['attributed_to_this_call']}),
    }
    missing = [name for name, met in criteria.items() if not met]
    exhausted = state['budget_exhausted'] or state['time_exhausted']
    if not missing:
        verdict, reason = 'PASS', 'all criteria met by the agent\'s own tool calls'
    else:
        if state['candidate_limit_hit']:
            klass, reason = 'NOT PASSED', 'run terminated by the candidate/generation limit'
        elif exhausted:
            klass = 'NOT REACHED'
            reason = ('LLM budget exhausted' if state['budget_exhausted'] else 'wall-clock window exhausted') + \
                     f" ({state['llm_calls']}/{state['llm_cap']} calls) before: " + ', '.join(missing)
        else:
            klass, reason = 'NOT PASSED', 'run ended with budget and time left; unmet: ' + ', '.join(missing)
        if state['terminal'] or state['budget_exhausted'] or final:
            verdict = klass
        else:
            verdict, reason = 'PENDING', f'run not finished; if it stopped now: {klass}; unmet so far: ' + ', '.join(missing)
    return dict(
        verdict=verdict, reason=reason, criteria=criteria, informational=informational, run_state=state,
        verdict_classes='PASS | NOT PASSED (capability/failure) | NOT REACHED (LLM budget or time exhausted) | PENDING (run still going)',
        report_citation=dict(corpus_source=corpus_source, min_pairs=min_pairs, gen0_reports_used=[p for p, _ in reports],
                             strong_pairs=strong, weak_pairs_not_counted=weak, field_names_without_value=names_only),
        gen0_candidate_writes=gen0_writes, gen1_candidate_writes=gen1_writes, proposal_writes=proposal_writes,
        next_candidates_after_first_report_read=next_candidates, gen0_fixes_after_first_evaluation=gen0_fixes,
        evaluate_calls=evaluations, report_reads=report_reads, first_report_read_result_line=first_read,
        candidates_on_disk=sorted(str(p.relative_to(workspace)) for p in (workspace / 'candidates').rglob('gen*/*.py')),
        reports_on_disk=sorted(str(p.relative_to(workspace)) for p in (workspace / 'runs').rglob('report.json')),
        tool_call_counts={name: sum(c.get('name') == name for c in calls) for name in sorted({str(c.get('name')) for c in calls})},
        subagent_tool_calls=sum(c['type'] == 'subagent_tool_call' for c in calls),
        gateway_calls_indexed=len(gateway), analysed_at=now.isoformat(), run=str(run),
        note='Smoke profile has no performance score; this verdict is about the connection only.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True, help='Run folder (read only)')
    parser.add_argument('--evidence-out', type=Path, required=True, help='Evidence JSON; must be outside --run')
    parser.add_argument('--min-pairs', type=int, default=2)
    parser.add_argument('--final', action='store_true', help='Treat the run as finished even without a terminal heartbeat')
    parser.add_argument('--now', help='ISO time used for the deadline test (tests only)')
    args = parser.parse_args(argv)
    run, target = args.run.resolve(), args.evidence_out.resolve()
    if target == run or run in target.parents:
        parser.error('--evidence-out must not be inside the run folder')
    now = datetime.fromisoformat(args.now) if args.now else None
    summary = analyse(run, now=now, min_pairs=args.min_pairs, final=args.final)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(dict(verdict=summary['verdict'], reason=summary['reason'], criteria=summary['criteria'],
                          informational=summary['informational'],
                          llm_calls=f"{summary['run_state']['llm_calls']}/{summary['run_state']['llm_cap']}",
                          strong_pairs=len(summary['report_citation']['strong_pairs']),
                          evaluate_calls=len(summary['evaluate_calls'])), indent=2, ensure_ascii=False))
    return {'PASS': 0, 'NOT PASSED': 1, 'NOT REACHED': 3, 'PENDING': 4}[summary['verdict']]


if __name__ == '__main__':
    sys.exit(main())
