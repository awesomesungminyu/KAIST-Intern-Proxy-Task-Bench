"""Synthetic-run tests for verify_agent_run_v2.py. Everything is created under
audit_20261112/fixtures/; no real run folder, model, GPU or evaluation is touched.

Cases (first four were requested): normal / gen0_only_edit / field_names_only /
budget_exhausted, plus foreign_report (report.json older than the call window) and
still_running (no terminal heartbeat).
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

# Repository copy of audit_20261112/test_verify_v2.py. HERE/FIXTURES are redirected to a
# temporary folder by tests/test_verify_agent_run_v2.py, so nothing is written in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integrations/evoscientist'))
import verify_agent_run_v2 as v2  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
BASE = 1_800_000_000.0  # arbitrary fixed epoch
PROMPT = ('# Fake run prompt\nreport.json has `performance_comparable=false` and null scores.\n'
          'Evaluate with bash ./evaluate.sh ./candidates/gen0/<name>.py ./runs/gen0_<name>\n')
CANDIDATE = 'class CandidateMixer:\n    """fake %s"""\n'


def report_for(path):
    return {
        'schema_version': 1, 'status': 'done', 'profile': 'smoke', 'performance_comparable': False,
        'candidate': {'path': str(path), 'class_name': 'CandidateMixer',
                      'sha256': hashlib.sha256(path.read_bytes() + b'\0CandidateMixer').hexdigest(), 'model_id': 'cand_1a2b3c4d'},
        'tasks': [{'task': 'T1-R0-04', 'name': 'Parity', 'tier': 1, 'status': 'done', 'mean': None, 'rank': None,
                   'baseline_incomplete': False, 'ranking_provisional': True, 'wall_seconds': 5.614473961}],
        'summary': {'requested_tasks': 16, 'scored_tasks': 0, 'complete_tasks': 16, 'tier1_mean': None,
                    'tier2_mean': None, 'overall_mean': None, 'mean_rank': None, 'provisional': True},
        'failures': [], 'timeouts': [], 'not_run': []}


class Run:
    """Builds events.jsonl + gateway transcript with consistent tool ids and mtimes."""

    def __init__(self, name):
        self.root = FIXTURES / name
        if self.root.exists():
            shutil.rmtree(self.root)
        self.workspace = self.root / 'workspace'
        for folder in ('candidates', 'runs', 'prompts', 'skills/proxy-task-bench'):
            (self.workspace / folder).mkdir(parents=True)
        (self.root / 'gateway').mkdir()
        (self.workspace / 'prompts/run_request.md').write_text(PROMPT)
        (self.workspace / 'skills/proxy-task-bench/SKILL.md').write_text('Run precheck, suite, report.\n')
        (self.workspace / 'evaluate.sh').write_text('#!/usr/bin/env bash\n')
        self.events, self.clock, self.llm_call, self.pending, self.counter = [], 0.0, 0, [], 0

    def stamp(self, path, seconds):
        os.utime(path, (BASE + seconds, BASE + seconds))

    def turn(self, text, tool_calls, seconds=3.0):
        """One model call: request carries earlier tool results, response issues tool_calls."""
        self.llm_call += 1
        request = self.root / f'gateway/call_{self.llm_call:03d}_request.json'
        messages = [{'role': 'user', 'content': PROMPT}] + [{'role': 'tool', 'tool_call_id': i, 'content': c} for i, c in self.pending]
        request.write_text(json.dumps({'model': 'fake', 'messages': messages}))
        self.stamp(request, self.clock)
        self.clock += seconds
        ids = []
        for name, args in tool_calls:
            self.counter += 1
            ids.append(f'call_fake{self.counter:04d}')
        response = self.root / f'gateway/call_{self.llm_call:03d}_response.txt'
        response.write_text(''.join(f'data: {{"id":"{i}"}}\n\n' for i in ids) or 'data: [DONE]\n')
        self.stamp(response, self.clock)
        for chunk in (text[:len(text) // 2], text[len(text) // 2:]):  # streamed text arrives in pieces
            self.events.append({'type': 'text', 'content': chunk})
        for i, (name, args) in zip(ids, tool_calls):
            self.events.append({'type': 'tool_call', 'name': name, 'id': i, 'args': args})
        return ids

    def result(self, tool_id, name, content, seconds=1.0):
        self.clock += seconds
        shown = content if len(content) <= 2000 else content[:1985] + '\n... (truncated)'  # as EvoScientist does
        self.events.append({'type': 'tool_result', 'name': name, 'id': tool_id, 'content': shown, 'success': True})
        self.pending.append((tool_id, content))

    def write(self, relative, body, tool='write_file', text='Writing.'):
        (tool_id,) = self.turn(text, [(tool, {'file_path': '/' + relative, 'content': body})])
        path = self.workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        self.result(tool_id, tool, f'Updated file /{relative}')
        return path

    def evaluate(self, candidate, out, report_at=None, exit_code=0):
        """report_at: seconds on the fixture clock for report.json; default = inside the call."""
        (tool_id,) = self.turn('Evaluating.', [('execute', {'command': f'bash ./evaluate.sh ./{candidate} ./{out}', 'timeout': 900})])
        folder = self.workspace / out
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'precheck.json').write_text(json.dumps({'status': 'passed'}))
        self.stamp(folder / 'precheck.json', self.clock + 10)
        (folder / 'report.json').write_text(json.dumps(report_for(self.workspace / candidate), indent=1))
        self.stamp(folder / 'report.json', self.clock + 90 if report_at is None else report_at)
        tail = '[Command succeeded with exit code 0]' if exit_code == 0 else f'Exit code: {exit_code}\n[Command failed with exit code {exit_code}]'
        lines = ''.join(json.dumps({'task': f'T{i}', 'model': 'cand_1a2b3c4d', 'seed': 0, 'status': 'done', 'error': None,
                                    'pad': 'x' * 60}) + '\n' for i in range(16))  # >2000 chars, like a real 16-task run
        (folder / 'suite.json').write_text(json.dumps({'status': 'done' if exit_code == 0 else 'incomplete'}))
        self.result(tool_id, 'execute', lines + '{"status": "done"}\n' + tail, seconds=95)

    def read_report(self, out):
        (tool_id,) = self.turn('Reading the report.', [('read_file', {'file_path': f'/{out}/report.json'})])
        self.result(tool_id, 'read_file', (self.workspace / out / 'report.json').read_text())

    def finish(self, calls, cap=30, phase='done', cap_error=False):
        self.turn('Final summary written. Stopping.', [])
        (self.root / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in self.events))
        self.stamp(self.root / 'events.jsonl', self.clock + 1)
        (self.root / 'llm_budget.json').write_text(json.dumps({'calls': calls}))
        (self.root / 'limits.json').write_text(json.dumps({'llm_cap': cap, 'deadline': '2099-01-01T00:00:00+09:00'}))
        beats = [{'time': 't0', 'phase': 'running'}] + ([{'time': 't1', 'phase': phase}] if phase else [])
        (self.root / 'heartbeat.log').write_text(''.join(json.dumps(b) + '\n' for b in beats))
        if cap_error:
            (self.root / 'gateway/errors.log').write_text('ValueError: Local LLM call cap exhausted\n')
        return self.root


GOOD_PROPOSAL = ('# gen1 proposal\nThe gen0 report says `requested_tasks`: 16, complete_tasks = 16 and scored_tasks=0;\n'
                 '`failures`: [] and timeouts: []. tier1_mean is null, performance_comparable=false, status: done.\n'
                 'wall_seconds on Parity was 5.61. So the next candidate keeps the interface.\n')
NAMES_ONLY_PROPOSAL = ('# gen1 proposal\nI looked at status, profile, performance_comparable, failures, tier1_mean and\n'
                       'complete_tasks in the report. These fields shaped the next design: a gated variant.\n')


def gen0(run):
    run.write('candidates/gen0/alpha.py', CANDIDATE % 'alpha')
    run.evaluate('candidates/gen0/alpha.py', 'runs/gen0_alpha')
    run.read_report('runs/gen0_alpha')


def case_normal():
    run = Run('normal'); gen0(run)
    run.write('proposals/gen1.md', GOOD_PROPOSAL)
    run.write('candidates/gen1/beta.py', CANDIDATE % 'beta')
    run.evaluate('candidates/gen1/beta.py', 'runs/gen1_beta'); run.read_report('runs/gen1_beta')
    return run.finish(calls=9)


def case_gen0_only_edit():
    run = Run('gen0_only_edit'); gen0(run)
    run.write('proposals/gen1.md', GOOD_PROPOSAL)
    run.write('candidates/gen0/alpha.py', CANDIDATE % 'alpha fixed', tool='edit_file')
    run.evaluate('candidates/gen0/alpha.py', 'runs/gen0_alpha'); run.read_report('runs/gen0_alpha')
    return run.finish(calls=9)


def case_field_names_only():
    run = Run('field_names_only'); gen0(run)
    run.write('proposals/gen1.md', NAMES_ONLY_PROPOSAL)
    run.write('candidates/gen1/beta.py', CANDIDATE % 'beta')
    run.evaluate('candidates/gen1/beta.py', 'runs/gen1_beta'); run.read_report('runs/gen1_beta')
    return run.finish(calls=9)


def case_budget_exhausted():
    run = Run('budget_exhausted'); gen0(run)
    return run.finish(calls=30, phase='failed', cap_error=True)


def case_foreign_report():
    """report.json exists but was written long before the agent's call (another process made it)."""
    run = Run('foreign_report')
    run.write('candidates/gen0/alpha.py', CANDIDATE % 'alpha')
    run.evaluate('candidates/gen0/alpha.py', 'runs/gen0_alpha', report_at=-3600)
    run.read_report('runs/gen0_alpha')
    run.write('proposals/gen1.md', GOOD_PROPOSAL)
    run.write('candidates/gen1/beta.py', CANDIDATE % 'beta')
    return run.finish(calls=7)


def case_still_running():
    run = Run('still_running'); gen0(run)
    return run.finish(calls=5, phase=None)


EXPECT = {
    'normal': ('PASS', []),
    'gen0_only_edit': ('NOT PASSED', ['gen1_candidate_written_after_report_read']),
    'field_names_only': ('NOT PASSED', ['report_field_value_pairs_cited']),
    'budget_exhausted': ('NOT REACHED', ['gen1_candidate_written_after_report_read', 'report_field_value_pairs_cited']),
    'foreign_report': ('NOT PASSED', ['agent_evaluation_exit0_report_in_call_window']),
    'still_running': ('PENDING', ['gen1_candidate_written_after_report_read', 'report_field_value_pairs_cited']),
}


def main():
    FIXTURES.mkdir(exist_ok=True)
    rows, failed = [], 0
    for name, (verdict, unmet) in EXPECT.items():
        root = globals()['case_' + name]()
        summary = v2.analyse(root)
        (FIXTURES / f'{name}_evidence.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        got_unmet = [k for k, met in summary['criteria'].items() if not met]
        checks = [summary['verdict'] == verdict, got_unmet == unmet]
        if name == 'normal':
            cite = summary['report_citation']
            checks += [len(cite['strong_pairs']) >= 2,
                       all(p['field'] != 'status' for p in cite['strong_pairs']),
                       any(p['field'] == 'performance_comparable' for p in cite['weak_pairs_not_counted']),
                       all(e['report_mtime_in_window'] and e['line'] and e['result_line'] for e in summary['evaluate_calls']),
                       all(e['events_result_truncated'] and e['exit_0'] and e['exit_text_source'] == 'gateway request'
                           and e['report_candidate_sha_matches_disk'] for e in summary['evaluate_calls']),
                       summary['informational']['gen1_candidate_evaluated']]
        if name == 'gen0_only_edit':
            checks += [summary['informational']['error_fix_applied_to_gen0'], not summary['gen1_candidate_writes'],
                       len(summary['report_citation']['strong_pairs']) >= 2]
        if name == 'field_names_only':
            checks += [not summary['report_citation']['strong_pairs'],
                       'failures' in summary['report_citation']['field_names_without_value']]
        if name == 'budget_exhausted':
            checks += [summary['run_state']['budget_exhausted'], summary['run_state']['cap_refusals_logged'] == 1]
        if name == 'foreign_report':
            checks += [summary['evaluate_calls'][0]['report_mtime_in_window'] is False,
                       summary['evaluate_calls'][0]['precheck_mtime_in_window'] is True,
                       'runs/gen0_alpha/report.json' in summary['informational']['reports_on_disk_not_attributed_to_any_agent_call']]
        ok = all(checks)
        failed += not ok
        rows.append(dict(case=name, expected=verdict, got=summary['verdict'], unmet=got_unmet,
                         strong_pairs=len(summary['report_citation']['strong_pairs']), ok=ok, checks=checks))
        print(f"{'OK  ' if ok else 'FAIL'} {name:18s} expected={verdict:11s} got={summary['verdict']:11s} unmet={got_unmet}")
    (HERE / 'test_verify_v2_result.json').write_text(json.dumps(rows, indent=2))
    print(f'{len(rows) - failed}/{len(rows)} cases passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
