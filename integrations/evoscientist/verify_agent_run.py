"""REFERENCE ONLY (v1). The official verdict script is verify_agent_run_v2.py.

Extract evidence of agent-initiated harness use from an agent-led run folder.

Reads only what EvoScientist itself emitted (stream-json events) plus the files in
its workspace. It evaluates nothing and never calls a model. The verdict is PASS only
when every criterion is met by the agent's own tool calls; otherwise NOT PASSED.
"""
import argparse
import json
from pathlib import Path
import re
import sys

WRITE_TOOLS = ('write_file', 'edit_file')
SHELL_TOOLS = ('execute', 'run_in_background')
REPORT_FIELDS = ('performance_comparable', 'failures', 'timeouts', 'not_run', 'tier1_mean', 'tier2_mean',
                 'overall_mean', 'mean_rank', 'normalized_score', 'ranking_provisional', 'baseline_incomplete',
                 'completed_seeds', 'profile', 'status')


def load_events(path):
    events = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event['_line'] = number
        events.append(event)
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    workspace = out / 'workspace'
    events = load_events(out / 'events.jsonl')
    results = {e['id']: e for e in events if e.get('type') == 'tool_result' and e.get('id')}
    calls = [e for e in events if e.get('type') == 'tool_call']

    candidate_writes, evaluations, report_reads, later_texts = [], [], [], []
    for call in calls:
        arguments = call.get('args') or {}
        result = results.get(call.get('id'), {})
        target = str(arguments.get('file_path') or arguments.get('path') or '')
        command = str(arguments.get('command') or '')
        row = dict(line=call['_line'], tool=call['name'], subagent=call.get('subagent') or '')
        if call['name'] in WRITE_TOOLS and re.search(r'candidates/gen\d+/[^/]+\.py$', target):
            candidate_writes.append(dict(row, file=target, success=result.get('success')))
        if call['name'] in SHELL_TOOLS and 'evaluate.sh' in command:
            content = str(result.get('content', ''))
            # stream-json truncates long tool output, so the exit line may be cut off. An evaluation counts
            # as completed when the tool returned without a failure marker and the report it names exists.
            target_run = re.search(r'evaluate\.sh\s+\S+\s+(?:\./)?(runs/[\w.\-]+)', command)
            report = workspace / target_run.group(1) / 'report.json' if target_run else None
            evaluations.append(dict(row, command=command, timeout=arguments.get('timeout'), result_line=result.get('_line'),
                                    succeeded=bool(result.get('success')) and 'Command failed' not in content
                                    and report is not None and report.is_file(),
                                    report=str(report.relative_to(workspace)) if report else None,
                                    result_head=content[:200], result_tail=content[-300:]))
        if (call['name'] == 'read_file' and target.endswith('report.json')) or \
                (call['name'] in SHELL_TOOLS and 'report.json' in command and 'evaluate.sh' not in command):
            content = str(result.get('content', ''))
            report_reads.append(dict(row, target=target or command, result_line=result.get('_line'),
                                     returned_report='performance_comparable' in content,
                                     result_head=content[:300]))

    # Text the agent wrote after its first successful report read: later candidates, proposals and prose.
    first_read = min((r['result_line'] for r in report_reads if r['returned_report'] and r['result_line']), default=None)
    cited = {}
    next_candidates = []
    if first_read is not None:
        prose = ''
        for event in events:
            if event['_line'] <= first_read:
                continue
            if event.get('type') == 'text':
                prose += event.get('content', '')
            elif event.get('type') == 'tool_call' and event['name'] in WRITE_TOOLS:
                arguments = event.get('args') or {}
                body = str(arguments.get('content') or arguments.get('new_string') or '')
                target = str(arguments.get('file_path') or '')
                later_texts.append((target, body))
                if re.search(r'candidates/gen\d+/[^/]+\.py$', target):
                    next_candidates.append(dict(line=event['_line'], file=target))
        corpus = prose + '\n'.join(body for _, body in later_texts)
        cited = {field: corpus.count(field) for field in REPORT_FIELDS if field in corpus}

    reports_on_disk = sorted(str(p.relative_to(workspace)) for p in (workspace / 'runs').rglob('report.json'))
    candidates_on_disk = sorted(str(p.relative_to(workspace)) for p in (workspace / 'candidates').rglob('gen*/*.py'))
    budget = json.loads((out / 'llm_budget.json').read_text()) if (out / 'llm_budget.json').exists() else {}
    criteria = {
        'agent_wrote_candidate_file': any(w['success'] for w in candidate_writes),
        'agent_shell_called_evaluate_sh': bool(evaluations),
        'an_agent_evaluation_exited_0': any(e['succeeded'] for e in evaluations),
        'agent_read_report_json': any(r['returned_report'] for r in report_reads),
        'next_candidate_written_after_report_read': bool(next_candidates),
        'report_fields_cited_after_read': bool(cited),
    }
    verdict = 'PASS' if all(criteria.values()) else 'NOT PASSED'
    summary = dict(verdict=verdict, criteria=criteria, llm_calls=budget.get('calls'),
                   candidate_writes=candidate_writes, evaluate_calls=evaluations, report_reads=report_reads,
                   next_candidates_after_first_report_read=next_candidates, report_fields_cited=cited,
                   candidates_on_disk=candidates_on_disk, reports_on_disk=reports_on_disk,
                   tool_call_counts={name: sum(c['name'] == name for c in calls) for name in sorted({c['name'] for c in calls})},
                   note='Smoke profile has no performance score; this verdict is about the connection only.')
    (out / 'agent_evidence.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(dict(verdict=verdict, criteria=criteria, llm_calls=budget.get('calls'),
                          evaluate_calls=len(evaluations), report_reads=len(report_reads)), indent=2))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
