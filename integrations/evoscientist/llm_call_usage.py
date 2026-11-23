"""Classify every counted LLM call of a run by the tool calls that ran right before it.

Read-only on the run folder. For gateway call N the "preceding tools" are the tool
results that request N carries and request N-1 did not (i.e. the tools issued by the
previous response). Output: a Markdown table and a CSV, written next to this script.

Categories: 경로 탐색 / skill·문서 읽기 / 후보 작성 / 평가 / report 읽기 / 설계 글쓰기 / 오류 재시도.
A separate flag marks path confusion (file-tool root '/' vs shell cwd, /workspace
absolute paths being rewritten by the shell sandbox).
"""
import argparse
import csv
import json
from pathlib import Path
import re

CONFUSION_RESULT = re.compile(r"cannot access '\.?/?workspace|can't cd to|Command blocked: contains '\.\.'|workspace/workspace")
PROBE = re.compile(r'^\s*(cd|ls|pwd|echo|find|realpath|stat|test|\[)\b')


def parse_response(path):
    text, calls, errors = '', {}, False
    if not path.exists() or path.stat().st_size == 0:
        return text, [], True
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith('data: ') or line.strip() == 'data: [DONE]':
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            text += delta.get('content') or ''
            for call in delta.get('tool_calls') or []:
                entry = calls.setdefault(call.get('index', 0), dict(name='', args=''))
                function = call.get('function', {})
                entry['name'] += function.get('name') or ''
                entry['args'] += function.get('arguments') or ''
    return text, [calls[k] for k in sorted(calls)], errors


def arguments(raw):
    try:
        return json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {}


def classify(name, args, result):
    """-> (category, path_confusion: bool)"""
    target = str(args.get('file_path') or args.get('path') or '')
    command = str(args.get('command') or '')
    result = str(result)
    if name in ('execute', 'run_in_background'):
        if 'evaluate.sh' in command:
            return '평가', False
        if 'report.json' in command or 'precheck.json' in command:
            return 'report 읽기', False
        confused = bool(CONFUSION_RESULT.search(result)) or '/workspace' in command or '..' in command \
            or bool(re.search(r'(^|[;&]\s*)ls\s+/\s', command + ' '))
        return '경로 탐색', confused
    if name in ('ls', 'glob', 'grep'):
        return '경로 탐색', False
    if name == 'read_file':
        if re.search(r'(report|precheck|suite)\.(json|md)$', target):
            return 'report 읽기', False
        if Path(target).is_absolute():
            return '경로 탐색', True  # file tool is rooted at the workspace; real absolute path cannot resolve
        return 'skill·문서 읽기', False
    if name in ('write_file', 'edit_file'):
        if re.search(r'candidates/gen\d+/[^/]+\.py$', target):
            return '후보 작성', False
        return '설계 글쓰기', False
    if name in ('write_todos', 'think_tool'):
        return '설계 글쓰기', False
    return '기타(' + name + ')', False


def brief(name, args):
    if name in ('execute', 'run_in_background'):
        return name + ': ' + str(args.get('command', ''))[:70].replace('\n', ' ').replace('|', '¦')
    target = args.get('file_path') or args.get('path') or ''
    return f'{name} {target}'.strip()


def analyse(gateway):
    rows, seen, names = [], set(), {}
    previous_body = None
    for request in sorted(gateway.glob('call_*_request.json')):
        number = int(request.name.split('_')[1])
        raw = request.read_bytes()
        body = json.loads(raw)
        messages = body.get('messages', [])
        for message in messages:
            for call in message.get('tool_calls') or []:
                names[call['id']] = (call['function']['name'], arguments(call['function'].get('arguments')))
        fresh = [m for m in messages if m.get('role') == 'tool' and m.get('tool_call_id') not in seen]
        seen |= {m['tool_call_id'] for m in fresh}
        text, issued, empty = parse_response(gateway / f'call_{number:03d}_response.txt')
        preceding = []
        for message in fresh:
            name, args = names.get(message['tool_call_id'], ('?', {}))
            category, confused = classify(name, args, message.get('content', ''))
            preceding.append(dict(name=name, brief=brief(name, args), category=category, confused=confused))
        retry = previous_body == raw or (empty and not issued and not text)
        if retry:
            categories, flag = ['오류 재시도'], '-'
        elif not preceding:
            categories, flag = ['(시작: 직전 도구 없음)'], '-'
        else:
            categories = list(dict.fromkeys(p['category'] for p in preceding))
            hits = sum(p['confused'] for p in preceding)
            flag = '전부' if hits == len(preceding) else '일부' if hits else '-'
        issued_rows = []
        for call in issued:
            args = arguments(call['args'])
            issued_rows.append(brief(call['name'], args))
        rows.append(dict(call=number, bytes=len(raw), preceding_tools='; '.join(p['brief'] for p in preceding) or '-',
                         category=' + '.join(categories), path_confusion=flag, response_text_chars=len(text),
                         response_empty=empty, issued='; '.join(issued_rows) or ('(텍스트만)' if text else '(응답 없음)')))
        previous_body = raw
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.out.resolve() == run or run in args.out.resolve().parents:
        parser.error('--out must not be inside the run folder')
    sections = [('attempt1_model_id_400', run / 'attempt1_model_id_400/gateway'), ('attempt2 (current gateway/)', run / 'gateway')]
    lines, table = ['# LLM 호출 사용처 (자동 생성: llm_call_usage.py)', ''], []
    for label, gateway in sections:
        if not gateway.is_dir():
            continue
        rows = analyse(gateway)
        errors = (gateway / 'errors.log').read_text(errors='replace').splitlines() if (gateway / 'errors.log').exists() else []
        lines += [f'## {label}: {len(rows)} calls, errors.log {len(errors)}줄', '',
                  '| 호출 | 직전에 실행된 도구 | 분류 | 경로 혼동 | 이 호출이 낸 도구 |', '|---:|---|---|:---:|---|']
        for row in rows:
            lines.append(f"| {row['call']} | {row['preceding_tools']} | {row['category']} | {row['path_confusion']} | {row['issued']} |")
            table.append(dict(attempt=label, **row))
        counts = {}
        for row in rows:
            for category in row['category'].split(' + '):
                counts[category] = counts.get(category, 0) + 1
        full = [r['call'] for r in rows if r['path_confusion'] == '전부']
        part = [r['call'] for r in rows if r['path_confusion'] == '일부']
        lines += ['', '분류별 호출 수(한 호출이 여러 분류에 걸치면 각각 1회): ' + ', '.join(f'{k} {v}' for k, v in counts.items()),
                  f'경로 혼동: 전부 {len(full)}회 {full}, 일부 {len(part)}회 {part}', '']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'llm_call_usage.md').write_text('\n'.join(lines) + '\n')
    with (args.out / 'llm_call_usage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else ['attempt'])
        writer.writeheader(); writer.writerows(table)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
