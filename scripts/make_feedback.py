"""Convert candidate prechecks/reports into honest next-generation input."""
import argparse
import json
from pathlib import Path


def make_feedback(reports, output):
    lines=['# Evaluation feedback for the next generation','',
           'Use full measured results for performance decisions. Smoke shows execution validity only; do not infer strong/weak tasks or ranks from loss.','']
    for path in reports:
        path=Path(path); report=json.loads(path.read_text());check_path=path.parent/'precheck.json'
        check=json.loads(check_path.read_text()) if check_path.exists() else {}
        candidate=(report.get('candidate') or {}).get('model_id','unknown')
        lines += [f'## {candidate}',f'- Precheck: {check.get("status","unknown")}; profile: {report["profile"]}; suite: {report["status"]}.']
        failed=[]
        for task,row in check.get('tasks',{}).items():
            if not row['passed']:
                failed.append(task+': '+row.get('error',', '.join(k for k,v in row.get('checks',{}).items() if not v)))
        lines += ['- Contract failures: '+('; '.join(failed) or check.get('error','none'))]
        if not report['performance_comparable']:
            lines += ['- Rank: unavailable. Strong tasks: unavailable. Weak tasks: unavailable (smoke; 성능 비교 불가).']
        else:
            scored=sorted([t for t in report['tasks'] if t['normalized_score'] is not None],key=lambda t:t['normalized_score'],reverse=True)
            lines += [f'- Mean rank: {report["summary"]["mean_rank"]}; provisional: {report["summary"]["provisional"]}.',
                      '- Strong tasks: '+', '.join(f'{t["task"]}={t["normalized_score"]:.4g}' for t in scored[:3]),
                      '- Weak tasks: '+', '.join(f'{t["task"]}={t["normalized_score"]:.4g}' for t in scored[-3:])]
        errors=report.get('failures',[])+report.get('timeouts',[])+report.get('not_run',[])
        lines += ['- Execution failures/timeouts/not run: '+(json.dumps(errors,ensure_ascii=False) if errors else 'none'),
                  f'- Source report: {path}', '']
    lines += ['## Next request','Propose at most 3 single-file causal mixers. Fix contract failures first. Preserve dimensions, frozen task budgets and the 10% Block parameter rule. Supply implementation rationale; never claim smoke performance improvement.']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines)+'\n')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();make_feedback(a.reports,a.out)
if __name__=='__main__':main()
