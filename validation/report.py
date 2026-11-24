"""Direction-aware descriptive reports; smoke cannot acquire a performance score."""
import json
import math
from pathlib import Path
import statistics

from .util import digest, finite, write_json


def normalized(score, baseline, ideal):
    score=finite(score,'score');baseline=finite(baseline,'baseline');ideal=finite(ideal,'ideal')
    if ideal==baseline:raise ValueError('Normalization ideal equals baseline')
    return (score-baseline)/(ideal-baseline)


def rank(score, others, direction):
    """Competition ranks: equal scores share rank; lower is better for minimize."""
    return 1+sum((v>score if direction=='maximize' else v<score) and not math.isclose(v,score,abs_tol=1e-10,rel_tol=1e-9) for v in others)


def make_report(folder, baseline_path):
    folder=Path(folder);suite=json.loads((folder/'suite.json').read_text())
    baseline=json.loads(Path(baseline_path).read_text())
    smoke=suite['profile']=='smoke'
    doc=dict(schema_version=1,status=suite['status'],candidate=suite.get('candidate'),profile=suite['profile'],
        performance_comparable=not smoke and suite.get('precheck')=='passed',
        interpretation='성능 비교 불가 (smoke: execution evidence only)' if smoke else 'Descriptive full-run comparison; provisional where seeds are incomplete',
        normalization=baseline['normalization'],baseline_sha256=digest(baseline_path),
        rank_definition='Competition rank among 9 baseline means plus candidate; ties share rank',
        tasks=[],summary={},failures=[],timeouts=[],not_run=[])
    config=suite.get('configuration',{})
    if config and config.get('protocol_sha256')!=baseline['protocol_sha256']:
        raise ValueError('Baseline and suite protocol hashes differ')
    for task in suite['tasks']:
        reference=baseline['tasks'][task];rows=[r for r in suite['runs'] if r['task']==task]
        good=[r for r in rows if r['status']=='done']
        item=dict(task=task,name=reference['name'],tier=reference['tier'],status='done' if len(good)==len(suite['seeds']) else 'incomplete',
            completed_seeds=sorted(r['seed'] for r in good),requested_seeds=suite['seeds'],mean=None,sample_std=None,
            normalized_score=None,rank=None,versus_transformer=None,versus_lstm=None,
            baseline_incomplete=reference['incomplete'],ranking_provisional=True,
            wall_seconds=sum(r.get('wall_seconds',0) for r in rows))
        for row in rows:
            if row['status']=='timeout':doc['timeouts'].append(dict(task=task,seed=row['seed'],error=row.get('error')))
            elif row['status']=='not_run':doc['not_run'].append(dict(task=task,seed=row['seed'],reason=row.get('reason')))
            elif row['status']!='done':doc['failures'].append(dict(task=task,seed=row['seed'],error=row.get('error',suite.get('reason'))))
        represented={r['seed'] for r in rows}
        for seed in suite['seeds']:
            if seed not in represented:doc['not_run'].append(dict(task=task,seed=seed,reason=suite.get('reason','missing result')))
        if good and not smoke:
            if len({r['seed'] for r in good})!=len(good):raise ValueError('Duplicate candidate seeds')
            if any(r.get('evidence')!='measured' or r.get('profile')!='full' for r in good):
                raise ValueError('Only measured full runs may receive scores')
            if any(r.get('protocol_sha256',config.get('protocol_sha256'))!=baseline['protocol_sha256'] for r in good):
                raise ValueError('Result protocol mismatch')
            values=[finite(r['metrics'][reference['metric']],reference['metric']) for r in good]
            mean=statistics.mean(values);models=reference['models']
            means=[m['mean'] for m in models.values() if m['mean'] is not None]
            item.update(mean=mean,sample_std=statistics.stdev(values) if len(values)>1 else None,
                normalized_score=normalized(mean,reference['baseline'],reference['ideal']),
                rank=rank(mean,means,reference['direction']) if len(means)==9 else None,
                ranking_provisional=reference['incomplete'] or len(values)<3 or len(means)!=9)
            for label,model in [('transformer','cs_transformer'),('lstm','cs_lstm')]:
                base=models[model]['mean']
                if base is None:continue
                delta=normalized(mean,reference['baseline'],reference['ideal'])-normalized(base,reference['baseline'],reference['ideal'])
                item['versus_'+label]=dict(baseline_mean=base,normalized_delta=delta,
                    position='tied' if math.isclose(delta,0,abs_tol=1e-9) else 'above' if delta>0 else 'below',
                    reference_rank=rank(base,means,reference['direction']))
        doc['tasks'].append(item)
    scored=[r for r in doc['tasks'] if r['normalized_score'] is not None]
    doc['summary']=dict(requested_tasks=len(suite['tasks']),scored_tasks=len(scored),
        complete_tasks=sum(t['status']=='done' for t in doc['tasks']),
        tier1_mean=statistics.mean([r['normalized_score'] for r in scored if r['tier']==1]) if any(r['tier']==1 for r in scored) else None,
        tier2_mean=statistics.mean([r['normalized_score'] for r in scored if r['tier']==2]) if any(r['tier']==2 for r in scored) else None,
        overall_mean=statistics.mean([r['normalized_score'] for r in scored]) if scored else None,
        mean_rank=statistics.mean([r['rank'] for r in scored if r['rank'] is not None]) if any(r['rank'] is not None for r in scored) else None,
        provisional=smoke or len(scored)<16 or any(r['ranking_provisional'] for r in scored),
        aggregation='Equal task weights of un-clipped normalized scores; partial coverage is not a full16 score')
    write_json(folder/'report.json',doc)
    md=f"# Candidate report — {suite.get('candidate',{}).get('model_id','unknown')}\n\n{doc['interpretation']}\n\n"
    md+='| Task | Status | n seeds | Mean ± sample SD | Normalized | Rank / 10 | Baseline |\n|---|---|---:|---|---:|---:|---|\n'
    for r in doc['tasks']:
        sd='—' if r['sample_std'] is None else f"{r['sample_std']:.6g}"
        mean='—' if r['mean'] is None else f"{r['mean']:.6g} ± {sd}"
        norm='—' if r['normalized_score'] is None else f"{r['normalized_score']:.6g}"
        md+=f"| {r['task']} {r['name']} | {r['status']} | {len(r['completed_seeds'])} | {mean} | {norm} | {r['rank'] or '—'} | {'incomplete' if r['baseline_incomplete'] else '3-seed complete'} |\n"
    md+='\nSummary:\n\n```json\n'+json.dumps(doc['summary'],ensure_ascii=False,indent=2)+'\n```\n'
    md+='\nNormalization: '+baseline['normalization']['formula']+'; '+baseline['normalization']['ideal']+'. Values are not clipped.\n'
    md+='Source: '+baseline['normalization']['source']+'.\n'
    md+=f"\nFailures: {len(doc['failures'])}; timeouts: {len(doc['timeouts'])}; not run: {len(doc['not_run'])}. See report.json for reasons.\n"
    (folder/'report.md').write_text(md)
    return doc
