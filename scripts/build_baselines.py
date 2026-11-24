"""Build auditable v4 baseline summaries from frozen completed full-training evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.runner import load_protocol, PROTOCOL
from validation.util import digest, finite, write_json


def build(snapshot, pool, verify=True):
    protocol=load_protocol(); entries=json.loads(Path(snapshot).read_text())
    norms={r['과제(ID · 짧은 이름)'].split(' · ')[0]:r for r in csv.DictReader(Path(pool).open())}
    doc=dict(schema_version=1,protocol_sha256=digest(PROTOCOL),required_seeds=[0,1,2],
             normalization=dict(formula='(raw_score - baseline) / (ideal - baseline)',
               ideal='1 for maximize; 0 for minimize',clip=False,
               source='results/final_pool_table_20261102_v4.csv; results/seed3_confirm_20261103/confirmation.md'),
             evidence=dict(snapshot_sha256=digest(snapshot),pool_sha256=digest(pool),
                           compatibility='Historical frozen full runs; not rerun with the new plugin API.'), tasks={})
    for task,spec in protocol['tasks'].items():
        r=norms[task]; taskdoc=dict(name=spec['name'],tier=spec['tier'],metric=spec['metric'],direction=spec['direction'],
            baseline=float(r['기준선']),ideal=1 if spec['direction']=='maximize' else 0,models={})
        for model in protocol['models']:
            samples=[]
            for seed in range(3):
                e=entries.get(f'{task}/{model}/{seed}')
                if not e:continue
                if (e['task'],e['model'],e['seed'])!=(task,model,seed):raise ValueError('Snapshot identity mismatch')
                if verify and digest(e['result'])!=e['sha256']:raise ValueError('Result hash mismatch')
                raw=json.loads(Path(e['result']).read_text())
                wall=e.get('wall_s',raw.get('wall_s',raw.get('wall_seconds',raw.get('t_wall_s'))))
                samples.append(dict(seed=seed,score=finite(float(e['score']),'score'),result_sha256=e['sha256'],wall_seconds=wall))
            scores=[e['score'] for e in samples]
            taskdoc['models'][model]=dict(samples=samples,n_seeds=len(scores),mean=statistics.mean(scores) if scores else None,
                sample_std=statistics.stdev(scores) if len(scores)>1 else None,incomplete=len(scores)<3)
        taskdoc['incomplete']=any(v['incomplete'] for v in taskdoc['models'].values())
        doc['tasks'][task]=taskdoc
    return doc


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--pool',type=Path,default=Path(__file__).resolve().parents[1]/'results/final_pool_table_20261102_v4.csv')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'validation/baselines/final16_v4.json')
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    doc=build(a.snapshot,a.pool);write_json(a.out,doc)
    print(json.dumps(dict(tasks=len(doc['tasks']),cells=sum(m['n_seeds'] for t in doc['tasks'].values() for m in t['models'].values()))))
if __name__=='__main__':main()
