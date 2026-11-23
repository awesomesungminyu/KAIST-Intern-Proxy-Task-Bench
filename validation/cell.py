"""One real-data optimizer update for suite smoke; never evaluates performance."""
import argparse
import json
from pathlib import Path
import time
import torch
from .runner import load_protocol
from .smoke import one_step
from .util import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True); p.add_argument('--model',required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--tfl-root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    a=p.parse_args(); torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    loss=one_step(a.task,load_protocol(a.task)[1],a.data_root,a.tfl_root,a.device,a.model,a.seed)
    write_json(a.out/'result.json',dict(status='done',task=a.task,model=a.model,seed=a.seed,
        profile='smoke',steps=1,loss=loss,performance_comparable=False,wall_seconds=time.monotonic()-start))
    return 0

if __name__=='__main__':raise SystemExit(main())
