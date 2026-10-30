"""Run one selected task with its frozen training configuration."""
import argparse
import json
import math
from pathlib import Path

import torch

from ..common.gpu_guard import require_cuda
from .catalog import build_part
from .training import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--profile', choices=['smoke', 'full'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    part = build_part(args.task)
    device = require_cuda(args.task)
    part.load(device)
    smoke = args.profile == 'smoke'
    result, model = fit(part, args.model.removeprefix('cs_'), args.seed, device,
                        min(3, part.steps) if smoke else part.steps, sanity=smoke)
    result.update(task=args.task, model=args.model, seed=args.seed, profile=args.profile,
                  score_name=part.score_name, direction=part.direction, data=part.describe())
    if result['status'] == 'done' and not smoke:
        model.eval()
        with torch.no_grad():
            result['eval'] = part.evaluate(model)
        if not math.isfinite(result['eval']['score']):
            raise ValueError('Non-finite evaluation score')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['status'] == 'done' else 1


if __name__ == '__main__':
    raise SystemExit(main())
