# Candidate-generation contract

You are proposing a causal sequence mixer for the frozen v4 final16 harness.
Write at most 3 single-file Python candidates. Define `class CandidateMixer(nn.Module)` with
`__init__(self, d_model, num_heads, l_max)` and `forward(x)` mapping float32 `[B,T,D]` to `[B,T,D]`.
Support CPU autograd and causal inference; no suffix-to-prefix leakage, non-finite tensors, downloads,
API calls, global RNG reseeding, filesystem writes, or training inside the candidate. Use only torch,
standard library and frozen proxy_tasks.common_set helpers. Place helpers in the same file, not sibling imports.

The **frozen repository parameter rule** is ±10% of the Transformer Block (mixer + identical RMSNorm/SwiGLU).
This is not a raw mixer-only 10% rule: precheck reports both counts. Preserve all task widths, depths,
heads, readouts, datasets, splits, seeds and full budgets. Do not edit mixers.py or protocol.json.

After placing a candidate at `$CANDIDATE`, the shell tool can run exactly these three commands from `$HARNESS`:

```bash
$ML_PYTHON -m validation precheck --mixer "$CANDIDATE:CandidateMixer" --data-root "$DATA_ROOT" --out "$OUT/precheck.json"
$ML_PYTHON -m validation suite --mixer "$CANDIDATE:CandidateMixer" --profile smoke --seeds 0 --out "$OUT" --data-root "$DATA_ROOT" --tfl-root "$TFL_ROOT" --device cpu --jobs 1 --timeout 240
$ML_PYTHON -m validation report --suite "$OUT"
```

Use the installed `evaluate.sh` adapter if absolute paths are unavailable in the EvoScientist virtual workspace.
Execution must remain in the explicitly supplied work/output directories; never alter the lab archive.
No push, sudo, host access, paid API, other users' processes, or old overnight supervisor changes.

Budget: maximum 3 new candidates per generation, <=30 local LLM calls for the ENTIRE loop (including retries/subagents),
one seed and one optimizer update per task in smoke; no full training on CPU. Stop starting training at
2026-11-06 07:40 KST and stop all runs at 08:00 KST for this overnight job. Runtime gateway enforces model-call cap;
a textual promise is insufficient. Do not bypass the gateway or switch providers.

Interpret report.json:
- `performance_comparable=false` for smoke: all scores/ranks/tier averages are null; losses are not performance.
- `tasks[].mean`, `sample_std`, `normalized_score`, `rank` apply only to completed measured full cells.
- Normalization `(raw-baseline)/(ideal-baseline)` is un-clipped; ideal=1 for maximize, 0 for minimize.
- Rank is competition rank (ties share rank), comparing candidate with nine historical reference means.
- `baseline_incomplete` and `ranking_provisional` mean the training seeds are not all complete.
- `failures`, `timeouts`, `not_run` are failures to diagnose, not low numerical scores.
- Tier/overall averages with partial coverage are not the final16 aggregate.

Use feedback.md from the previous generation to repair validity before seeking improvements.
For the controlled generation API, write `generated_candidates.json` with
`{"candidates":[{"name":"short_name","code":"complete single-file Python source"}]}`.
State your mechanism, interpretation of the source idea and limits. Do not invent full-training outcomes.
