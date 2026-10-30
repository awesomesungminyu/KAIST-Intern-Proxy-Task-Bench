# proxy-task-bench

[한국어](README.md) · English

A four-tier validation harness that screens new Transformer alternatives proposed by EvoScientist using inexpensive experiments.

## Why this exists

Testing every new architecture directly with large-scale training is expensive.
The aim is to screen candidates before large-scale training using initial-state metrics, synthetic unit tests, and small-scale real-data tasks in successive tiers.

## Tiers 0–3

| Tier | Name | What it measures | Cost |
|---|---|---|---|
| 0 | Zero-cost | Initial-state metrics without training | Seconds |
| 1 | Synthetic unit tests | One isolated capability at a time | Minutes |
| 2 | Small-scale real data | Task performance at a reduced scale | Hours |
| 3 | ETRI large-scale autoregressive LM | Large-scale autoregressive LM performance | Large-scale training |

## Tasks by tier

| Tier | Label | Tasks | Purpose |
|---|---|---|---|
| Tier 0 | - | tier0 | Compute metrics without training |
| Tier 1 | Transformer win | MAD Compression, Positional Histogram, String Matching, RegBench | Compression, positional frequencies, string matching, regular-language ICL |
| Tier 1 | Transformer lose | Parity, Modular Arithmetic, Cycle Navigation, Random Automaton | Parity, modular arithmetic, cycle navigation, automaton state tracking |
| Tier 2 | Transformer win | Permuted MNIST, Permuted Kuzushiji-49, Speech Commands (raw waveform)*, Beijing PM2.5* | Permuted image classification, speech classification, air-quality forecasting |
| Tier 2 | Transformer lose | TAPE Stability, enwik8, IMDB (byte-level), Lipophilicity (SMILES)* | Protein stability, byte-level language modeling, sentiment classification, molecular property prediction |
| Tier 3 | - | (ETRI) | Large-scale autoregressive LM |

The final selection contains 16 tasks: 8 in Tier 1 and 8 in Tier 2. Tier 0 and Tier 3 are outside this count.
Win/lose labels are from the **Transformer's perspective**. Asterisks (*) are preserved from the selection list; their meaning has not yet been specified.

## Two protocol versions

The two versions below record earlier task sets; neither is the protocol for the final 16 tasks. Task selection is separate from implementation and protocol updates.

| Protocol | Tasks and models |
|---|---|
| `v1.json` | 7 tasks (regbench, g2p, zinc, parity, tomita5, tomita6, fflm), each with its own reference models |
| v2.json | A common set of 9 mixers (transformer, gqa, linear_attention, gla, mamba, hyena, lstm, s4d, retnet) in the same model skeleton, applied to 8 tasks (regbench, mqar, jelassi_copy, parity, tomita5, tomita6, scifar, fflm) |

For v2, see the `common-arch-set-v2` branch in the original benchmark repository. This snapshot does not contain `v2.json`. **Results from the two protocols cannot be compared with `compare`.**

## Task interfaces

| Task | Input | Metric (↑ higher is better; ↓ lower is better) | Data |
|---|---|---|---|
| T1-R1-05 · MAD Compression | Tokens | To be finalized | Generated synthetic data |
| T1-R2-01 · Positional Histogram | Tokens | To be finalized | Generated synthetic data |
| T1-R2-02 · String Matching | Tokens | To be finalized | Generated synthetic data |
| T1-R0-07 · RegBench | Tokens | `next_token_acc` ↑ | Generated PFA streams |
| T1-R0-04 · Parity | Tokens | `mean_best_val_accuracy` ↑ | Files: TFL corpus |
| T1-R2-03 · Modular Arithmetic | Tokens | To be finalized | Generated synthetic data |
| T1-R2-04 · Cycle Navigation | Tokens | To be finalized | Generated synthetic data |
| T1-R2-05 · Random Automaton | Tokens | To be finalized | Generated synthetic data |
| T2-R1-11 · Permuted MNIST | Image sequences | To be finalized | Files: MNIST |
| T2-R3-05 · Permuted Kuzushiji-49 | Image sequences | To be finalized | Files: Kuzushiji-49 |
| T2-R4-02 · Speech Commands (raw waveform)\* | Continuous values | To be finalized | Files: Speech Commands |
| T2-R2-07 · Beijing PM2.5\* | Continuous values | To be finalized | Files: Beijing PM2.5 |
| T2-R1-04 · TAPE Stability | Amino acid sequences | To be finalized | Files: TAPE Stability |
| T2-R1-09 · enwik8 | Bytes | To be finalized | Files: enwik8 |
| T2-R1-08 · IMDB (byte-level) | Bytes | To be finalized | Files: IMDB |
| T2-R2-02 · Lipophilicity (SMILES)\* | SMILES tokens | To be finalized | Files: Lipophilicity |

Metric keys are listed only where verified in the existing RegBench and Parity implementations. Final metric keys and aggregation rules for the remaining tasks will be specified when the protocol is updated.

## Quick start

The commands below inspect the existing execution interface in this snapshot. The output of `validation list` may differ from the final 16-task selection.

Download experimental data from [proxy-task-bench-data](https://drive.google.com/drive/folders/1FXi_UlvUnvrlfBj7YryFY0dmoE7uoZes?usp=drive_link) on Google Drive.
Verify downloaded archives against the accompanying `SHA256SUMS.txt`.
After extraction, export `PROXY_DATA_ROOT` as the directory containing the task data folders.

From the parent directory of your clone, enter the personal repository as shown below. `check` inspects CUDA and required assets; it does not train a model.

```bash
cd KAIST-Intern-Proxy-Task-Bench
python -m validation list
python -m validation check --task regbench --model transformer
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

## Limitations

- In v1, transformer implementations differed across tasks.
- Results and datasets are not included in this repository.
- The final 16-task selection differs from the implementations and protocols in this snapshot.
- Tier 2 mixes language and non-language tasks; evidence for predicting Tier 3 performance requires task-by-task review.
