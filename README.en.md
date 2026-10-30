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

### Tier 0

- **tier0**: Compute metrics without training

### Tier 1 · Synthetic tasks · Transformer win

- **MAD Compression**: Compression
- **Positional Histogram**: Positional frequencies
- **String Matching**: String matching
- **RegBench**: Regular-language ICL

### Tier 1 · Synthetic tasks · Transformer lose

- **Parity**: Parity
- **Modular Arithmetic**: Modular arithmetic
- **Cycle Navigation**: Cycle navigation
- **Random Automaton**: Automaton state tracking

### Tier 2 · Real data · Transformer win

- **Permuted MNIST**: Permuted image classification
- **Permuted Kuzushiji-49**: Permuted image classification
- **Speech Commands (raw waveform)\***: Speech classification
- **Beijing PM2.5\***: Air-quality forecasting

### Tier 2 · Real data · Transformer lose

- **TAPE Stability**: Protein stability
- **enwik8**: Byte-level language modeling
- **IMDB (byte-level)**: Sentiment classification
- **Lipophilicity (SMILES)\***: Molecular property prediction

### Tier 3

- **(ETRI)**: Large-scale autoregressive LM

The final selection contains **8 Tier 1 tasks + 8 Tier 2 tasks = 16 tasks**. Tier 0 and Tier 3 are outside this count.
Win/lose labels are from the **Transformer's perspective**.

## Task interfaces

| Task | Input | Metric (↑ higher is better; ↓ lower is better) | Data |
|---|---|---|---|
| [T1-R1-05 · MAD Compression](tasks/T1-R1-05/README.md) | Tokens | `token_accuracy` ↑ | Files: synthetic cache |
| [T1-R2-01 · Positional Histogram](tasks/T1-R2-01/README.md) | Tokens | `slot_accuracy` ↑ | Generated at runtime |
| [T1-R2-02 · String Matching](tasks/T1-R2-02/README.md) | Tokens | `accuracy` ↑ | Generated at runtime |
| [T1-R0-07 · RegBench](tasks/T1-R0-07/README.md) | Tokens | `next_token_acc` ↑ | Generated at runtime |
| [T1-R0-04 · Parity](tasks/T1-R0-04/README.md) | Tokens | `mean_best_val_accuracy` ↑ | Files |
| [T1-R2-03 · Modular Arithmetic](tasks/T1-R2-03/README.md) | Tokens | `accuracy_id50_ex50` ↑ | Generated at runtime |
| [T1-R2-04 · Cycle Navigation](tasks/T1-R2-04/README.md) | Tokens | `accuracy_id50_ex50` ↑ | Generated at runtime |
| [T1-R2-05 · Random Automaton](tasks/T1-R2-05/README.md) | Tokens | `last_state_accuracy_id50_ex50` ↑ | Generated at runtime |
| [T2-R1-11 · Permuted MNIST](tasks/T2-R1-11/README.md) | Image sequences | `accuracy` ↑ | Files |
| [T2-R3-05 · Permuted Kuzushiji-49](tasks/T2-R3-05/README.md) | Image sequences | `accuracy` ↑ | Files |
| [T2-R4-02 · Speech Commands (raw waveform)\*](tasks/T2-R4-02/README.md) | Continuous values | `accuracy` ↑ | Files |
| [T2-R2-07 · Beijing PM2.5\*](tasks/T2-R2-07/README.md) | Continuous values | `mse_ctx_id50_long50` ↓ | Files |
| [T2-R1-04 · TAPE Stability](tasks/T2-R1-04/README.md) | Amino acid sequences | `spearman_valid` ↑ | Files |
| [T2-R1-09 · enwik8](tasks/T2-R1-09/README.md) | Bytes | `val_bpc` ↓ | Files |
| [T2-R1-08 · IMDB (byte-level)](tasks/T2-R1-08/README.md) | Bytes | `accuracy` ↑ | Files |
| [T2-R2-02 · Lipophilicity (SMILES)\*](tasks/T2-R2-02/README.md) | SMILES tokens | `rmse_valid` ↓ | Files |

## Quick start

**Data location: [proxy-task-bench-data (Google Drive)](https://drive.google.com/drive/folders/1FXi_UlvUnvrlfBj7YryFY0dmoE7uoZes?usp=drive_link)**

Tasks that generate their data at runtime use their built-in generators. For all other tasks, download and use the data and caches provided in this Drive folder. EvoScientist should consult this folder first and use the supplied data rather than regenerate it or substitute another dataset.
Verify archives with the accompanying `SHA256SUMS.txt`.
See [data setup](docs/DATA.md) for the directory layout and required files.

```bash
cd KAIST-Intern-Proxy-Task-Bench
export PROXY_DATA_ROOT=/absolute/path/to/data
python -m validation list
python -m validation check --task T1-R0-07 --model cs_transformer
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

Install a CUDA build of PyTorch 2.5.1 and the dependencies in `requirements.txt`. Prepare Parity's external runner with `bash setup.sh parity`, and RetNet's CUDA kernel with `bash setup.sh retnet`.
`check` inspects the environment and required files without training. `list` contains exactly the final 16 tasks.

```bash
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/string-matching-smoke
# Full training budget
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile full --seeds 0 1 2 --out artifacts/string-matching-full
```

See [models](docs/MODELS.md) for the nine shared models and task-specific settings. Execution settings are in `validation/protocol.json`.

## Limitations

- Final task selection does not imply completed repeated-seed validation of every model.
- Results, datasets, and checkpoints are not included.
- `smoke` checks execution; it is not performance evidence.
- RegBench evaluates a repeatedly observed development stream. Parity averages the best accuracy per bin and does not represent one selected checkpoint.
- Tier 2 mixes language and non-language tasks; predicting Tier 3 performance requires task-by-task review.
