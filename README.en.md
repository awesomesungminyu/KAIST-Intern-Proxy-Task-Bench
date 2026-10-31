# proxy-task-bench

[한국어](README.md) · English

A benchmark of 16 synthetic and small real-data tasks for evaluating Transformer alternatives. EvoScientist can use it through shell commands.

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
- **Tomita 5**: Regular-language parity tracking
- **Cycle Navigation**: Cycle navigation
- **Random Automaton**: Automaton state tracking

### Tier 2 · Real data · Transformer win

- **Permuted MNIST**: Permuted image classification
- **Permuted Kuzushiji-49**: Permuted image classification
- **Speech Commands (raw waveform)\***: Speech classification
- **Permuted EMNIST (letters)†**: Permuted handwriting classification

### Tier 2 · Real data · Transformer lose

- **TAPE Stability**: Protein stability
- **enwik8**: Byte-level language modeling
- **IMDB (byte-level)**: Sentiment classification
- **SNLI (byte-level)†**: Natural-language inference

### Tier 3

- **(ETRI)**: Large-scale autoregressive LM

The final selection contains **8 Tier 1 tasks + 8 Tier 2 tasks = 16 tasks**. Tier 0 and Tier 3 are outside this count.
Win/loss labels come from comparing the Transformer with the LSTM. The ranking across all nine architectures differs by task.

## Task interfaces

| Task | Input | Metric (↑ higher is better; ↓ lower is better) | Data |
|---|---|---|---|
| [T1-R1-05 · MAD Compression](tasks/T1-R1-05/README.md) | Tokens | `token_accuracy` ↑ | Files: synthetic cache |
| [T1-R2-01 · Positional Histogram](tasks/T1-R2-01/README.md) | Tokens | `slot_accuracy` ↑ | Generated at runtime |
| [T1-R2-02 · String Matching](tasks/T1-R2-02/README.md) | Tokens | `accuracy` ↑ | Generated at runtime |
| [T1-R0-07 · RegBench](tasks/T1-R0-07/README.md) | Tokens | `next_token_acc` ↑ | Generated at runtime |
| [T1-R0-04 · Parity](tasks/T1-R0-04/README.md) | Tokens | `mean_best_val_accuracy` ↑ | Files |
| [T1-R0-05 · Tomita 5](tasks/T1-R0-05/README.md) | Tokens | `mean_best_val_accuracy` ↑ | Files |
| [T1-R2-04 · Cycle Navigation](tasks/T1-R2-04/README.md) | Tokens | `accuracy_id50_ex50` ↑ | Generated at runtime |
| [T1-R2-05 · Random Automaton](tasks/T1-R2-05/README.md) | Tokens | `last_state_accuracy_id50_ex50` ↑ | Generated at runtime |
| [T2-R1-11 · Permuted MNIST](tasks/T2-R1-11/README.md) | Image sequences | `accuracy` ↑ | Files |
| [T2-R3-05 · Permuted Kuzushiji-49](tasks/T2-R3-05/README.md) | Image sequences | `accuracy` ↑ | Files |
| [T2-R4-02 · Speech Commands (raw waveform)\*](tasks/T2-R4-02/README.md) | Continuous values | `accuracy` ↑ | Files |
| [T2-R3-02 · Permuted EMNIST (letters)†](tasks/T2-R3-02/README.md) | Image sequences | `accuracy` ↑ | Files |
| [T2-R1-04 · TAPE Stability](tasks/T2-R1-04/README.md) | Amino acid sequences | `spearman_valid` ↑ | Files |
| [T2-R1-09 · enwik8](tasks/T2-R1-09/README.md) | Bytes | `val_bpc` ↓ | Files |
| [T2-R1-08 · IMDB (byte-level)](tasks/T2-R1-08/README.md) | Bytes | `accuracy` ↑ | Files |
| [T2-R2-01 · SNLI (byte-level)†](tasks/T2-R2-01/README.md) | Bytes | `accuracy` ↑ | Files |

\* S5 confirmed: Speech Commands only (Transformer/LSTM checked over three seeds). † Reserve (diversity excess): EMNIST and SNLI. Tomita 5 replaces Modular Arithmetic using the original round-0 three-seed results. [Final v3 scores](results/final_pool_table_20261031_v3.md).

## Quick start

Use Linux, Python 3.11, and an NVIDIA CUDA GPU. Listing tasks and running CPU contract tests do not require a GPU.

```bash
git clone https://github.com/awesomesungminyu/KAIST-Intern-Proxy-Task-Bench.git
cd KAIST-Intern-Proxy-Task-Bench
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m validation list
```

The example installs the CUDA 12.4 build. Consult the [official installation table](https://pytorch.org/get-started/previous-versions/#v251) for other driver setups, keeping PyTorch 2.5.1.

**Data: [drive_upload_20261030 (Google Drive)](https://drive.google.com/drive/folders/1KAhMI1h7CzB8pR4h6JvhyoG4rD_1dH7l)**

Of the 16 tasks, 5 generate data at run time and 11 need supplied files. The four existing Drive bundles cover 8 of those 11 tasks; Tomita 5, EMNIST and SNLI need the six additional files listed in [data preparation](docs/DATA.md). No new archive was created.

```bash
# Prepare files from the directory containing your extracted downloads.
python -m validation prepare --source downloads/extracted --data-root data
# Check data availability without CUDA.
python -m validation data --data-root data
# Start with a generated task that needs no downloaded data.
python -m validation check --task T1-R2-02 --model cs_transformer
python -m validation run --task T1-R2-02 --model cs_transformer --profile smoke --seeds 0
```

Defaults are `data/<task ID>/` for inputs and `artifacts/<task ID>/<model>/<run ID>/` for outputs. Use `--data-root` or the optional settings in `.env.example` to relocate storage. `.env` is not loaded automatically.
Prepare the shared Parity/Tomita 5 external runner with `bash setup.sh formal`, and RetNet's CUDA kernel with `bash setup.sh retnet`.

```bash
# Full training budget; choose a new output directory.
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile full --seeds 0 1 2 --out artifacts/string-matching-full
# Compare completed runs with the same task, data, settings, and seeds.
python -m validation compare --baseline artifacts/baseline --candidate artifacts/candidate
# CPU contract tests
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
# Import and train one Transformer step on each of the 16 tasks, after preparing data.
python scripts/smoke_tasks.py --data-root data --device cuda
```

`check` inspects files and the runtime. Smoke and one-step checks are execution checks, not performance evidence. Only completed `full` runs with result status `done` can be compared.
See [models](docs/MODELS.md) for architecture changes and matched comparisons, and the [EvoScientist guide](docs/EVOSCIENTIST.md) for agent instructions.

## Known issues

- The existing bundles omit the six files for Tomita 5, EMNIST and SNLI. See [the upload list](docs/DRIVE_ADDITIONS_20261031.csv); these three tasks need those files before preparation.

## Limitations

- Selection, imports, and one-step training checks do not establish full-training performance or validate every model/task combination.
- Results, datasets, and checkpoints are excluded from Git.
- RegBench evaluates a repeatedly observed development stream. Parity and Tomita 5 average the best accuracy per bin and does not represent one selected checkpoint.
- Tiers 0 and 3 describe the wider validation framework. This repository executes only the final 16 Tier 1 and Tier 2 tasks.
