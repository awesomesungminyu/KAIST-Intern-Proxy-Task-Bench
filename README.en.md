# Proxy Task Bench

For EvoScientist integration, use [kaist_evo_proxy](https://github.com/awesomesungminyu/kaist_evo_proxy).

[한국어](README.md) · English

Evaluate mixers on eight synthetic Tier 1 tasks and ten dataset-based Tier 2 tasks. Lipophilicity replaces SNLI to cover chemistry. Genomics and Bach are provided as a separate extension.

### Tier 1

| ID | Task | Input | Metric | Data |
|---|---|---|---|---|
| T1-R1-05 | MAD Compression | tokens | token_accuracy ↑ | file |
| T1-R0-07 | RegBench | tokens | next_token_acc ↑ | generated |
| T1-R2-02 | String Matching | tokens | accuracy ↑ | generated |
| T1-R2-01 | Positional Histogram | tokens | slot_accuracy ↑ | generated |
| T1-R2-05 | Random Automaton | tokens | last_state_accuracy_id50_ex50 ↑ | generated |
| T1-R2-04 | Cycle Navigation | tokens | accuracy_id50_ex50 ↑ | generated |
| T1-R0-04 | Parity | tokens | mean_best_val_accuracy ↑ | file |
| T1-R0-05 | Tomita 5 | tokens | mean_best_val_accuracy ↑ | file |

### Tier 2

| ID | Task | Input | Metric | Data |
|---|---|---|---|---|
| T2-R1-11 | Permuted MNIST | continuous sequence | accuracy ↑ | file |
| T2-R3-05 | Permuted Kuzushiji-49 | continuous sequence | accuracy ↑ | file |
| T2-R4-02 | Speech Commands (raw waveform) | continuous sequence | accuracy ↑ | file |
| T2-R3-02 | Permuted EMNIST (letters) | continuous sequence | accuracy ↑ | file |
| T2-R1-04 | TAPE Stability | amino acid tokens | spearman_valid ↑ | file |
| T2-R1-09 | enwik8 | bytes | val_bpc ↓ | file |
| T2-R1-08 | IMDB (byte-level) | bytes | accuracy ↑ | file |
| T2-R2-02 | Lipophilicity (SMILES) | SMILES tokens | rmse_valid ↓ | file |
| promoters | Human non-TATA promoter classification | ACGTN IDs, 0 padding | accuracy ↑ | file |
| bach | DCMLab Bach chorales | 128 MIDI-pitch binary flags per sixteenth note | NLL (nats per next frame) ↓ | file |

## Data

Extract the [four original and supplemental bundles](drive_upload_20261123/README.md) into a single directory.

```bash
pip install -r requirements.txt
bash setup.sh formal
bash setup.sh retnet
python -m validation prepare --source incoming --data-root data
python -m release_tasks data --source incoming --data-root data
```

The frozen `validation/protocol.json` still defines the original 16 tasks. Genomics and Bach use the [extension protocol](release_tasks/protocol.json).
