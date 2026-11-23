# Proxy Task Bench

EvoScientist와 연결하려면 [kaist_evo_proxy](https://github.com/awesomesungminyu/kaist_evo_proxy)를 사용합니다.

한국어 · [English](README.en.md)

Tier 1 합성 과제 8개와 Tier 2 데이터 과제 10개로 후보 mixer를 평가합니다. Tier 2의 SNLI는 화학 도메인을 포함하기 위해 Lipophilicity로 교체했습니다. Genomics와 Bach는 별도 확장으로 추가했습니다.

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

## 데이터

[기존 4개와 보완 묶음](drive_upload_20261123/README.md)을 한 폴더에 풉니다.

```bash
pip install -r requirements.txt
bash setup.sh formal
bash setup.sh retnet
python -m validation prepare --source incoming --data-root data
python -m release_tasks data --source incoming --data-root data
```

기존 `validation/protocol.json`은 16개 과제 설정을 유지합니다. Genomics와 Bach는 [별도 프로토콜](release_tasks/protocol.json)에서 읽습니다.
