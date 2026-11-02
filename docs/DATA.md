# 데이터 준비 — 최종 v4

생성형 5개 / 나머지 11개다. 생성형은 Histogram, String Matching, RegBench, Cycle Navigation, Random Automaton이다. Tomita 5는 기존 코퍼스를 재사용하므로 파일이 필요하다.

## 1. 기존 묶음과 추가 파일

[drive_upload_20261030 (Google Drive)](https://drive.google.com/drive/folders/1KAhMI1h7CzB8pR4h6JvhyoG4rD_1dH7l) 폴더의 기존 묶음은 아래와 같다. 기존 4개 묶음은 그대로 두고 additions와 5개 묶음용 SHA256SUMS.txt를 준비했다. 서버 사본이며 Drive 업로드 완료를 뜻하지 않는다.

| 파일 | 크기 | 내용과 v4 사용 여부 |
|---|---|---|
| `proxy-data-final-pool-small.tar.gz` | 152,684,158 B | MNIST, Kuzushiji49, TAPE Stability, enwik8, IMDB, Lipophilicity. Beijing은 미사용 |
| `proxy-data-final-pool-speech_commands.tar.gz` | 2,427,893,343 B | Speech Commands 원 파형 |
| `proxy-data-final-pool-mad.tar.gz` | 327,195 B | MAD Compression 캐시 |
| `proxy-data-final-pool-parity.tar.gz` | 279,977 B | Parity 코퍼스 |
| `proxy-data-final-pool-additions.tar.gz` | 31,513,733 B | Tomita 5·EMNIST 파일 4개 |
| `SHA256SUMS.txt` | 518 B | 위 5개 묶음 해시 |

Lipophilicity 점검: small 안의 CSV는 복원 원본과 동일(SHA256 `aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd`)하여 이상 없음. small-v2는 만들지 않았다.

**아래 4개 파일은 additions 묶음에만 담았다.** [파일별 SHA256·크기](DRIVE_ADDITIONS_20261031.csv). 받은 파일은 `downloads/extracted/` 아래 다음 상대 경로에 둔다.

| 상대 경로 | 크기(B) | 용도 |
|---|---:|---|
| `T1-R0-05/train_corpus.pk` | 870567 | T1-R0-05 |
| `T1-R0-05/val_corpus_bins.pk` | 577782 | T1-R0-05 |
| `T2-R3-02/train_x.npy` | 97843328 | T2-R3-02 |
| `T2-R3-02/train_y.npy` | 998528 | T2-R3-02 |

## 2. 검증 · 압축 해제 · 준비

```bash
(cd downloads && sha256sum -c SHA256SUMS.txt)
mkdir -p downloads/extracted
for f in downloads/proxy-data-final-pool-*.tar.gz; do
  tar -xzf "$f" -C downloads/extracted
done
# additions도 함께 풀면 위 4개 파일이 <ID>/ 아래에 놓인다.
python -m validation prepare --source downloads/extracted --data-root data
python -m validation data --data-root data
```

추가 파일은 CSV의 SHA256과 대조한다. `prepare`는 원자료를 바꾸거나 다운로드·학습하지 않는다. 이미 완료한 폴더를 재사용하고 불완전한 폴더를 덮어쓰지 않는다. `--task <ID>`로 하나씩 준비할 수 있다. additions가 없으면 해당 2개 과제만 준비에 실패하며 나머지는 계속 처리한다.

## 3. 필요한 파일

모든 과제는 `--data-root/<ID>/`를 사용한다. 기본값은 `data/`, 환경변수는 `PROXY_DATA_ROOT`다. `.env`를 자동으로 읽지 않는다.

| ID | 과제 | 필요한 파일 |
|---|---|---|
| T1-R1-05 | MAD Compression | `inputs.npy` |
| T1-R2-01 | Positional Histogram | 없음 (실행 시 생성) |
| T1-R2-02 | String Matching | 없음 (실행 시 생성) |
| T1-R0-07 | RegBench | 없음 (실행 시 생성) |
| T1-R0-04 | Parity | `train_corpus.pk`, `val_corpus_bins.pk` |
| T1-R0-05 | Tomita 5 | `train_corpus.pk`, `val_corpus_bins.pk` |
| T1-R2-04 | Cycle Navigation | 없음 (실행 시 생성) |
| T1-R2-05 | Random Automaton | 없음 (실행 시 생성) |
| T2-R1-11 | Permuted MNIST | `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz` |
| T2-R3-05 | Permuted Kuzushiji-49 | `train_x.npy`, `train_y.npy` |
| T2-R4-02 | Speech Commands (raw waveform) | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R3-02 | Permuted EMNIST (letters) | `train_x.npy`, `train_y.npy` |
| T2-R1-04 | TAPE Stability | `train.csv`, `valid.csv` |
| T2-R1-09 | enwik8 | `enwik8.zip` |
| T2-R1-08 | IMDB (byte-level) | `train_x.npy`, `train_y.npy`, `val_x.npy`, `val_y.npy` |
| T2-R2-02 | Lipophilicity (SMILES) | `Lipophilicity.csv`, `split.npz` |

EMNIST는 upright 원화소 배열을 받아 seed0 위치 순열과 학습 통계 표준화를 적용한다. 공식 letters IDX gzip 두 파일을 공급하면 전치 및1→0 레이블 변환으로 같은 배열을 준비할 수도 있다.


Lipophilicity는 small 묶음의 `round2/t2m_lipo/Lipophilicity.csv`를 읽는다. `split.npz`가 없으면 RDKit으로 scaffold 분할을 생성한다. SMILES 문자 토큰, PAD=0, 학습 라벨 통계 표준화를 사용한다. Kuzushiji-49·IMDB·Speech Commands도 원본에서 캐시 전처리가 필요하다. MAD·Parity·Tomita 5·EMNIST는 제공 캐시/코퍼스, MNIST·TAPE·enwik8은 원본 파일을 복사하며 로더가 변환한다.

## 4. 분할

| 과제 | 학습 / 검증 | 규칙 |
|---|---|---|
| Tomita 5 | 10,000 / 2,000×2 bins | 기존 라운드0 코퍼스. 학습 길이2~50, 검증4~50와52~100. bin별 독립 최고 epoch 정확도의 평균 |
| Parity | 제공된 코퍼스 | 기존 코퍼스와 검증 bins를 그대로 재사용. bin별 독립 최고 정확도 평균 |
| MAD Compression | 11,520 / 1,280 | 제공된12,800개 캐시에서 마지막10%를 검증으로 사용 |
| Histogram / String Matching / Cycle / Random Automaton | 실행 시 생성 | 코드에 고정된 학습 생성기와 별도 검증 길이·seed 규칙 |
| RegBench | 생성한 학습 / 반복 관측 개발 스트림 | 평가 스트림을 최종 독립 test로 해석하지 않음 |
| Permuted MNIST | 55,000 / 5,000 | 공식 학습 풀에서 seed1 검증 분리, 위치 순열seed0 |
| Permuted EMNIST letters | 119,800 / 5,000 | 공식 학습124,800, seed1 검증 분리, transpose로 upright, 위치 순열seed0. 공식 test 미사용 |
| Permuted Kuzushiji49 | 227,365 / 5,000 | 공식 학습 풀에서 seed1 검증 분리, 위치 순열seed0 |
| Speech Commands | 84,843 / 9,981 | 공식 train/validation, test11,005 미사용. 16kHz→2kHz·길이2000, 학습 통계 표준화 |
| Lipophilicity (SMILES) | 3,360 / 420 | Bemis–Murcko scaffold 80/10/10(DeepChem 규칙, 키랄성 무시). test 420 미사용 |
| TAPE Stability | 53,614 / 2,512 | 공식 train/valid, test 미사용 |
| IMDB bytes | 22,500 / 2,500 | 공식 train에서 seed0로10% 검증 분리, 길이2048 |
| enwik8 | 처음90M 바이트 / 다음1M 바이트 | `[90,000,000,91,000,000)`의 검증을 길이256·비중첩3,906구간으로 평가. 나머지9M 미사용 |

Speech Commands만 S5 확정 표시(*)를 쓴다. EMNIST는 예비(다양성 초과)이며 파일 배포 여부와 판정 상태를 혼동하지 않는다.

Lipophilicity: 도메인 우선 규칙으로 포함(판별비 3.47, 기준 5 미달). [docs/PREREG_DOMAIN_PRIORITY.md](PREREG_DOMAIN_PRIORITY.md) 참고.
