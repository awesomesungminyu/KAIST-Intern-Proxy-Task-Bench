# proxy-task-bench

한국어 · [English](README.en.md)

Transformer 대안 구조를 합성 과제와 소규모 실제 데이터로 평가하는 16개 과제 벤치마크. 셸 명령을 실행할 수 있는 EvoScientist에서도 사용할 수 있다.

## 왜 하는가

새 구조를 대규모 학습으로 바로 검증하면 비용이 큽니다.
학습 없는 지표, 합성 단위 테스트, 축소한 실제 과제를 단계적으로 활용해 대규모 학습 전에 후보를 선별하려 합니다.

## Tier 0~3

| Tier | 이름 | 무엇을 재는가 | 비용 |
|---|---|---|---|
| 0 | Zero-cost | 학습 없이 초기 상태 지표 | 초 단위 |
| 1 | 합성 단위 테스트 | 능력 하나를 격리해 검증 | 분 단위 |
| 2 | 축소한 실제 데이터 | 작은 규모에서 실제 과제 수행 능력 | 시간 단위 |
| 3 | ETRI 대규모 AR LM | 대규모 AR LM 성능 | 대규모 학습 |

## 티어별 소속 태스크

### Tier 0

- **tier0**: 학습 없이 지표 계산

### Tier 1 · 합성 과제 · Transformer 승

- **MAD Compression**: 압축
- **Positional Histogram**: 위치별 빈도
- **String Matching**: 문자열 매칭
- **RegBench**: 정규언어 ICL

### Tier 1 · 합성 과제 · Transformer 패

- **Parity**: 패리티
- **Modular Arithmetic**: 모듈러 연산
- **Cycle Navigation**: 순환 경로 탐색
- **Random Automaton**: 오토마톤 상태 추적

### Tier 2 · 실제 데이터 · Transformer 승

- **Permuted MNIST**: 순열 이미지 분류
- **Permuted Kuzushiji-49**: 순열 이미지 분류
- **Speech Commands (raw waveform)\***: 음성 분류
- **Beijing PM2.5\***: 대기 시계열 예측

### Tier 2 · 실제 데이터 · Transformer 패

- **TAPE Stability**: 단백질 안정성
- **enwik8**: 바이트 언어 모델링
- **IMDB (byte-level)**: 감성 분류
- **Lipophilicity (SMILES)\***: 분자 물성 예측

### Tier 3

- **(ETRI)**: 대규모 AR LM

최종 과제는 **Tier 1 8개 + Tier 2 8개 = 총 16개**다. Tier 0과 Tier 3은 이 개수에 포함하지 않는다.
승·패는 **Transformer 기준** 분류다.

## 테스트 대상

| 태스크 | 입력 | 지표(↑ 높을수록, ↓ 낮을수록 좋음) | 데이터 |
|---|---|---|---|
| [T1-R1-05 · MAD Compression](tasks/T1-R1-05/README.md) | 토큰 | `token_accuracy` ↑ | 파일(합성 캐시) |
| [T1-R2-01 · Positional Histogram](tasks/T1-R2-01/README.md) | 토큰 | `slot_accuracy` ↑ | 실행 시 생성 |
| [T1-R2-02 · String Matching](tasks/T1-R2-02/README.md) | 토큰 | `accuracy` ↑ | 실행 시 생성 |
| [T1-R0-07 · RegBench](tasks/T1-R0-07/README.md) | 토큰 | `next_token_acc` ↑ | 실행 시 생성 |
| [T1-R0-04 · Parity](tasks/T1-R0-04/README.md) | 토큰 | `mean_best_val_accuracy` ↑ | 파일 |
| [T1-R2-03 · Modular Arithmetic](tasks/T1-R2-03/README.md) | 토큰 | `accuracy_id50_ex50` ↑ | 실행 시 생성 |
| [T1-R2-04 · Cycle Navigation](tasks/T1-R2-04/README.md) | 토큰 | `accuracy_id50_ex50` ↑ | 실행 시 생성 |
| [T1-R2-05 · Random Automaton](tasks/T1-R2-05/README.md) | 토큰 | `last_state_accuracy_id50_ex50` ↑ | 실행 시 생성 |
| [T2-R1-11 · Permuted MNIST](tasks/T2-R1-11/README.md) | 이미지 시퀀스 | `accuracy` ↑ | 파일 |
| [T2-R3-05 · Permuted Kuzushiji-49](tasks/T2-R3-05/README.md) | 이미지 시퀀스 | `accuracy` ↑ | 파일 |
| [T2-R4-02 · Speech Commands (raw waveform)\*](tasks/T2-R4-02/README.md) | 연속값 | `accuracy` ↑ | 파일 |
| [T2-R2-07 · Beijing PM2.5\*](tasks/T2-R2-07/README.md) | 연속값 | `mse_ctx_id50_long50` ↓ | 파일 |
| [T2-R1-04 · TAPE Stability](tasks/T2-R1-04/README.md) | 아미노산 서열 | `spearman_valid` ↑ | 파일 |
| [T2-R1-09 · enwik8](tasks/T2-R1-09/README.md) | 바이트 | `val_bpc` ↓ | 파일 |
| [T2-R1-08 · IMDB (byte-level)](tasks/T2-R1-08/README.md) | 바이트 | `accuracy` ↑ | 파일 |
| [T2-R2-02 · Lipophilicity (SMILES)\*](tasks/T2-R2-02/README.md) | SMILES 토큰 | `rmse_valid` ↓ | 파일 |

## 빠른 시작

Linux, Python 3.11, NVIDIA CUDA GPU를 사용한다. 과제 목록 조회와 CPU 테스트는 GPU 없이도 가능하다.

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

PyTorch 설치 예시는 CUDA 12.4 빌드다. 다른 드라이버 환경은 [공식 설치 표](https://pytorch.org/get-started/previous-versions/#v251)를 참고하되 PyTorch 2.5.1을 유지한다.

**데이터: [drive_upload_20261030 (Google Drive)](<drive_upload_20261030 폴더 링크>)**

생성형 과제 6개는 다운로드 불필요, 나머지 10개는 이 폴더의 묶음 4개로 준비한다. 다운로드할 파일과 체크섬 검증·압축 해제 방법은 [데이터 준비](docs/DATA.md)에 있다.

```bash
# 다운로드한 압축 파일을 풀어 둔 폴더에서 입력을 준비한다.
python -m validation prepare --source downloads/extracted --data-root data
# GPU 없이 필요한 파일의 준비 상태를 확인한다.
python -m validation data --data-root data
# 별도 데이터가 없는 생성형 과제로 실행을 먼저 확인한다.
python -m validation check --task T1-R2-02 --model cs_transformer
python -m validation run --task T1-R2-02 --model cs_transformer --profile smoke --seeds 0
```

기본 데이터 위치는 `data/<과제 ID>/`, 결과 위치는 `artifacts/<과제 ID>/<모델>/<실행 ID>/`다. 다른 저장장치는 `--data-root` 또는 `.env.example`의 선택 설정으로 지정한다. `.env`는 자동으로 읽지 않는다.

Parity는 `bash setup.sh parity`, RetNet은 `bash setup.sh retnet`으로 외부 실행기·커널을 먼저 준비한다.

```bash
# 전체 예산 실행: 저장 폴더는 새 경로를 사용한다.
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile full --seeds 0 1 2 --out artifacts/string-matching-full
# 같은 과제·데이터·설정·시드의 완료된 두 실행을 비교한다.
python -m validation compare --baseline artifacts/baseline --candidate artifacts/candidate
# CPU 계약 테스트
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
# 16개 과제의 Transformer import + 학습 1스텝 확인 (데이터 준비 후)
python scripts/smoke_tasks.py --data-root data --device cuda
```

`check`는 환경·파일 점검, `smoke`와 1스텝 검사는 동작 확인이며 성능 비교 근거가 아니다. 결과 JSON의 상태가 `done`인 `full` 실행만 비교할 수 있다.
새 구조 추가와 동일 조건 비교는 [모델 안내](docs/MODELS.md), 에이전트 실행 순서는 [EvoScientist 안내](docs/EVOSCIENTIST.md)를 참고한다.

## 알려진 문제

- Drive 폴더에서 묶음 4개(`proxy-data-final-pool-{mad,parity,small,speech_commands}.tar.gz`) 중 보이지 않는 것이 있으면 폴더 소유자에게 요청해야 한다. 해당 데이터가 필요한 과제는 준비할 수 없다.

## 한계

- 16개 과제 선정과 import·학습 1스텝 검증은 전체 학습·모든 모델 조합의 성능 검증 완료를 뜻하지 않는다.
- 결과·데이터·체크포인트는 Git에 포함하지 않는다.
- RegBench는 반복 관측한 개발 스트림을 평가한다. Parity는 bin별 최고 정확도의 평균이며 단일 체크포인트의 점수가 아니다.
- Tier 0·3은 검증 체계 설명용이며 이 저장소의 실행 목록에는 Tier 1·2의 최종 16개만 포함한다.
