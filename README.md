# proxy-task-bench

한국어 · [English](README.en.md)

EvoScientist가 제안한 새 Transformer 대안 구조를 싼 실험으로 미리 걸러내는 4단계(Tier) 검증 하니스.

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

| Tier | 구분 | 태스크 | 무엇을 하나 |
|---|---|---|---|
| Tier 0 | - | tier0 | 학습 없이 지표 계산 |
| Tier 1 | Transformer 승 | MAD Compression, Positional Histogram, String Matching, RegBench | 압축, 위치별 빈도, 문자열 매칭, 정규언어 ICL |
| Tier 1 | Transformer 패 | Parity, Modular Arithmetic, Cycle Navigation, Random Automaton | 패리티, 모듈러 연산, 순환 경로 탐색, 오토마톤 상태 추적 |
| Tier 2 | Transformer 승 | Permuted MNIST, Permuted Kuzushiji-49, Speech Commands (raw waveform)*, Beijing PM2.5* | 순열 이미지 분류, 음성 분류, 대기 시계열 예측 |
| Tier 2 | Transformer 패 | TAPE Stability, enwik8, IMDB (byte-level), Lipophilicity (SMILES)* | 단백질 안정성, 바이트 언어 모델링, 감성 분류, 분자 물성 예측 |
| Tier 3 | - | (ETRI) | 대규모 AR LM |

최종 과제는 Tier 1 8개, Tier 2 8개로 총 16개다. Tier 0과 Tier 3은 이 개수에 포함하지 않는다.
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

실험 데이터는 [proxy-task-bench-data](https://drive.google.com/drive/folders/1FXi_UlvUnvrlfBj7YryFY0dmoE7uoZes?usp=drive_link)에서 받는다.
다운로드한 압축 파일은 함께 제공된 `SHA256SUMS.txt`로 검증한다.
데이터 폴더 구조와 과제별 필수 파일은 [데이터 안내](docs/DATA.md)를 따른다.

```bash
cd KAIST-Intern-Proxy-Task-Bench
export PROXY_DATA_ROOT=/absolute/path/to/data
python -m validation list
python -m validation check --task T1-R0-07 --model cs_transformer
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

CUDA용 PyTorch 2.5.1과 `requirements.txt`의 의존성이 필요하다. Parity의 외부 실행기는 `bash setup.sh parity`, RetNet의 CUDA 커널은 `bash setup.sh retnet`으로 준비한다.
`check`는 환경·필수 파일 점검이며 학습하지 않는다. `list`에는 최종 16개 과제만 표시된다.

```bash
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile smoke --seeds 0 --out artifacts/string-matching-smoke
# 전체 예산 실행
python -m validation run --task T1-R2-02 --model cs_transformer \
  --profile full --seeds 0 1 2 --out artifacts/string-matching-full
```

9종 공통 모델과 과제별 설정은 [모델 안내](docs/MODELS.md), 실행 설정은 `validation/protocol.json`에 있다.

## 한계

- 16개 과제의 최종 선정은 모든 모델·시드의 반복 검증 완료를 뜻하지 않는다.
- 결과·데이터·체크포인트는 저장소에 포함하지 않는다.
- `smoke`는 동작 점검이며 성능 비교 근거가 아니다.
- RegBench는 반복 관측한 개발 스트림을 평가한다. Parity는 bin별 최고 정확도의 평균으로, 단일 체크포인트의 점수가 아니다.
- Tier 2는 언어·비언어 과제가 섞여 있으므로 Tier 3 예측 근거는 과제별로 검토한다.
