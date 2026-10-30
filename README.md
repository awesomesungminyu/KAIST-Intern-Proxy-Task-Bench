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
승·패는 **Transformer 기준** 분류다. 별표(*)는 선정 목록의 표기를 유지했으며 의미는 아직 명시되지 않았다.

## 프로토콜 두 버전

아래 두 버전은 이전 과제 구성을 기록한 프로토콜이며 최종 16개 과제용 프로토콜이 아니다. 최종 목록 선정과 실행 구현·프로토콜 갱신은 구분한다.

| 프로토콜 | 대상과 모델 |
|---|---|
| `v1.json` | 7개 태스크(regbench, g2p, zinc, parity, tomita5, tomita6, fflm), 태스크별 기준 모델 |
| v2.json | 공통 집합: 같은 뼈대에 mixer만 바꾼 9종(transformer, gqa, linear_attention, gla, mamba, hyena, lstm, s4d, retnet)을 8개 태스크(regbench, mqar, jelassi_copy, parity, tomita5, tomita6, scifar, fflm)에 적용 |

v2는 원본 벤치마크 저장소의 `common-arch-set-v2` 브랜치 참고. 이 스냅샷에는 `v2.json`이 없다. **두 프로토콜 결과는 `compare` 불가.**

## 테스트 대상

| 태스크 | 입력 | 지표(↑ 높을수록, ↓ 낮을수록 좋음) | 데이터 |
|---|---|---|---|
| T1-R1-05 · MAD Compression | 토큰 | 미확정 | 생성(합성) |
| T1-R2-01 · Positional Histogram | 토큰 | 미확정 | 생성(합성) |
| T1-R2-02 · String Matching | 토큰 | 미확정 | 생성(합성) |
| T1-R0-07 · RegBench | 토큰 | `next_token_acc` ↑ | 생성(PFA) |
| T1-R0-04 · Parity | 토큰 | `mean_best_val_accuracy` ↑ | 파일(TFL corpus) |
| T1-R2-03 · Modular Arithmetic | 토큰 | 미확정 | 생성(합성) |
| T1-R2-04 · Cycle Navigation | 토큰 | 미확정 | 생성(합성) |
| T1-R2-05 · Random Automaton | 토큰 | 미확정 | 생성(합성) |
| T2-R1-11 · Permuted MNIST | 이미지 시퀀스 | 미확정 | 파일(MNIST) |
| T2-R3-05 · Permuted Kuzushiji-49 | 이미지 시퀀스 | 미확정 | 파일(Kuzushiji-49) |
| T2-R4-02 · Speech Commands (raw waveform)\* | 연속값 | 미확정 | 파일(Speech Commands) |
| T2-R2-07 · Beijing PM2.5\* | 연속값 | 미확정 | 파일(Beijing PM2.5) |
| T2-R1-04 · TAPE Stability | 아미노산 서열 | 미확정 | 파일(TAPE Stability) |
| T2-R1-09 · enwik8 | 바이트 | 미확정 | 파일(enwik8) |
| T2-R1-08 · IMDB (byte-level) | 바이트 | 미확정 | 파일(IMDB) |
| T2-R2-02 · Lipophilicity (SMILES)\* | SMILES 토큰 | 미확정 | 파일(Lipophilicity) |

지표 키는 기존 RegBench·Parity 구현에서 확인된 항목만 기재했다. 나머지 과제의 최종 지표 키와 평가 집계 방식은 프로토콜 갱신 시 확정한다.

<a id="실행-인터페이스"></a>

## 빠른 시작

아래 명령은 현재 스냅샷의 기존 실행 인터페이스를 점검한다. `validation list` 출력은 최종 16개 선정 목록과 다를 수 있다.

실험 데이터는 [proxy-task-bench-data](https://drive.google.com/drive/folders/1FXi_UlvUnvrlfBj7YryFY0dmoE7uoZes?usp=drive_link)에서 받는다.
다운로드한 압축 파일은 함께 제공된 `SHA256SUMS.txt`로 검증한다.
압축을 푼 뒤 태스크별 데이터 폴더의 상위 경로를 `PROXY_DATA_ROOT`로 export한다.

개인 저장소를 clone한 상위 경로에서 아래와 같이 프로젝트 폴더로 이동한다. `check`는 CUDA 환경·자산 점검이며 학습하지 않는다.

```bash
cd KAIST-Intern-Proxy-Task-Bench
python -m validation list
python -m validation check --task regbench --model transformer
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

## 한계

- v1은 태스크마다 transformer 구현이 달랐다.
- 결과·데이터는 저장소에 포함하지 않는다.
- 최종 16개 선정 목록과 기존 스냅샷의 구현·프로토콜 사이에 차이가 있다.
- Tier 2는 언어·비언어 과제가 섞여 있으므로 Tier 3 예측 근거는 과제별로 검토한다.
