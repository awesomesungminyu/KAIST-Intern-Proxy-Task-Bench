# proxy-task-bench

EvoScientist가 제안한 새 Transformer 대안 구조를 싼 실험으로 미리 걸러내는 4단계(Tier) 검증 하니스.

## 왜 하는가

새 구조를 대규모 학습(Tier 3)으로 바로 검증하면 비싸다.
싼 Tier의 순위가 비싼 Tier의 순위를 예측하는지 인접 Tier 간 Spearman ρ로 확인하려 한다.
예측력이 있으면 싼 실험으로 후보를 먼저 거를 수 있다.
**이 ρ는 아직 측정하지 않았다.**

## Tier 0~3

| Tier | 이름 | 무엇을 재는가 | 비용 |
|---|---|---|---|
| 0 | Zero-cost | 학습 없이 초기 상태 지표 | 초 단위 |
| 1 | 합성 단위 테스트 | 능력 하나를 격리해 검증 | 분 단위 |
| 2 | 축소한 실제 데이터 | 작은 규모에서 실제 과제 수행 능력 | 시간 단위 |
| 3 | 대규모 언어 LM | 대규모 AR LM 성능 | 대규모 학습 — ETRI 담당, 저장소 범위 밖 |

## 티어별 소속 태스크

| Tier | 구분 | 태스크 | 무엇을 하나 |
|---|---|---|---|
| Tier 0 | - | tier0 | 학습 없이 지표 계산 |
| Tier 1 | win | mqar, mad_tsc, jelassi_copy, iclreg(후보, 미확정) | 연관 회상, 선택 복사, 문자열 복사, in-context 회귀 |
| Tier 1 | lose | parity, tomita5, tomita6, fflm | 형식언어, flip-flop(상태 추적) |
| Tier 2 | win | regbench, g2p, zinc | 정규언어 ICL, 자소→음소, 분자 그래프 |
| Tier 2 | lose | ett, scifar, sc_raw | 시계열, 이미지 시퀀스, 원 파형 |
| Tier 3 | - | (ETRI) | 대규모 AR LM |

win = 새 구조가 이길 것으로 예상, lose = 질 것으로 예상하는 문헌 기대 라벨이며 측정으로 확정된 것이 아니다.

## 프로토콜 두 버전

| 프로토콜 | 대상과 모델 |
|---|---|
| [v1.json](validation/protocols/v1.json) | 15개 태스크, 태스크마다 기준 모델(transformer, lstm 등) |
| v2.json | 공통 집합: 같은 뼈대에 mixer만 바꾼 9종(transformer, gqa, linear_attention, gla, mamba, hyena, lstm, s4d, retnet)을 8개 태스크(regbench, mqar, jelassi_copy, parity, tomita5, tomita6, scifar, fflm)에 적용 |

v2는 `common-arch-set-v2` 브랜치 참고. 현재 브랜치에는 `v2.json`이 없다. **두 프로토콜 결과는 `compare` 불가.**

## 테스트 대상

| 태스크 | 입력 | 지표(↑ 높을수록, ↓ 낮을수록 좋음) | 데이터 |
|---|---|---|---|
| tier0 | 토큰 | `multi_metric`: proxy ↑, control ↓ (단일 순위 없음) | 생성(랜덤) |
| mqar | 토큰 | `best_valid_accuracy` ↑ | 생성(zoology) |
| mad_tsc | 토큰 | `val_accuracy` ↑ | 파일(합성 캐시) |
| jelassi_copy | 토큰 | `ood_exact_match` ↑ | 생성 |
| iclreg | 연속값 | `val_k_ge_d_mse` ↓ | 생성 |
| parity | 토큰 | `mean_best_val_accuracy` ↑ | 파일(TFL corpus) |
| tomita5 | 토큰 | `mean_best_val_accuracy` ↑ | 파일(TFL corpus) |
| tomita6 | 토큰 | `mean_best_val_accuracy` ↑ | 파일(TFL corpus) |
| fflm | 토큰 | `id_read_error_rate` ↓ | 파일(고정 패키지) + 학습 시 생성 |
| regbench | 토큰 | `next_token_acc` ↑ | 생성(PFA) |
| g2p | 토큰 | `val_word_acc` ↑ | 파일(CMUDict) |
| zinc | 그래프 | `best_val`(validation MAE) ↓ | 파일(ZINC) |
| ett | 연속값 | `val_mse` ↓ | 파일(ETTh1) |
| scifar | 연속값 | `val_accuracy` ↑ | 파일(CIFAR-10) |
| sc_raw | 연속값 | `val_accuracy` ↑ | 파일(Speech Commands) |

<a id="실행-인터페이스"></a>

## 빠른 시작

저장소 루트에서 실행한다. `check`는 CUDA 환경·자산 점검이며 학습하지 않는다.

```bash
python -m validation list
python -m validation check --task regbench --model transformer
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

## 한계

- 인접 Tier 간 Spearman ρ는 아직 미측정이다.
- v1은 태스크마다 transformer 구현이 달랐다.
- 결과·데이터는 저장소에 포함하지 않는다.
- Tier 2 lose는 비언어 태스크이므로 Tier 3 예측 근거에서 제외한다.

## 더 읽을 문서

[TASK_INDEX](docs/TASK_INDEX.md) · [REPO_MAP](docs/REPO_MAP.md) · [TIER_FRAMEWORK](docs/TIER_FRAMEWORK.md) · [ADDING_A_MODEL](docs/ADDING_A_MODEL.md) · [README_DETAILED](docs/README_DETAILED.md)
