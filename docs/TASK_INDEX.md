# 태스크 인덱스

`validation/protocols/v1.json`에 등록된 15개 task ID를 Tier별로 묶은 표다. Tier는 [Tier 프레임워크](TIER_FRAMEWORK.md)의
구현 현황 표를 따른 **문헌 기대 라벨**이다. "win"은 대상 아키텍처가 이길 것으로 예상되는 태스크, "lose"는 질 것으로
예상되는 태스크다. 이 라벨은 측정으로 확정한 순위가 아니다.

카드(`TASK_CARD.md`)는 입력·출력, 길이, vocab, 지표, 모델, 실행 명령, 데이터, 한계를 한 화면에 모은 요약이다.
세부 설정과 경위는 각 태스크의 기존 README에 있다. 모델 수는 `models.py` 레지스트리와 프로토콜 `models`의 항목 수다
(2026-10-23, `b45b67f` 기준 두 목록 일치).

## Tier 0 — Zero-cost proxy

| 태스크 ID | 유형 | 시퀀스 | 지표 (방향) | 모델 수 | 기준 모델 | 문서 |
|---|---|---|---|---:|---|---|
| tier0 | 균일 랜덤 토큰, 학습 없음 | 시퀀스 | `multi_metric`: proxy 5개 ↑, control 3개 ↓ (단일 순위 없음) | 11 | transformer | [카드](../proxy_tasks/tier0/TASK_CARD.md) · [README](../proxy_tasks/tier0/README.md) |

## Tier 1 — Synthetic unit test

| 태스크 ID | Tier 라벨 | 유형 | 시퀀스 | 지표 (방향) | 모델 수 | 기준 모델 | 문서 |
|---|---|---|---|---|---:|---|---|
| mqar | Tier 1 win | 합성 (zoology 생성기) | 시퀀스 | `best_valid_accuracy` ↑ | 4 | transformer | [카드](../proxy_tasks/mqar/TASK_CARD.md) · [README](../proxy_tasks/mqar/README.md) |
| mad_tsc | Tier 1 win | 합성 (MAD 고정 캐시) | 시퀀스 | `val_accuracy` ↑ | 16 | mh_attention_swiglu | [카드](../proxy_tasks/mad_tsc/TASK_CARD.md) · [README](../proxy_tasks/mad_tsc/README.md) |
| jelassi_copy | Tier 1 win | 합성 (실행 시 생성) | 시퀀스 | `ood_exact_match` ↑ | 3 | transformer | [카드](../proxy_tasks/jelassi_copy/TASK_CARD.md) · [README](../proxy_tasks/jelassi_copy/README.md) |
| iclreg | Tier 1 win 후보 (**미확정**) | 합성 (실행 시 생성, 연속 벡터) | 시퀀스 | `val_k_ge_d_mse` ↓ | 2 | transformer | [카드](../proxy_tasks/iclreg/TASK_CARD.md) · [README](../proxy_tasks/iclreg/README.md) |
| parity | Tier 1 lose | 합성 (TFL corpus) | 시퀀스 | `mean_best_val_accuracy` ↑ | 3 | san_abs | [카드](../proxy_tasks/formal_languages/TASK_CARD.md) · [README](../proxy_tasks/formal_languages/README.md) |
| tomita5 | Tier 1 lose | 합성 (TFL corpus) | 시퀀스 | `mean_best_val_accuracy` ↑ | 3 | san_abs | [카드](../proxy_tasks/formal_languages/TASK_CARD.md) · [README](../proxy_tasks/formal_languages/README.md) |
| tomita6 | Tier 1 lose | 합성 (TFL corpus) | 시퀀스 | `mean_best_val_accuracy` ↑ | 3 | san_abs | [카드](../proxy_tasks/formal_languages/TASK_CARD.md) · [README](../proxy_tasks/formal_languages/README.md) |
| fflm | Tier 1 lose | 합성 (고정 패키지) | 시퀀스 | `id_read_error_rate` ↓ | 2 | transformer | [카드](../proxy_tasks/fflm/TASK_CARD.md) · [README](../proxy_tasks/fflm/README.md) |

## Tier 2 — Small-scale

| 태스크 ID | Tier 라벨 | 유형 | 시퀀스 | 지표 (방향) | 모델 수 | 기준 모델 | 문서 |
|---|---|---|---|---|---:|---|---|
| regbench | Tier 2 win | 생성 (PFA, 실행 시 생성) | 시퀀스 | `next_token_acc` ↑ | 2 | transformer | [카드](../proxy_tasks/regbench/TASK_CARD.md) · [README](../proxy_tasks/regbench/README.md) |
| g2p | Tier 2 win | 실제 (CMUDict) | 시퀀스 (encoder-decoder) | `val_word_acc` ↑ | 2 | transformer | [카드](../proxy_tasks/g2p/TASK_CARD.md) · [README](../proxy_tasks/g2p/README.md) |
| zinc | Tier 2 win | 실제 (ZINC-12K) | **비시퀀스 (그래프)** | `best_val` (validation MAE) ↓ | 3 | gps | [카드](../proxy_tasks/zinc/TASK_CARD.md) · [README](../proxy_tasks/zinc/README.md) |
| ett | Tier 2 lose | 실제 (ETTh1) | 시퀀스 (연속값) | `val_mse` ↓ | 3 | transformer | [카드](../proxy_tasks/ett/TASK_CARD.md) · [README](../proxy_tasks/ett/README.md) |
| scifar | Tier 2 lose | 실제 (CIFAR-10) | 시퀀스 (연속값) | `val_accuracy` ↑ | 2 | transformer | [카드](../proxy_tasks/scifar/TASK_CARD.md) · [README](../proxy_tasks/scifar/README.md) |
| sc_raw | Tier 2 lose | 실제 (Speech Commands) | 시퀀스 (연속값) | `val_accuracy` ↑ | 2 | transformer | [카드](../proxy_tasks/sc_raw/TASK_CARD.md) · [README](../proxy_tasks/sc_raw/README.md) |

↑ 높을수록 좋음, ↓ 낮을수록 좋음 (프로토콜 `direction`).

## 읽을 때 주의

- 같은 이름의 `transformer`라도 태스크마다 구현이 다르다. 모든 태스크에 공통인 아키텍처 집합은 아직 없고,
  Tier 간 Spearman ρ는 측정하지 않았다 ([Tier 프레임워크 미해결 문제](TIER_FRAMEWORK.md#미해결-문제)).
- Tier 2 lose 행(ett, scifar, sc_raw)은 비언어 태스크라 Tier 3 예측 주장의 근거로 쓰지 않는다.
- Tier 3(대규모 AR LM)은 이 저장소 범위 밖이다.
- 실행 명령, 결과 계약, 비교 규칙은 [EVOSCIENTIST](EVOSCIENTIST.md), 모델 추가는 [ADDING_A_MODEL](ADDING_A_MODEL.md),
  디렉터리 구조는 [REPO_MAP](REPO_MAP.md)을 따른다.
