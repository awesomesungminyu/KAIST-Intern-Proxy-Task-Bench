# tier0 — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | 학습 없이 init 상태 모델의 zero-cost proxy 지표와 대조군을 계산 (v2) |
| Tier (문헌 기대 라벨) | Tier 0 |
| 유형 | 균일 랜덤 토큰, 시퀀스 (학습 0회) |
| 입력 → 출력 | 등록 단위는 **mixer** `[B,L,d] → [B,L,d]`, causal. `family`, `state_elements(cfg)`를 함께 등록 |
| 시퀀스 길이 | full L = 256, probe 128 / smoke(`--quick`) L = 64, probe 8 |
| vocab | 256 (`tier0_proxy.py`의 `Cfg.vocab`) |
| 지표 | `multi_metric` (단일 순위 없음, 기술통계만). proxy: `snip`, `mix_nuc`, `mlp_sal`, `naswot`, `ntk_cond` (**높을수록**) / control: `params`, `flops`, `state_cap` (**낮을수록**) |
| 예산 | 학습 0 step. full d128 / 4 layers / batch 8, seed당 한 번 점수화 / smoke `--quick` |
| 런타임 | torch 2.5.1, `--device cuda` |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`, 11종): `transformer`, `swa`, `gla`, `rwkv6`, `based`, `s4d`, `mamba`, `h3`, `hyena`, `lstm`, `cnn`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task tier0 --model transformer
# 점수화를 승인한 세션에서만 (학습은 없음)
python -m validation run --task tier0 --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 하니스 도움말: `cd proxy_tasks && python -m tier0.tier0_proxy --help` (직접 CLI의 seed 기본값은 1234부터)

## 데이터

- 출처: 입력 파일 없음. 균일 랜덤 토큰을 생성한다.
- 라이선스: 해당 없음.

## 알려진 한계

- `mlp_sal`은 11종에서 상수라 순위에 쓰면 안 된다 (프로토콜 limitations).
- `snip`과 `mix_nuc`는 중복일 수 있다. `ntk_cond`의 좌표 수·좌표 seed 안정성은 미검증이다 (프로토콜 limitations).
- `mix_nuc`는 원 DSS++가 아니다. 입력이 균일 랜덤 토큰이라 과제 의존 예측력이 없다. Tier 간 Spearman은 측정하지 않았다 (프로토콜 limitations).
- `flops`, `state_cap`은 분석식 대조군이다. `state_cap`은 FP32 점수화와 달리 원소당 2 bytes를 가정한다 (프로토콜 limitations).
- smoke에서는 window attention과 full attention이 구분되지 않는다 (프로토콜 limitations).
- v1(CPU)과 v2(GPU) 수치를 직접 비교하지 않는다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#tier-0-하니스)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/tier0-<model>/`. 과거 v1/v2 하니스 결과는 `artifacts/tier0/`.

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 상태 크기 분석식(`state_elements`)을 함께 등록한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
