# iclreg — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | In-context 선형 회귀 (Garg et al. 2022 축소판, 자체 구현) |
| Tier (문헌 기대 라벨) | Tier 1 win 후보. 저장소 판정은 **미확정**이며 Tier 1 win 행에 넣지 않았다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#iclreg--tier-1-win-미확정)) |
| 유형 | 생성 데이터, 시퀀스 (**연속 벡터 입력**, 토큰 아님) |
| 입력 → 출력 | `forward(xs[B,K,d] float32, ys[B,K] float32) → preds[B,K]`. `preds[:,i]`는 `ys[:,i]`를 보면 안 된다 |
| 시퀀스 길이 | 프롬프트당 41점 (K = 41), interleave 후 2K = 82. 학습과 평가 길이가 같다 |
| vocab | 해당 없음 (연속값). 입력 차원 d = 20 |
| 지표 | `val_k_ge_d_mse` (k = 20..40 구간의 (ŷ−y)²/d 평균), **낮을수록 좋음**. split `fixed_generated_validation`, 선택 `final_step` |
| 예산 | full 45000 step, batch 64, Adam lr 1e-4, curriculum 간격 2000 / smoke `--smoke_test` (20 step) |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task iclreg --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task iclreg --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m iclreg.train --help`

## 데이터

- 출처: 입력 파일 없음. w ~ N(0, I), x ~ N(0, I), y = w·x (잡음 없음)를 매 step 생성한다. validation은 seed 2147483647로 고정한 2048 프롬프트다.
- 라이선스: 해당 없음 (생성 데이터).
- 별도 test split은 없다.

## 알려진 한계

- Transformer 우위가 검증되지 않아 Tier 1 win으로 확정하지 않았다. 코드와 프로토콜 등록은 유지한다 ([README](README.md), [TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md)).
- `mamba`는 등록하지 않았다. 순수 torch 스캔이 이 예산에서 seed당 30분을 넘는다 (프로토콜 limitations).
- 연속 벡터 입력이라 다른 태스크의 token LM을 그대로 꽂을 수 없다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#iclreg-연속-벡터-입력)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/tier1_new_20261022/full/iclreg-<model>-s<seed>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. `Interleave`, `read_x_positions`로 read-in/read-out 규약을 맞춘다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#iclreg-연속-벡터-입력)).
