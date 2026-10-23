# sc_raw — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | Speech Commands 12-class 분류, 1초 원 파형 입력 |
| Tier (문헌 기대 라벨) | Tier 2 lose |
| 유형 | 실제 데이터, 시퀀스 (연속값 파형) |
| 입력 → 출력 | `[B,16000,1]` → `logits[B,12]`. cross-entropy. 다운샘플링·패치화 없음 |
| 시퀀스 길이 | 16000 |
| vocab | 해당 없음 (연속값). 클래스 12 |
| 지표 | `val_accuracy`, **높을수록 좋음**. split `validation`, 선택 `final_update` |
| 예산 | full 2048 update, batch 2 (epoch 0회) / smoke `--smoke_test` (3 update) |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task sc_raw --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task sc_raw --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m sc_raw.train --help`

## 데이터

- 출처: Speech Commands 원 파형 캐시 `{training,validation,testing}_{x,y}.npy` (train 36923 / val 4443 / test 4888, `sc_raw/data.py`의 `SPLIT_SIZES`).
  캐시는 `experiments/scraw/scraw_bootstrap.py`가 기존 12-class 캐시를 원 파형으로 다시 쓴 것이다.
- 다운로드: 자동 다운로드 없음. 재구성에는 원 데이터와 기존 12-class 선택 캐시가 필요하다 ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 라이선스: 확인 불가 (저장소에 기재 없음).

## 알려진 한계

- budget_limited: 2048 update, 완료 epoch 0회다. 과거 파일럿은 무작위 수준이었으며 acceptance 결과가 아니다 (프로토콜 limitations).
- 학습은 seed에 따라 다른 subset을 중복 없이 쓴다. 수렴이나 full-epoch 주장은 없다 (프로토콜 limitations).
- 길이 16000을 임의로 패치화하면 평가 변경이다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#그대로-연결할-수-없는-경우)).
- 비언어 태스크라 Tier 1↔2 변별력 검증에만 쓰고 Tier 3 예측 근거로는 쓰지 않는다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#5-tier-2-lose-행은-tier-3-예측-주장에서-제외한다)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/sc_raw-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 인자 없는 생성 함수를 등록하고 출력 shape를 지킨다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
