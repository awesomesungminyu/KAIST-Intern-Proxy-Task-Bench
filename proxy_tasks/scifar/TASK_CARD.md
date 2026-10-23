# scifar — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | Sequential CIFAR-10 (RGB 픽셀을 1024 시점 시퀀스로 분류) |
| Tier (문헌 기대 라벨) | Tier 2 lose |
| 유형 | 실제 데이터, 시퀀스 (연속값 픽셀) |
| 입력 → 출력 | train 통계로 정규화한 RGB raster `[B,1024,3]` → `logits[B,10]`. cross-entropy |
| 시퀀스 길이 | 1024 |
| vocab | 해당 없음 (연속값). 채널 3, 클래스 10 |
| 지표 | `val_accuracy`, **높을수록 좋음**. split `validation`, 선택 `highest_validation_accuracy_checkpoint` |
| 예산 | full 10 epoch, batch 64, lr 1e-3, wd 0.01 / smoke `--smoke_test` (1 epoch) |
| 런타임 | torch 2.5.1 + torchvision 0.20.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task scifar --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task scifar --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m scifar.train --help`

## 데이터

- 출처: CIFAR-10 Python batches (`cifar/cifar-10-batches-py/`), torchvision `CIFAR10(download=False)`로 읽는다 (`scifar/data.py`).
  train 50000장을 seed 42로 45000/5000 (train/validation)으로 나누고, 공식 test 10000장은 따로 둔다.
- 다운로드: 직접 CLI는 다운로드하지 않는다. 새 서버는 torchvision CIFAR10으로 준비한다 ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 라이선스: 확인 불가 (저장소에 기재 없음).

## 알려진 한계

- budget_limited: 과거 파일럿 두 개 모두 10 epoch에서 미수렴했으며 acceptance 결과가 아니다 (프로토콜 limitations).
- augmentation이 없다. test는 후보 확정 뒤에만 평가한다 (프로토콜 limitations).
- 비언어 태스크라 Tier 1↔2 변별력 검증에만 쓰고 Tier 3 예측 근거로는 쓰지 않는다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#5-tier-2-lose-행은-tier-3-예측-주장에서-제외한다)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/scifar-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 인자 없는 생성 함수를 등록하고 출력 shape를 지킨다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
