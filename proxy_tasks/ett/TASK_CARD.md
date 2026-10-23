# ett — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | ETTh1 다변량 시계열 예측, 과거 96시점 → 다음 96시점 |
| Tier (문헌 기대 라벨) | Tier 2 lose |
| 유형 | 실제 데이터, 시퀀스 (연속값 시계열) |
| 입력 → 출력 | train 통계로 표준화한 `[B,96,7]` → `[B,96,7]`. 손실 MSE |
| 시퀀스 길이 | 입력 96, 예측 96 (`ett/data.py`의 `INPUT_STEPS`, `PREDICTION_STEPS`) |
| vocab | 해당 없음 (연속값). 채널 7개 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |
| 지표 | `val_mse`, **낮을수록 좋음**. split `validation`, 선택 `lowest_validation_mse_checkpoint` |
| 예산 | full 최대 20 epoch, patience 3 조기 종료, `naive`는 학습 0 epoch / smoke `--smoke_test` |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `linear` (DLinear), `transformer`, `naive`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task ett --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task ett --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m ett.train --help`

## 데이터

- 출처: `zhouhaoyi/ETDataset`의 `ETT-small/{ETTh1,ETTm1}.csv` (`experiments/quick_suite/quick_prepare.py`의 다운로드 URL). 두 CSV의 SHA256은 `ett/data.py`에 고정되어 있다.
- 다운로드: 직접 CLI는 다운로드하지 않는다. 같은 CSV를 `$PROXY_DATA_ROOT/ett/`에 둔다 ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 라이선스: 확인 불가 (저장소에 기재 없음).

## 알려진 한계

- 프로토콜은 ETTh1만 쓴다. ETTm1은 직접 CLI `--dataset ETTm1`로만 실행하며 섞지 않는다 (프로토콜 limitations).
- validation이 체크포인트를 고른다. test는 후보를 확정한 뒤 직접 trainer `--final_test`로만 평가한다 (프로토콜 limitations).
- 과거 파일럿은 acceptance 결과가 아니다 (프로토콜 limitations).
- `naive`는 학습 예산 0이라 학습 모델과의 matched-budget compare가 거부된다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
- 비언어 태스크라 Tier 1↔2 변별력 검증에만 쓰고 Tier 3 예측 근거로는 쓰지 않는다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#5-tier-2-lose-행은-tier-3-예측-주장에서-제외한다)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/ett-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 학습 파라미터가 없는 모델은 `NO_TRAINING`과 프로토콜 `no_training_models`에도 추가한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
