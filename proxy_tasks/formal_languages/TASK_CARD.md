# parity / tomita5 / tomita6 — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

이 디렉터리 하나가 task ID 세 개(`parity`, `tomita5`, `tomita6`)를 담당한다. 세 ID는 데이터셋만 다르고
모델, 지표, 예산이 같다.

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | 형식언어 인식 (Bhattamishra et al. 2020, 상류 Transformer-Formal-Languages 고정 커밋) |
| Tier (문헌 기대 라벨) | Tier 1 lose (세 ID 모두) |
| 유형 | 합성 데이터 (사전 생성 corpus), 시퀀스 |
| 데이터셋 | `parity` → `Parity`, `tomita5` → `Tomita-5`, `tomita6` → `Tomita-6` (프로토콜 `dataset`) |
| 입력 → 출력 | 상류 TFL `src/components/` 모델 계약을 따른다. 이 저장소에 shape 명세 없음 → 확인 불가 |
| 시퀀스 길이 | 확인 불가 (사전 생성 corpus에 따라 다름. 이 저장소 코드에 명시 없음) |
| vocab | 확인 불가 (상류 생성기 소관) |
| 지표 | `mean_best_val_accuracy`, **높을수록 좋음**. split `diagnostic_validation_bins`, 선택 `independent_best_epoch_per_bin` |
| 예산 | full 최대 100 epoch / smoke 2 epoch |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py`의 TFL 인자 변환표 = 프로토콜 `models`): `san_abs`, `san_mask`, `lstm`
기준 모델: `san_abs`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).
`--task`에 `parity`, `tomita5`, `tomita6` 중 하나를 준다.

```bash
python -m validation list
python -m validation check --task parity --model san_abs
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task parity --model san_abs \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

`--tfl-root`는 소스만, `--data-root`는 해당 태스크의 데이터 폴더를 지정한다 ([EVOSCIENTIST](../../docs/EVOSCIENTIST.md#자산-준비-및-서버에서-재사용)).

## 데이터

- 출처: 상류 TFL이 만든 corpus pickle (`train_corpus.pk`, `val_corpus_bins.pk` 등). 이 서버에서는 `/workspace/tfl/data/{Parity,Tomita-5,Tomita-6}`에서 복사했다 ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 다운로드: 자동 다운로드 없음. `setup.sh`는 소스만 준비하며 corpus를 만들지 않는다. 생성법은 상류 문서를 따른다.
- 라이선스: 상류 코드는 MIT ([third_party/README](../../third_party/README.md)). corpus 데이터의 라이선스는 확인 불가.

## 알려진 한계

- bin마다 다른 epoch를 고를 수 있어 평균은 진단값이다. 하나의 체크포인트 점수나 최종 test가 아니다 (프로토콜 limitations).
- 고정된 상류 코드가 평균 validation 정확도 0.999 이상에서 조기 종료할 수 있다. epoch 수는 최대 예산이다 (프로토콜 limitations).
- 상류 LSTM이 GPU 길이 텐서를 `pack_padded_sequence`에 넘기는 문제가 있다. validation runner가 프로세스 안에서만 CPU 길이로 바꾼다 ([EVOSCIENTIST](../../docs/EVOSCIENTIST.md)).
- 공개 상류 코드에 없는 SARNN은 쓸 수 없다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#외부-구현을-쓰는-태스크)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/{parity,tomita5,tomita6}-<model>/`

## 아키텍처 추가 규칙

모델 등록은 `models.py`에서만 한다. 단, 이 태스크의 `models.py`는 상류 CLI 인자 변환표이므로 새 구조는
`third_party/tfl/src/components/`에 구현하고 `third_party/tfl.patch`로 기록해야 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#외부-구현을-쓰는-태스크)).
