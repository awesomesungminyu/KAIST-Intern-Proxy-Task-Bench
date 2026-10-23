# mqar — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | Multi-query associative recall (zoology MQAR 생성기) |
| Tier (문헌 기대 라벨) | Tier 1 win |
| 유형 | 합성 데이터, 시퀀스 |
| 입력 → 출력 | 등록 단위는 **mixer** `[B,T,D] → [B,T,D]`, causal 필수. 공통 `TransformerLM`이 embedding·FFN·출력을 담당 |
| 시퀀스 길이 | 64 (`mqar/train.py`의 `CONFIG["sequence_length"]`) |
| vocab | 8192, KV 쌍 8개 |
| 지표 | `best_valid_accuracy`, **높을수록 좋음**. split `validation`, 선택 `highest_validation_accuracy_checkpoint` |
| 예산 | full 최대 30 epoch, validation 정확도 0.99 도달 시 조기 종료 / smoke `--smoke_test` (1 epoch) |
| 런타임 | `/data/envs/hexa-mad/bin/python` (torch 2.2.2, pydantic·wandb·einops), CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `linear_attention`, `gated_linear_attention`, `gqa2`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task mqar --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task mqar --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && /data/envs/hexa-mad/bin/python -m mqar.train --help`

## 데이터

- 출처: 입력 파일 없음. `third_party/zoology` 고정 커밋(`third_party/zoology.commit`)의 MQAR 생성기가 실행 시 만든다.
  train 20000 / valid 2000, 생성 seed 123 / 2147483789.
- 다운로드: `./setup.sh mqar`가 zoology를 `$PROXY_VENDOR_ROOT/zoology`에 클론한다. `--zoology-root`로 덮어쓸 수 있다.
- 라이선스: zoology 코드 Apache-2.0 ([third_party/README](../../third_party/README.md)). 생성 데이터는 해당 없음.

## 알려진 한계

- 별도 final test split이 없다 (프로토콜 limitations).
- 과거 harness_validation 파일럿 기록은 acceptance 결과가 아니다 (프로토콜 limitations, [experiments/README](../../experiments/README.md)).
- mixer는 causal이어야 하고 위치 정보를 내부에서 제공해야 한다. 양방향 encoder를 넣으면 정답 누출이다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/mqar-<model>/` (일부 모델은 `-retry2` 폴더가 함께 있다)

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
