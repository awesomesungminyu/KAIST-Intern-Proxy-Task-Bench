# g2p — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | CMUDict 자소→음소 변환 (encoder-decoder seq2seq) |
| Tier (문헌 기대 라벨) | Tier 2 win |
| 유형 | 실제 데이터, 시퀀스 (encoder-decoder, causal mixer 아님) |
| 입력 → 출력 | `forward(src[B,S], tgt_in[B,T]) → logits[B,T,Vt]`. greedy decoding은 trainer가 수행 (PAD/BOS/EOS = 0/1/2) |
| 시퀀스 길이 | 최대 40 (`--max_len 40`) |
| vocab | 데이터에서 생성 (`g2p/prepare_data.py`의 `build_vocab`). 현재 서버 `vocab.json` 기준 문자 30, 음소 72 (특수 토큰 3개 포함) |
| 지표 | `val_word_acc`, **높을수록 좋음**. split `validation`, 선택 `final_epoch` |
| 예산 | full 10 epoch, batch 128, lr 1e-3 / smoke `--smoke_test` (1 epoch) |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task g2p --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task g2p --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m g2p.train --help`

## 데이터

- 출처: `https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict` (`g2p/prepare_data.py`의 `CMUDICT_URL`).
- 다운로드·split: `cd proxy_tasks && python -m g2p.prepare_data`. 단어 단위 결정론적 90/5/5 split (seed 0).
  필요한 파일은 `vocab.json`, `train.tsv`, `val.tsv`, `test.tsv` ([DATA_INVENTORY](../../docs/DATA_INVENTORY.md)).
- 라이선스: 확인 불가 (저장소에 기재 없음).

## 알려진 한계

- 논문 split이 아닌 CMUDict 재구성 split이다. trainer는 validation 정확도만 노출하며 최종 test 주장은 지원하지 않는다 (프로토콜 limitations).
- encoder-decoder라 범용 causal mixer를 그대로 꽂을 수 없다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#4-g2p는-encoder-decoder라-causal-mixer를-그대로-꽂을-수-없다)).
- v1에서 greedy decoding의 train/eval 모드 복원을 고쳤으므로 그 이전 G2P 결과는 v1 결과로 재사용하지 않는다 ([EVOSCIENTIST](../../docs/EVOSCIENTIST.md#검증-명령과-프로토콜-변경)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/g2p-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
