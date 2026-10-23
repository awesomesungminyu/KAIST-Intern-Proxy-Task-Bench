# regbench — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | 랜덤 DFA→PFA로 생성한 정규언어 예시 문자열의 in-context next-token 예측 (RegBench 계열) |
| Tier (문헌 기대 라벨) | Tier 2 win |
| 유형 | 생성 데이터, 시퀀스 (causal LM) |
| 입력 → 출력 | `tokens[B,T]` int64 → `logits[B,T,V]`, causal. `num_params()` 제공 |
| 시퀀스 길이 | 최대 256 (`--max_total_len 256`), 시퀀스당 예시 20개 (`--examples_per_seq 20`) |
| vocab | 22 = repo preset 심볼 20 + 특수 토큰 2 (PAD, SEP) (`regbench/pfa.py`의 `repo_default`, `vocab_size_with_specials`) |
| 지표 | `next_token_acc`, **높을수록 좋음**. split `reused_test_stream`, 선택 `final_step` |
| 예산 | full 2000 step, batch 64, lr 3e-4 / smoke `--smoke_test` (20 step) |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task regbench --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task regbench --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m regbench.train --help`

## 데이터

- 출처: 파일 없음. `regbench/pfa.py`가 실행 시 생성한다. 다운로드 없음.
- 프리셋: `repo`(공식 구현 berlino/seq_icl 기본값 계열)와 `paper`(Akyürek et al. 2024 계열) 두 벌이 `pfa.py`에 있고, 프로토콜은 `repo`를 쓴다.
- 라이선스: 해당 없음 (생성 데이터).

## 알려진 한계

- test stream을 학습 중 반복 평가하므로 개발용 지표다 (프로토콜 limitations).
- repo PFA 프리셋이며 RegBench 논문의 정확한 재현이 아니다 (프로토콜 limitations).
- 논문과 후속 논문의 하이퍼파라미터가 서로 다르다 ([proxy_tasks/README](../README.md#검증되지-않은-부분-직접-확인-필요)).
- `transformer`(`TinyGPT`)는 tier0 `transformer`와 구현이 다르다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md)).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/final_20261022/full/seed{0,1,2}/regbench-<model>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md)).
