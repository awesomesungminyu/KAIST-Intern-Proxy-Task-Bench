# jelassi_copy — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | 문자열 전체 복사와 학습 길이 밖으로의 일반화 (Jelassi et al. 2024, 자체 재구현) |
| Tier (문헌 기대 라벨) | Tier 1 win (저장소 판정 기록은 [TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#tier-1-win-신규-편입-판정-2026-10-22)) |
| 유형 | 생성 데이터, 시퀀스 (causal LM) |
| 입력 → 출력 | `tokens[B,T]` → `logits[B,T,30]`, causal |
| 시퀀스 길이 | 학습 context 220 토큰 (문자열 길이 5..50을 이어 붙임). 평가 구간 id 5..50, ood 51..100 / 101..200 / 201..300, 최대 603 토큰 |
| vocab | 30 (a..z 26자 + `$`, `\|`, `.`, `*`) |
| 지표 | `ood_exact_match` (OOD 세 구간의 문자열 exact match 평균), **높을수록 좋음**. split `fixed_generated_validation`, 선택 `final_step` |
| 예산 | full 4000 step, batch 32, AdamW lr 3e-4 wd 0.1 / smoke `--smoke_test` (20 step) |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer` (Hard-ALiBi), `lstm`, `mamba` (순수 torch)
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task jelassi_copy --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task jelassi_copy --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 trainer 도움말: `cd proxy_tasks && python -m jelassi_copy.train --help`

## 데이터

- 출처: 입력 파일 없음. 실행 시 생성한다. 평가 문자열은 seed 2147483647로 구간당 256개를 고정한다.
- 형식 참고: 공식 코드 `sjelassi/transformers_ssm_copy` 커밋 `1188c47` (vendor하지 않음, [README](README.md#vendor-하지-않은-이유)).
- 라이선스: 해당 없음 (생성 데이터). 공식 코드의 라이선스는 확인 불가.

## 알려진 한계

- 공식 코드를 vendor하지 않고 재구현했다. 공식 구현과의 수치 동일성은 검증하지 않았다 ([README](README.md)).
- `transformer`는 Hard-ALiBi 변형이며 다른 태스크의 `transformer`와 구현이 다르다.
- `mamba`는 mamba-ssm 커널이 아닌 순수 torch 순차 스캔이다.
- 모델 크기와 학습량이 논문보다 훨씬 작다. 논문 수치의 정량 재현이 아니다.
- 태스크 ID가 `copy`가 아닌 이유: `proxy_tasks/copy/`가 표준 라이브러리 `copy`를 가린다 (프로토콜 limitations).
- 결과 위치 (Git 제외, 수치는 여기 옮기지 않음): `artifacts/tier1_new_20261022/full/jelassi_copy-<model>-s<seed>/`

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 평가 길이가 학습 context보다 길므로 위치 정보는 모델 내부에서 처리한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#iclreg-연속-벡터-입력)).
