# fflm — 태스크 카드

기존 [README](README.md)를 대체하지 않는 요약이다. 코드, `validation/protocols/v1.json`, 기존 문서에서
확인한 내용만 적었다. 확인하지 못한 항목은 "확인 불가"로 표기한다. 전체 목록은 [TASK_INDEX](../../docs/TASK_INDEX.md).

| 항목 | 내용 |
|---|---|
| 한 줄 설명 | Flip-Flop Language Modeling 재구성 (Liu et al. 2023, deterministic clean 과제) |
| Tier (문헌 기대 라벨) | Tier 1 lose |
| 유형 | 합성 데이터, 시퀀스 (causal LM) |
| 입력 → 출력 | `tokens[B,T]` → `logits[B,T,5]`. 토큰 w/r/i/0/1 |
| 시퀀스 길이 | 학습 T = 512 (`run_fflm.py`의 `fresh_batch(length=512)`) |
| vocab | 5 |
| 지표 | `id_read_error_rate`, **낮을수록 좋음**. split `id_development_prefix`, 선택 `final_step_no_ood_selection` |
| 예산 | full: lstm 500 step / transformer 10000 step, data seed 1000 / smoke `--smoke --batch 2 --eval-batch 2` |
| 런타임 | torch 2.5.1, CUDA 필수 |

## 모델

등록 ID (`models.py` 레지스트리 = 프로토콜 `models`): `transformer`, `lstm`
기준 모델: `transformer`

## 실행

저장소 루트에서 실행한다. `.env`를 export한 뒤 사용한다 ([README](../../README.md#실행-인터페이스)).

```bash
python -m validation list
python -m validation check --task fflm --model transformer
# 학습 권한과 GPU 예산이 있는 세션에서만
python -m validation run --task fflm --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/<새 폴더>
```

직접 실행기 도움말: `cd proxy_tasks && python -m fflm.run_fflm --help`. 직접 실행기의 기본값은 ID/sparse/dense 전체를 평가하므로
구조 탐색에는 `python -m validation`을 쓴다 ([EVOSCIENTIST](../../docs/EVOSCIENTIST.md)).

## 데이터

- 출처: `package_manifest.json`으로 고정한 data/vendor 파일. 평가 배열은 공식 데이터셋 revision `91ef5c91274612fd3fe3ac18da5c7c820a39378e`의 prefix다 ([README](README.md)).
- 다운로드: 자동 다운로드 도구 없음. 검증된 패키지에서 복원한다:
  `python scripts/restore_fflm_assets.py --source <E3-FFLM 패키지 경로>` ([EVOSCIENTIST](../../docs/EVOSCIENTIST.md#자산-준비-및-서버에서-재사용)).
- 라이선스: 데이터 라이선스는 확인 불가. vendor 패키지의 LICENSE 파일은 manifest에 포함되어 있다.

## 알려진 한계

- 저자 원 학습 구현이 아닌 재구성이다 (프로토콜 limitations).
- lstm 500 step, transformer 10000 step으로 예산이 달라 두 구조의 교차 비교를 `compare`가 거부한다. 지금은 순위를 낼 수 없다 ([TIER_FRAMEWORK](../../docs/TIER_FRAMEWORK.md#3-fflm은-지금-순위를-낼-수-없다)).
- 반복 모니터링한 ID prefix만 점수화한다. sparse/dense OOD는 후보 확정 뒤에만 평가한다. 오류 0건이 정답성을 보장하지 않는다 (프로토콜 limitations).
- 결과 위치: `artifacts/final_20261022/full/` 아래에 fflm 디렉터리가 없다 (2026-10-23 목록 확인). smoke 기록은 `artifacts/acceptance-final-20261021/fflm/` ([VERIFICATION_20261021](../../docs/VERIFICATION_20261021.md)).

## 아키텍처 추가 규칙

아키텍처 추가·교체는 `models.py`에서만 한다. 새 이름을 추가하면 프로토콜 `steps`에도 예산을 기록해야 한다 ([ADDING_A_MODEL](../../docs/ADDING_A_MODEL.md#그대로-연결할-수-없는-경우)).
