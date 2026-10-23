# String copying with length generalization (jelassi_copy)

출처: Jelassi, Brandfonbrener, Kakade, Malach, *Repeat After Me: Transformers are Better than
State Space Models at Copying*, ICML 2024, arXiv:2402.01032.
공식 코드 github.com/sjelassi/transformers_ssm_copy (마지막 커밋 `1188c4799b4f7654ac79349f61e8868b167f103e`, 2024-02-26).

**mad_tsc 와 다른 과제다.** mad_tsc 는 표시된 토큰만 골라 복사하고, 이 과제는 문자열 전체를
순서대로 전사하며 학습 길이 밖으로의 일반화를 잰다.

태스크 ID 가 `copy` 가 아닌 이유: `proxy_tasks/copy/` 패키지는 모든 태스크가 cwd 로 쓰는
`proxy_tasks/` 에서 표준 라이브러리 `copy` 를 가린다. 실제로 `import torch` 가
`cannot import name 'deepcopy' from 'copy'` 로 실패했다.

## vendor 하지 않은 이유

공식 코드를 확인한 결과 현재 서버에서 그대로 돌지 않는다.

- `synthetic_exps/model_utils.py` 가 최상단에서 `mamba_ssm` 을 무조건 import 한다.
  mamba-ssm 은 sm_75 에서 설치되지 않고 외부 패키지 추가도 금지다.
- `train_utils.py` 는 `accelerate` 가 필요하다 (torch 2.5.1 환경과 ai4science 환경 모두 미설치).
- 모델은 transformers 의 GPT-NeoX 내부 클래스를 상속한 사본이라 transformers 버전에 묶인다.

데이터 형식, 손실 mask, exact match 정의, 최적화 형태만 공식 코드를 따르고 나머지는 자체 구현했다.
따라서 third_party/ 에 추가한 것은 없다.

## 설정

데이터: 입력 파일 없음. 예제는 `$` + s + `|` + s + `.` (s 는 a..z 균일). 어휘 30 (26자 + `$|.*`).

- 학습: 길이 5..50 예제를 220 토큰 context 에 이어 붙인다 (공식 기본값). 손실은 `|` 이후
  전사 문자와 `.` 에만 건다. 잘린 마지막 예제는 손실에서 뺀다.
- 평가: seed 2147483647 로 고정한 구간별 256 문자열. 예제 하나를 위치 0 에 두고 padding 한다.
  `id` 5..50 (학습 구간), `ood_51_100`, `ood_101_200`, `ood_201_300` (최대 603 토큰).
- 지표: 문자열 단위 exact match. 프로토콜 지표 `ood_exact_match` 는 세 OOD 구간 평균이다.
  `id_exact_match` 와 구간별 문자 정확도는 history.json 에 있다. teacher-forced argmax 로 재지만
  모든 문자가 맞을 때만 1 이므로 greedy decoding 의 exact match 와 같다.
- 예산: 4000 step, batch 32, AdamW lr 3e-4 weight decay 0.1, warmup 100 + linear decay, clip 1.0. 최종 step 평가.

모델 (논문보다 훨씬 작다. 논문 기본은 12층/폭 1024):

| 이름 | 구조 | 파라미터 |
|---|---|---|
| transformer | Hard-ALiBi: NoPE, 4층/8헤드/폭 256, 앞 4개 헤드는 직전 1..4 토큰 창 | 3,174,912 |
| lstm | 2층/폭 512 | 4,233,246 |
| mamba | 순수 torch Mamba 4층/폭 256, d_state 16, expand 2 (MLP 없음) | 1,767,680 |

공식 Hard-ALiBi 구현은 q·k 에 rotary 미적용 성분을 한 번 더 이어 붙이는 코드가 있으나 여기서는 표준 q·k 를 쓴다.
공식 GPT-NeoX 는 parallel residual 이고 여기는 순차 pre-LN 이다. 공식 LSTM 의 dropout 0.65 는 쓰지 않았다.

런타임: torch 2.5.1. 학습은 CUDA가 필수다.

```bash
python -m validation check --task jelassi_copy --model transformer
python -m validation run --task jelassi_copy --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/jelassi_copy-transformer
cd proxy_tasks && python -m jelassi_copy.train --help
```

판정: **Tier 1 win 합격** (3 seed, full, TITAN RTX). OOD 평균 exact match
transformer 0.918 ± 0.031, lstm 0.000 ± 0.000, mamba 0.000 ± 0.000 (mean ± population std).
구간별 표는 [Tier 프레임워크](../../docs/TIER_FRAMEWORK.md#tier-1-win-신규-편입-판정-2026-10-22)에 있다.

한계 (확인된 것만):

- transformer는 Hard-ALiBi 변형(NoPE + 창 제한 헤드)이며 다른 태스크의 transformer와 구현이 다르다.
- 공식 코드를 vendor하지 않고 재구현했다. 데이터 형식·손실 mask·exact match 정의만 맞췄고,
  공식 구현과의 수치 동일성은 검증하지 않았다.
- lstm/mamba는 4000 step 예산에서 학습 구간(5..50)도 풀지 못했다 (id exact match 0.172 ± 0.012,
  0.378 ± 0.024). OOD 격차의 일부는 길이 일반화 실패가 아니라 학습 실패 때문이다.
- mamba는 파라미터가 transformer의 약 절반(1.77M vs 3.17M)이고, mamba-ssm 커널이 아닌 순수 torch
  순차 스캔이다 (순차 autograd 구현과 fp64에서 일치).
- 모델 크기와 학습량이 논문보다 훨씬 작다. 논문 수치의 정량 재현이 아니다.
- 프로토콜(v1.json)의 limitations 필드는 실행 전에 확정한 문장 그대로다. 프로토콜을 고치면 모든 태스크의
  evaluation_sha256이 바뀌므로, 측정 후 확인한 한계는 이 README와 Tier 프레임워크 문서에만 적었다.
