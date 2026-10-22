# Tier 프레임워크

싼 실험으로 비싼 실험의 결과를 미리 가늠하기 위한 4계층 구조다.
이 문서는 정의, 핵심 주장, 현재 구현 현황, 그리고 아직 풀리지 않은 문제를 적는다.
**미해결 문제 절을 먼저 읽을 것.** 현재 상태로는 Tier 간 비교를 계산할 수 없다.

## 4계층 정의

| Tier | 이름 | 내용 | 비용 |
|---|---|---|---|
| 0 | Zero-Cost Proxy | 학습 0회. init 상태의 신호(기울기·활성 패턴·야코비안) | 초 단위 |
| 1 | Synthetic Unit Test | 합성 단위 테스트. 특정 능력 하나를 격리해 측정 | 분 단위 |
| 2 | Small-Scale | 축소한 실제 말뭉치·데이터셋 | 시간 단위 |
| 3 | Full-Scale | 대규모 autoregressive LM. ETRI 담당이며 이 리포 범위 밖 | 일 단위 이상 |

## 핵심 주장

인접 Tier 간에 아키텍처 순위의 Spearman ρ가 높으면, 싼 Tier로 비싼 Tier의
순위를 예측할 수 있다.

**이 주장은 아직 어느 Tier 쌍에서도 측정하지 않았다.** 지금까지 계산한 ρ는
Tier 0 안에서 지표끼리 비교한 값뿐이고, Tier 간 예측력과는 무관하다.

## 구현 현황

"win"은 대상 아키텍처가 이길 것으로 예상되는 태스크, "lose"는 질 것으로 예상되는
태스크다. 두 방향을 모두 갖춰야 변별력을 볼 수 있다.

| 구분 | 태스크 | 상태 |
|---|---|---|
| Tier 0 | `tier0` (`proxy_tasks/tier0/tier0_proxy.py`) | 구현됨, 한계 있음 (아래) |
| Tier 1 lose | `parity`, `tomita5`, `tomita6`, `fflm` | 구현됨 |
| Tier 1 win | `mqar`, `mad_tsc` (MAD selective copying) | 구현됨, 런타임·파일럿 한계 있음 |
| Tier 1 win | `jelassi_copy` (Jelassi et al. 2024 copy) | 구현됨, **합격** (3 seed, 아래 판정 절. 한계 있음) |
| Tier 1 win 후보 | `iclreg` (Garg et al. 2022) | 구현·등록됨, **Tier 1 win 미확정** (Transformer 우위 미검증, 아래 판정 절) |
| Tier 2 win | `regbench`, `g2p`, `zinc` | 구현됨 |
| Tier 2 lose | `sc_raw`, `scifar`, `ett` | 구현됨, budget_limited/미수렴 이력은 프로토콜에 명시 |
| Tier 3 | 대규모 AR LM | 이 리포 범위 밖 (ETRI) |

## 미해결 문제

### (1) 공통 아키텍처 집합이 없다 — ρ 측정 시점까지 보류

현재 편입 목표는 EvoScientist가 제안한 모델을 태스크별로 등록·실행할 수 있게 하는 것이다.
공통 아키텍처 집합 구성과 Tier 간 Spearman은 **ρ 측정 시점까지 보류**한다.
아래 문제를 해결했다는 뜻은 아니며, 다음 비교 단계에서 다시 다뤄야 한다.

`validation/protocols/v1.json` 기준 태스크별 모델은 다음과 같다.

| 태스크 | 모델 |
|---|---|
| regbench, g2p, fflm | transformer, lstm |
| parity, tomita5, tomita6 | san_abs, san_mask, lstm |
| zinc | gcn, gine, gps |
| ett | linear, transformer, naive |
| scifar, sc_raw | transformer, lstm |
| mqar | transformer, linear_attention, gated_linear_attention, gqa2 |
| mad_tsc | MAD 레시피 16종 (models.py); 과거 official benchmark는 2종 중 1종 FAILED |
| jelassi_copy | transformer (Hard-ALiBi), lstm, mamba (순수 torch) |
| iclreg | transformer, lstm (연속 벡터 입력) |
| tier0 | 시퀀스 믹서 11종 (transformer, swa, gla, rwkv6, based, s4d, mamba, h3, hyena, lstm, cnn) |

모든 태스크에 공통으로 들어 있는 아키텍처가 없다. 이름이 같은 `transformer`도 구현이 다르다.

| | regbench `TinyGPT` | tier0 `transformer` |
|---|---|---|
| 층 / 헤드 | 8층 / 2헤드 | 4층 / 4헤드 |
| 정규화 | LayerNorm | RMSNorm |
| MLP | GELU | SwiGLU |
| 위치 정보 | 학습형 위치임베딩 | 없음 |

jelassi_copy의 `transformer`는 위치 인코딩 없는 Hard-ALiBi 변형(4층/8헤드/폭 256)이라 위 두 구현과도 다르다.

이 상태로는 Tier 간 Spearman을 계산할 수 없다. 같은 이름, 같은 구현의 아키텍처
집합을 먼저 정해 모든 Tier에 꽂아야 한다.

### (2) zinc는 그래프 모델이라 시퀀스 믹서를 돌릴 수 없다

처리 방향을 정하지 않았다. 후보는 세 가지다.

- SMILES로 시퀀스화한다.
- 다른 Tier 2 태스크로 교체한다.
- 그래프 트랙을 따로 분리한다.

### (3) fflm은 지금 순위를 낼 수 없다

full 예산이 lstm 500 step, transformer 10000 step으로 다르다. 그래서
`python -m validation compare`가 두 구조의 교차 비교를 예산 불일치로 거부한다.

### (4) g2p는 encoder-decoder라 causal mixer를 그대로 꽂을 수 없다

`TransformerG2P`와 `LSTMG2P`(Luong attention) 모두 encoder-decoder 구조다.

### (5) Tier 2 lose 행은 Tier 3 예측 주장에서 제외한다

Tier 3은 언어 도메인 하나뿐이다. 비언어 태스크인 Tier 2 lose 행(sc_raw, scifar,
ett)은 Tier 1↔Tier 2 변별력 검증에만 쓰고, Tier 3 예측 주장의 근거로는 쓰지 않는다.

## 통계 주의

아키텍처 n=11에서 Spearman ρ의 유의 임계는 약 0.60(p<0.05, 양측 표값 0.618)이다.
그보다 낮은 값을 상관으로 해석하면 안 된다. 반대로 |ρ|>0.9는 두 지표가 같은 것을
재고 있다는 신호일 수 있다.

## Tier 0 하니스

`proxy_tasks/tier0/models.py`의 레지스트리와 `tier0_proxy.py`는 torch만으로 11종 믹서를 init 상태에서
스코어링한다. 학습은 하지 않는다.

```bash
# 공통 provenance가 필요한 실행 (학습/점수화 승인이 있는 세션)
python -m validation run --task tier0 --model transformer --profile full \
  --seeds 0 1 2 --out artifacts/tier0-transformer
# 직접 하니스 (기본 seed 1234부터)
cd proxy_tasks
python -m tier0.tier0_proxy --quick                                  # 스모크
python -m tier0.tier0_proxy --seeds 3 --seq-len 256 --probe-len 128  # 본 실행
```

`--out`을 주지 않으면 결과는 `$PROXY_OUTPUT_ROOT/tier0/<models>-s1234/tier0_results.json`에
저장된다(1234는 첫 seed).

Git 이력에는 두 버전이 있다.

- **v1** (커밋 `Add Tier 0 zero-cost proxy harness`): 처음 측정에 쓴 코드. 알고리즘은 바꾸지 않고
  확인된 한계만 docstring에 적었다.
- **v2** (그다음 커밋): v1의 한계를 고친 버전. 무엇을 고쳤고 무엇이 남았는지는
  파일 상단 docstring에 있다.

### 측정 기록

결과 JSON은 `artifacts/tier0/`에 있고 커밋하지 않는다(`/artifacts/` 무시).
모든 실행은 공통적으로 d_model 128, 4층, vocab 256이다(`--quick` 제외). seed는 1234부터 시작한다.

| 파일 | 코드 | 장치 | 설정 |
|---|---|---|---|
| `quick.json` | v1 | **CPU** | `--quick` (d_model 32, 2층, L 64, batch 4, seed 1개, probe_len 8) |
| `main_b8.json` | v1 | **CPU** | seeds 3, L 256, probe_len 24, batch 8 |
| `main_b16.json` | v1 | **CPU** | 위와 같고 batch 16 |
| `main_b32.json` | v1 | **CPU** | 위와 같고 batch 32 |
| `quick_v2.json` | v2 | TITAN RTX | `--quick` |
| `v2_b8.json` | v2 | TITAN RTX | seeds 3, L 256, probe_len 128, batch 8 |
| `v2_b32.json` | v2 | TITAN RTX | 위와 같고 batch 32 |

**v1 결과는 GPU가 아니라 CPU에서 나왔다.** 실행 당시 서버의 CUDA 드라이버 초기화가
실패했고, 스크립트가 기본 장치로 CPU를 골랐다(각 JSON의 `env.device`, 로그 첫 줄 경고).
같은 v1 코드를 GPU에서 돌리면 값이 소수 셋째 자리부터 달라진다.

**v1과 v2의 수치를 직접 비교하지 않는다.** v1은 CPU, v2는 GPU에서 실행되어 코드 변경과
장치 변경이 교란되어 있다. v1 수치의 역할은 SynFlow·NASWOT이 사용 불가라는 구조적
근거를 뒷받침하는 것뿐이며, 그 근거는 장치와 무관하다.

### GPU에서만 드러난 LSTM cuDNN 실패

v1을 GPU에서 돌리면 `lstm`의 snip 이후 지표가 모두 실패한다
(`RuntimeError: cudnn RNN backward can only be called in training mode`).
하니스는 init 상태를 보려고 `model.eval()`로 두는데, cuDNN의 RNN 커널은 eval 모드
backward를 지원하지 않는다. CPU에서는 cuDNN을 쓰지 않으므로 v1 CPU 실행에서는 보이지
않았다. v2가 LSTM mixer forward 안에서만 cuDNN을 끄는(`torch.backends.cudnn.flags(enabled=False)`)
실질적 근거가 이것이다. v2 GPU 결과의 `lstm` 행은 cuDNN 없이 계산된 값이다.

### 지표 판단 (현재)

| 지표 | 판단 | 근거 |
|---|---|---|
| synflow | 폐기 | ones 입력이라 token mixing 차이가 안 보임. mamba/h3/hyena는 NaN |
| mlp_sal | 폐기 | 11종 모두 34.05로 상수 |
| snip, mix_nuc | 둘 중 하나 | 서로 ρ=0.82 |
| ntk_cond | 보류 | 다른 지표와 독립적이지만, 출력 좌표 설정에 대한 안정성을 확인하지 않음 |
| naswot | 조건부 | params·flops와의 ρ가 0.50/0.56으로 내려감. 4종은 여전히 구분 안 됨 |
| params, flops, state_cap | 대조군 | |

입력이 균일 랜덤 토큰이고 target이 입력과 같다는 데이터 설계 문제는 아직 정하지 않았다.

## Tier 1 win 신규 편입 판정 (2026-10-22)

두 과제를 새로 구현하고 학습을 돌려 "Transformer가 실제로 이기는가"를 확인했다. 실행은 모두
`python -m validation run --profile full`, NVIDIA TITAN RTX (sm_75, 각 train.log의 GPU-GUARD 줄로 확인),
torch 2.5.1+cu124, 소스 커밋 `fb2bc74`. 결과 파일은 `artifacts/tier1_new_20261022/`에 있고 커밋하지 않는다.
표기는 mean ± population std (ddof=0), seed 0/1/2. 각 seed는 별도 `validation run` 호출이다
(직전 실행 종료 직후 runner의 GPU idle 검사가 사용률 샘플 지연으로 실패해, 호출 전 10초 idle을 기다렸다).

### jelassi_copy — 합격

합격 기준: OOD 구간에서 transformer가 lstm/SSM보다 높고 차이가 seed 표준편차보다 클 것.
지표는 문자열 exact match, 4000 step, 학습 길이 5..50, 구간당 고정 문자열 256개.

| 모델 | 파라미터 | id 5..50 | ood 51..100 | ood 101..200 | ood 201..300 | **OOD 평균** | seed당 시간 |
|---|---|---|---|---|---|---|---|
| transformer | 3,174,912 | 0.999 ± 0.002 | 0.984 ± 0.003 | 0.945 ± 0.026 | 0.826 ± 0.070 | **0.918 ± 0.031** | 1.8분 |
| lstm | 4,233,246 | 0.172 ± 0.012 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** | 2.8분 |
| mamba | 1,767,680 | 0.378 ± 0.024 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | **0.000 ± 0.000** | 7.0분 |

차이 0.918은 transformer std 0.031의 약 30배다. 기준상 합격이다. 해석 한계:

- transformer는 Hard-ALiBi 변형(NoPE + 창 제한 헤드)이며 다른 태스크의 transformer와 구현이 다르다.
- 공식 코드를 vendor하지 않고 재구현했다. 데이터 형식·손실 mask·exact match 정의만 맞췄고,
  공식 구현과의 수치 동일성은 검증하지 않았다.
- lstm/mamba는 4000 step 예산에서 **학습 구간(id)도 풀지 못했다** (0.172, 0.378). OOD 격차의 일부는
  길이 일반화 실패가 아니라 학습 실패 때문이다. 학습 구간을 푼 SSM과의 OOD 비교는 하지 않았다.
- mamba는 파라미터가 transformer의 약 절반이고, mamba-ssm 커널이 아닌 순수 torch 순차 스캔이다.
- 3 seed, 논문보다 훨씬 작은 모델·학습량이다. 논문 수치의 정량 재현이 아니다.

### iclreg — Tier 1 win 미확정

합격 기준: k ≥ d 구간에서 transformer 오차가 lstm보다 seed 표준편차를 넘게 낮고 LS oracle에 접근할 것.
지표는 k = 20..40 평균 (ŷ−y)²/d, 낮을수록 좋다. LS oracle은 0, y=0 예측은 약 0.97.

| 모델 | seed 0 | seed 1 | seed 2 | 평균 |
|---|---|---|---|---|
| transformer | 0.084 | 0.796 | 0.476 | 0.452 ± 0.291 |
| lstm | 0.249 | 중단 | 미실행 | n=1 |

**Transformer 우위 미검증**: 45000 step에서 transformer가 수렴한 seed는 1/3(seed 0)이다. seed 1·2는
마지막 평가까지 오차가 내려가는 중이었다. transformer 평균이 lstm seed 0보다 높고 표준편차가 0.29여서
기준을 충족하지 못한다. lstm seed 1은 사용자 지시로 학습 도중 중단했고 seed 2는 돌리지 않았다.
더 긴 학습은 시도하지 않았다. 그래서 iclreg는 등록(코드·프로토콜)을 유지하되 Tier 1 win 행에 넣지 않는다.
대체 후보(induction heads, sparse regression 등)는 시도하지 않았다.

참고: transformer seed 0 k별 오차는 k=5 0.744 (LS 0.736), k=10 0.548 (0.522), k=19 0.193 (0.043),
k=20 0.187 (0), k=40 0.050 (0)으로, k<d 구간은 LS를 거의 따라가고 k ≥ d에서는 0에 다가가는 중이었다.
transformer의 seed당 시간은 14.8분, lstm seed 0은 18.3분이다.

## 신규 편입의 해석 범위

신규 태스크 코드가 있다는 것과 수렴·정량 재현을 검증했다는 것은 다르다.
sc_raw는 2048 update/epoch 0회 무작위 수준, scifar는 10 epoch 미수렴이었다.
MQAR는 4모델 × 3 seed 완료 이력이 있으나 linear/gated는 0.99 미달, GQA-2 분산이 컸다.
MAD official은 seed 12345 하나, attention 완료 / striped Hyena MoE 실패였으며 SDPA로 kernel_exact=false다.
ETT linear/transformer는 3 seed, naive는 1 seed 이력이다. 이 파일럿을 새 acceptance로 반입하지 않는다.
모델 추가 계약은 [ADDING_A_MODEL](ADDING_A_MODEL.md), 데이터 준비는 [인벤토리](DATA_INVENTORY.md)를 따른다.
