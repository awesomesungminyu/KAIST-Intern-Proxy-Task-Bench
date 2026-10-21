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
| Tier 1 win | `mqar`, `copy`, `iclreg` | **미구현** |
| Tier 2 win | `regbench`, `g2p`, `zinc` | 구현됨 |
| Tier 2 lose | `sc_raw`, `scifar`, `ett` | **미구현** |
| Tier 3 | 대규모 AR LM | 이 리포 범위 밖 (ETRI) |

## 미해결 문제

### (1) 공통 아키텍처 집합이 없다 — 최우선 과제

`validation/protocols/v1.json` 기준 태스크별 모델은 다음과 같다.

| 태스크 | 모델 |
|---|---|
| regbench, g2p, fflm | transformer, lstm |
| parity, tomita5, tomita6 | san_abs, san_mask, lstm |
| zinc | gcn, gine, gps |
| tier0 | 시퀀스 믹서 11종 (transformer, swa, gla, rwkv6, based, s4d, mamba, h3, hyena, lstm, cnn) |

네 그룹 모두에 들어 있는 아키텍처가 없다. 이름이 같은 `transformer`도 구현이 다르다.

| | regbench `TinyGPT` | tier0 `transformer` |
|---|---|---|
| 층 / 헤드 | 8층 / 2헤드 | 4층 / 4헤드 |
| 정규화 | LayerNorm | RMSNorm |
| MLP | GELU | SwiGLU |
| 위치 정보 | 학습형 위치임베딩 | 없음 |

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

`proxy_tasks/tier0/tier0_proxy.py`는 torch만으로 11종 믹서를 init 상태에서
스코어링한다. 학습은 하지 않는다.

```bash
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
