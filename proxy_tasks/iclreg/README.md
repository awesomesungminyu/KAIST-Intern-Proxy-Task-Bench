# In-context linear regression (iclreg)

출처: Garg, Tsipras, Liang, Valiant, *What Can Transformers Learn In-Context? A Case Study of
Simple Function Classes*, NeurIPS 2022, arXiv:2208.01066. 자체 구현 축소판이다.

모델: transformer, lstm. 모델 등록·입출력 계약은 [모델 추가 가이드](../../docs/ADDING_A_MODEL.md#iclreg-연속-벡터-입력)를 따른다.
**입력이 토큰이 아니라 연속 벡터다.** 다른 태스크의 모델을 그대로 꽂을 수 없다.

데이터: 입력 파일 없음. 시퀀스마다 w ~ N(0, I_20), x_i ~ N(0, I_20), y_i = w·x_i (잡음 없음).
프롬프트당 41점이므로 query 직전 예제 수 k = 0..40 을 한 forward 로 잰다.
학습 프롬프트는 매 step 새로 생성하고, validation 은 seed 2147483647 로 고정한 2048 프롬프트다.
별도 test split 은 없다.

지표: query 에서의 (ŷ − y)² / d 를 k 별로 기록한다. 프로토콜 지표 `val_k_ge_d_mse`는
k = 20..40 의 평균(낮을수록 좋음)이다. 같은 프롬프트에서 계산한 기준선 두 개를 history.json 에 함께 저장한다.

- `least_squares_by_k`: 최소 노름 최소제곱 (pinv, Garg et al. 과 같음). k ≥ d 에서 0.
- `zero_by_k`: y = 0 예측. 기대값 1.

예산: 45000 step, batch 64, Adam lr 1e-4, grad clip 1.0, 최종 step 평가.
Garg et al. curriculum 그대로: 차원 5→20 (+1), 점 11→41 (+2), 2000 step 마다. 30000 step 에 최대 크기에 도달한다.
transformer 는 GPT-2 형 6층/8헤드/폭 256 (논문 12층), lstm 은 3층/폭 512.

런타임: torch 2.5.1. 학습은 CUDA가 필수다. CPU로 자동 대체하지 않는다.

```bash
# 저장소 루트, .env를 export한 뒤 환경 점검 (학습 없음)
python -m validation check --task iclreg --model transformer
# 학습 권한이 있는 세션에서만 실행
python -m validation run --task iclreg --model transformer \
  --profile full --seeds 0 1 2 --out artifacts/iclreg-transformer
# 직접 CLI 도움말 (proxy_tasks/에서)
python -m iclreg.train --help
```

결과는 실행별 native/의 `<model>_seed<seed>/history.json`과 공통 result.json에 기록한다.

한계 (확인된 것만):

- mamba 는 등록하지 않았다. 순수 torch 스캔이 이 크기에서 약 112 ms/step 이라 45000 step 이 seed 당 30분을 넘는다.
- 학습 곡선은 curriculum 이 최대 크기에 도달한 뒤(약 30000 step)에야 k ≥ d 오차가 크게 떨어진다.
  2000 대신 500 step 간격 curriculum 으로는 6000 step 동안 오차가 y=0 수준(≈0.97)에 머물렀다 (파일럿, acceptance 아님).
- **Tier 1 win 미확정** (Transformer 우위 미검증). 45000 step, k ≥ d 오차 transformer seed 0/1/2 =
  0.084/0.796/0.476 (0.452 ± 0.291, 수렴 seed 1/3), lstm seed 0 = 0.249. lstm seed 1은 도중 중단, seed 2는 미실행.
  더 긴 학습은 시도하지 않았다. 코드와 프로토콜 등록은 유지한다.
  자세한 수치: [Tier 프레임워크](../../docs/TIER_FRAMEWORK.md#tier-1-win-신규-편입-판정-2026-10-22).
