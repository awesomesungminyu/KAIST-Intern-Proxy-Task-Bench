# Tier 0 v2

학습 없이 init 모델의 snip, mix_nuc, mlp_sal, naswot, ntk_cond와 대조군
params/flops/state_cap을 계산한다. 11종 mixer는 models.py에 있고, 새 등록에는
family와 state_elements(cfg)가 필요하다. [계약과 예제](../../docs/ADDING_A_MODEL.md).

```bash
python -m validation check --task tier0 --model transformer
# 점수화를 승인한 세션에서만 실행 (학습은 없음)
python -m validation run --task tier0 --model transformer --profile full \
  --seeds 0 1 2 --out artifacts/tier0-transformer
```

validation full: FP32/CUDA, d128, 4 layers, L256, probe128, batch8.
외부 runner가 seed 3개를 반복하고 child는 seed당 한 번만 실행한다.
smoke는 --quick: d32, 2 layers, L64, probe8, batch4, seed 하나당 한 번.
직접 CLI는 `cd proxy_tasks && python -m tier0.tier0_proxy --help`.
기존 직접 CLI seed 기본값은 1234부터 3개다. --seed-start로 바꿀 수 있다.

compare는 지표별 mean/population std만 제공한다. 단일 개선량이나 과제 성능을
추론하지 않는다. 비정상 값은 실패로 기록한다. 입력 데이터 파일은 없다.

v2 한계: mlp_sal 상수, snip↔mix_nuc 과거 rho 0.82, ntk_cond 좌표 안정성 미확인,
mix_nuc는 원 DSS++가 아님, 균일 랜덤 토큰으로 과제 의존성 없음.
flops/state_cap은 분석식이며 state_cap은 기존 v2의 2 bytes/element 가정이다.
smoke에서는 window64와 L64/probe8 때문에 sliding/full attention이 구분되지 않는다.
과거 v1 결과는 CPU, v2 결과만 TITAN RTX였다. Tier 간 rho는 아직 측정하지 않았다.
자세한 기록은 [Tier 프레임워크](../../docs/TIER_FRAMEWORK.md)를 따른다.
