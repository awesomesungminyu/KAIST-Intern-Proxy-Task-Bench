"""In-context linear regression 데이터와 기준선 (Garg et al. 2022, arXiv:2208.01066).

시퀀스마다 w ~ N(0, I_d) 를 새로 뽑고 x_i ~ N(0, I_d), y_i = w·x_i (잡음 없음).
점 n_points 개를 한 프롬프트로 두면, 위치 i 의 예측은 앞선 예제 k=i 개를 본 query 예측이다.
따라서 한 번의 forward 로 k=0..n_points-1 을 모두 잰다.

오차는 (ŷ - y)^2 / d 로 정규화한다 (y 의 분산이 d 이므로 y=0 예측의 기대 오차가 1).
기준선: least-squares oracle (k<d 는 최소 노름 해 = pinv, Garg et al. 과 같음), 평균 예측 y=0.
파일이 필요 없다. 모든 데이터는 실행 시 생성한다.
"""
from __future__ import annotations

import torch

# validation prompt 고정 seed. 학습 seed 와 겹치지 않게 큰 값을 쓴다 (모든 모델·seed 공통).
EVAL_SEED = 2_147_483_647


def sample(batch, n_points, n_dims, generator, device, active_dims=None):
    """xs[B,K,d], ys[B,K]. active_dims<d 이면 뒤쪽 좌표를 0 으로 둔다 (Garg et al. 차원 curriculum)."""
    xs = torch.randn(batch, n_points, n_dims, generator=generator, device=device)
    if active_dims is not None and active_dims < n_dims:
        xs[..., active_dims:] = 0
    w = torch.randn(batch, n_dims, 1, generator=generator, device=device)
    return xs, (xs @ w)[..., 0]


def eval_set(n_prompts, n_points, n_dims):
    g = torch.Generator().manual_seed(EVAL_SEED)
    return sample(n_prompts, n_points, n_dims, g, "cpu")


def normalized_error_by_k(pred, ys, n_dims):
    """위치(=문맥 예제 수 k)별 평균 (ŷ-y)^2/d. 반환 길이 K."""
    return ((pred - ys) ** 2).mean(0) / n_dims


@torch.no_grad()
def least_squares_predictions(xs, ys):
    """위치 i 에서 앞선 i 개 예제의 최소 노름 최소제곱 해로 y_i 를 예측. i=0 은 0."""
    xs64, ys64 = xs.double(), ys.double()
    pred = torch.zeros_like(ys64)
    for i in range(1, xs.shape[1]):
        w_hat = torch.linalg.pinv(xs64[:, :i]) @ ys64[:, :i, None]   # [B,d,1]
        pred[:, i] = (xs64[:, i:i + 1] @ w_hat)[:, 0, 0]
    return pred.float()


def baselines(xs, ys, n_dims):
    return {
        "least_squares_by_k": normalized_error_by_k(least_squares_predictions(xs, ys), ys, n_dims).tolist(),
        "zero_by_k": normalized_error_by_k(torch.zeros_like(ys), ys, n_dims).tolist(),
    }
