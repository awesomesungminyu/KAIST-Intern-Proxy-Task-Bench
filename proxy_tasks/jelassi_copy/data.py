"""문자열 전사(copy) 데이터 (Jelassi et al. 2024, arXiv:2402.01032).

형식은 공식 코드(github.com/sjelassi/transformers_ssm_copy, synthetic_exps/data_utils.py)를 따른다.
  예제 문자열: '$' + s + '|' + s + '.'   s 는 a..z 26자에서 균일 추출한 길이 L 문자열
  학습: 길이 L ~ U[min_len, max_len] 예제를 context_len 토큰까지 이어 붙인다 (마지막 예제는 잘림).
        손실 mask 는 구분자 '|' 이후의 전사 문자와 '.' 에만 1 (잘린 예제는 전부 0).
  평가: 예제 하나를 위치 0 에 두고 '*' 로 오른쪽 padding. causal 모델이므로 padding 은 결과에 영향이 없다.
지표는 문자열 단위 exact match: 전사 문자 L 개의 teacher-forced argmax 가 모두 맞으면 1.
모든 위치가 맞을 때만 1 이므로 greedy autoregressive decoding 의 exact match 와 같다.
공식 코드와 같이 '.' 는 exact match 에 넣지 않는다.

파일이 필요 없다. 모든 데이터는 실행 시 생성한다.
"""
from __future__ import annotations

import numpy as np
import torch

LETTERS = 26
BOS, SEP, EOS, PAD = LETTERS, LETTERS + 1, LETTERS + 2, LETTERS + 3   # '$', '|', '.', '*'
VOCAB = LETTERS + 4
# validation 문자열 고정 seed. 모든 모델·seed 가 같은 문자열을 본다.
EVAL_SEED = 2_147_483_647


def example(rng, length):
    s = rng.integers(0, LETTERS, size=length)
    return np.concatenate([[BOS], s, [SEP], s, [EOS]]), length


def train_batch(rng, batch, context_len, min_len, max_len):
    """tokens[B,context_len] int64, mask[B,context_len] float32 (mask[t]=1 이면 토큰 t 가 손실 대상)."""
    if context_len <= 2 * max_len + 3:
        raise ValueError(f"context_len {context_len} cannot hold a length-{max_len} example")
    tokens = np.empty((batch, context_len), dtype=np.int64)
    mask = np.zeros((batch, context_len), dtype=np.float32)
    for b in range(batch):
        pos = 0
        while pos < context_len:
            ids, length = example(rng, int(rng.integers(min_len, max_len + 1)))
            take = min(len(ids), context_len - pos)
            tokens[b, pos:pos + take] = ids[:take]
            if take == len(ids):
                mask[b, pos + length + 2:pos + take] = 1
            pos += take
    return torch.from_numpy(tokens), torch.from_numpy(mask)


def eval_strings(bins, per_bin):
    """구간별 고정 문자열. bins: [(name, lo, hi), ...] (양끝 포함)."""
    rng = np.random.default_rng(EVAL_SEED)
    return {name: [example(rng, int(rng.integers(lo, hi + 1)))[0] for _ in range(per_bin)] for name, lo, hi in bins}


def pad_batch(examples):
    width = max(len(e) for e in examples)
    tokens = np.full((len(examples), width), PAD, dtype=np.int64)
    for i, e in enumerate(examples):
        tokens[i, :len(e)] = e
    return torch.from_numpy(tokens)


def score(logits, tokens, lengths):
    """teacher-forced exact match 와 문자 정확도. logits[t] 는 tokens[t+1] 예측."""
    pred = logits.argmax(-1)
    exact, chars = [], []
    for i, L in enumerate(lengths):
        # 전사 문자는 tokens[L+2 : 2L+2], 그 예측은 logits[L+1 : 2L+1]
        hit = (pred[i, L + 1:2 * L + 1] == tokens[i, L + 2:2 * L + 2]).float()
        exact.append(float(hit.all()))
        chars.append(hit.mean().item())
    return exact, chars
