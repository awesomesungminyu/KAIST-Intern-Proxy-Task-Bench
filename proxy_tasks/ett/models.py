"""ETT 모델 등록 지점.

입력: (B, 96, 7) float32, train 구간으로 표준화된 과거 96 시점 7채널.
출력: (B, 96, 7) 다음 96 시점 예측. 손실은 MSE.
생성 함수는 인자를 받지 않는다. 학습할 파라미터가 없는 모델은 NO_TRAINING 에 이름을 넣는다.

원본: experiments/tier2_compare/tier2_compare.py. linear(DLinear)·transformer 는 약 18,624
파라미터로 맞춰져 있었다.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.registry import ModelRegistry

MODELS = ModelRegistry("ett")
register_model = MODELS.register
NO_TRAINING = {"naive"}


class Position(nn.Module):
    def __init__(self, length, width=32):
        super().__init__()
        pos = torch.arange(length).float()[:, None]
        w = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        enc = torch.empty(length, width)
        enc[:, 0::2] = torch.sin(pos * w)
        enc[:, 1::2] = torch.cos(pos * w)
        self.register_buffer("encoding", enc[None], persistent=True)

    def forward(self, x):
        return x + self.encoding[:, :x.shape[1]]


class Encoder(nn.Module):
    def __init__(self, features, length, layers):
        super().__init__()
        self.embed = nn.Linear(features, 32)
        self.pos = Position(length)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.1, activation="relu",
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x):
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return self.encoder(self.pos(self.embed(x)), is_causal=False)


class ForecastTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(7, 96, 1)
        self.project = nn.Linear(32, 7)
        self.time_head = nn.Linear(96, 96)

    def forward(self, x):
        return self.time_head(self.project(self.encoder(x)).transpose(1, 2)).transpose(1, 2)


class DLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.trend = nn.Linear(96, 96)
        self.remainder = nn.Linear(96, 96)

    def forward(self, x):
        z = x.transpose(1, 2)
        trend = F.avg_pool1d(F.pad(z, (12, 12), mode="replicate"), 25, stride=1)
        return (self.trend(trend) + self.remainder(z - trend)).transpose(1, 2)


class Naive(nn.Module):
    def forward(self, x):
        return x[:, -1:, :].expand(-1, 96, -1)


@register_model("linear")
def _linear():
    return DLinear()


@register_model("transformer")
def _transformer():
    return ForecastTransformer()


@register_model("naive")
def _naive():
    return Naive()
