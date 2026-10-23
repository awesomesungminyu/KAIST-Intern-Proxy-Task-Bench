"""sCIFAR 모델 등록 지점.

입력: (B, 1024, 3) float32. CIFAR-10 RGB 를 좌→우, 상→하로 펼친 1024 시점 3채널, train 부분집합 통계로 정규화.
출력: (B, 10) logits. 손실은 cross-entropy. 생성 함수는 인자를 받지 않는다.

원본: experiments/tier2_compare/tier2_compare.py 의 Classifier.
"""
import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.registry import ModelRegistry

MODELS = ModelRegistry("scifar")
register_model = MODELS.register


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


class Classifier(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        if arch == "transformer":
            self.mixer = Encoder(3, 1024, 2)
            self.head = nn.Linear(32, 10)
        else:
            self.mixer = nn.LSTM(3, 64, 1, batch_first=True)
            self.head = nn.Linear(64, 10)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if x.shape[1:] != (1024, 3):
            raise ValueError("Expected [B,1024,3]")
        z = self.mixer(x)
        if self.arch == "lstm":
            z = z[0]
        return self.head(self.drop(z.mean(1)))


@register_model("transformer")
def _transformer():
    return Classifier("transformer")


@register_model("lstm")
def _lstm():
    return Classifier("lstm")
