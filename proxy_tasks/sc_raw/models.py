"""sc_raw 모델 등록 지점.

입력: (B, 16000, 1) float32. Speech Commands 1초 원본 파형(PCM16/32768, zero-pad/crop), 다운샘플링·패치화 없음.
출력: (B, 12) logits (10 단어 + unknown + silence). 손실은 cross-entropy. 생성 함수는 인자를 받지 않는다.
길이 16000 을 그대로 받으므로 메모리가 O(L^2) 인 구조는 batch 2 에서도 무겁다.

원본: experiments/tier2_followup30/tier2_raw_followup.py 의 RawClassifier.
"""
import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from common.registry import ModelRegistry

MODELS = ModelRegistry("sc_raw")
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


class RawClassifier(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.arch = arch
        self.mixer = Encoder(1, 16000, 1) if arch == "transformer" else nn.LSTM(1, 44, 1, batch_first=True)
        self.head = nn.Linear(32 if arch == "transformer" else 44, 12)

    def forward(self, x):
        assert x.shape[1:] == (16000, 1)
        z = self.mixer(x)
        return self.head((z[0] if self.arch == "lstm" else z).mean(1))


@register_model("transformer")
def _transformer():
    return RawClassifier("transformer")


@register_model("lstm")
def _lstm():
    return RawClassifier("lstm")
