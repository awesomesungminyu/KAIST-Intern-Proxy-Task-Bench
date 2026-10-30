"""Common skeleton: only the mixer differs between the common-set models of a task.

  Block     x + mixer(RMSNorm(x)); x + SwiGLU(RMSNorm(x))   (tier0 RMSNorm eps 1e-6 and SwiGLU, no bias)
  Backbone  n_layers x Block -> RMSNorm
  TokenLM   Embedding -> Backbone -> Linear head (untied, no bias): logits per position
  SequenceClassifier  Linear(features, d) -> Backbone -> mean over time -> Linear(d, classes)

Initialisation is PyTorch's default plus each mixer's own (mamba dt/A, s4d, hyena, gla gate bias).
d_model, n_layers and n_heads are the task's; they are the same for every mixer of the task.
retention_backend is passed to the retnet mixer only (mixers.make_mixer); it changes no parameters.

Parameter matching: a mixer keeps its official internal width (mixers.OFFICIAL_WIDTH) when its Block
has within PARAM_TOLERANCE of the transformer Block's parameters. Otherwise the width is the multiple
of 8 (8 .. 4 d_model) whose Block is closest. Embedding, head and final norm are identical across mixers,
so matching Blocks matches non-embedding parameters; tests check the whole-model count as well.
"""
import functools

import torch
from torch import nn

from .mixers import NAMES, OFFICIAL_WIDTH, make_mixer

PARAM_TOLERANCE = 0.10


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        h = int(d * mult * 2 / 3 / 8) * 8
        self.up = nn.Linear(d, h, bias=False)
        self.gate = nn.Linear(d, h, bias=False)
        self.act = nn.SiLU()
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, mixer):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mixer, self.mlp = mixer, SwiGLU(d)

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


def block_parameters(mixer, d_model, n_heads, l_max, width=None):
    with torch.device("meta"):
        block = Block(d_model, make_mixer(mixer, d_model, n_heads, l_max, width))
    return sum(p.numel() for p in block.parameters())


@functools.lru_cache(maxsize=None)
def mixer_width(mixer, d_model, n_heads, l_max):
    """Internal width used for `mixer` (None for mixers without one); see module docstring."""
    if mixer not in OFFICIAL_WIDTH:
        return None
    target = block_parameters("transformer", d_model, n_heads, l_max)
    official = OFFICIAL_WIDTH[mixer](d_model)
    if abs(block_parameters(mixer, d_model, n_heads, l_max, official) / target - 1) <= PARAM_TOLERANCE:
        return official
    return min(range(8, 4 * d_model + 1, 8),
               key=lambda w: abs(block_parameters(mixer, d_model, n_heads, l_max, w) - target))


class Backbone(nn.Module):
    def __init__(self, mixer, d_model, n_layers, n_heads, l_max, retention_backend="auto"):
        super().__init__()
        if mixer not in NAMES:
            raise ValueError(f"unknown common-set mixer: {mixer}; known: {NAMES}")
        width = mixer_width(mixer, d_model, n_heads, l_max)
        self.mixer, self.width = mixer, width
        self.blocks = nn.ModuleList([Block(d_model, make_mixer(mixer, d_model, n_heads, l_max, width, retention_backend))
                                     for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TokenLM(nn.Module):
    """tokens [B,T] int64 -> logits [B,T,V]."""

    def __init__(self, vocab_size, mixer, d_model, n_layers, n_heads, l_max, retention_backend="auto"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.backbone = Backbone(mixer, d_model, n_layers, n_heads, l_max, retention_backend)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        return self.head(self.backbone(self.embed(tokens)))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class SequenceClassifier(nn.Module):
    """x [B,T,features] float -> logits [B,classes]; every mixer is causal, pooling is a time mean."""

    def __init__(self, features, classes, mixer, d_model, n_layers, n_heads, l_max):
        super().__init__()
        self.read_in = nn.Linear(features, d_model)
        self.backbone = Backbone(mixer, d_model, n_layers, n_heads, l_max)
        self.head = nn.Linear(d_model, classes)

    def forward(self, x):
        return self.head(self.backbone(self.read_in(x)).mean(1))
