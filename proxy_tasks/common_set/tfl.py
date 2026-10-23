"""Transformer-Formal-Languages adapter (parity, tomita5, tomita6) for -model_type CS.

TFL feeds src [T,B] int64 (right-padded with 0) and trains outputs [T,B,noutputs] in [0,1] with MSE,
like its SAN. The TFL v2 patch (third_party/tfl-common-set.patch) imports this module as
proxy_tasks.common_set.tfl; proxy_tasks/formal_languages/runner.py puts the repository root on sys.path.
"""
import torch
from torch import nn

from .backbone import Backbone

# Longest formal-language input: validation bins reach length 100.
L_MAX = 128


class CommonSetTFLModel(nn.Module):
    def __init__(self, ntoken, noutputs, mixer, d_model, heads, depth, bias=True):
        super().__init__()
        self.encoder = nn.Embedding(ntoken, d_model)
        self.backbone = Backbone(mixer, d_model, depth, heads, L_MAX)
        self.decoder = nn.Linear(d_model, noutputs, bias=bias)

    def forward(self, src):
        h = self.backbone(self.encoder(src.transpose(0, 1)))
        return torch.sigmoid(self.decoder(h)).transpose(0, 1)
