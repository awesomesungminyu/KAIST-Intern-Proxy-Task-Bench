"""Parity adapter for the patched Transformer-Formal-Languages runner.

Consumes token indices [T,B] and emits sigmoid outputs [T,B,noutputs] for MSE training.
Prepare the upstream code with bash setup.sh parity.
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
