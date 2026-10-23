"""G2P encoder-decoder registry.

Builder(src_vocab, tgt_vocab, max_len) returns nn.Module:
forward(src[B,S], tgt_in[B,T]) -> logits[B,T,tgt_vocab]; num_params() -> int.
PAD/BOS/EOS = 0/1/2. The trainer owns greedy decoding and mode restoration.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import ModelRegistry

MODELS = ModelRegistry("g2p")
register_model = MODELS.register

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerG2P(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 40,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_pad_mask = src == PAD_ID
        tgt_pad_mask = tgt_in == PAD_ID
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )
        s = self.pos_enc(self.src_emb(src))
        t = self.pos_enc(self.tgt_emb(tgt_in))
        out = self.transformer(
            s, t,
            tgt_mask=tgt_causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.out(out)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LuongAttention(nn.Module):
    """Dot-product attention (Luong et al. 2015 스타일)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, dec_h: torch.Tensor, enc_out: torch.Tensor, src_pad_mask: torch.Tensor):
        # dec_h: (B, T_dec, D), enc_out: (B, T_src, D), src_pad_mask: (B, T_src) True=pad
        scores = torch.bmm(dec_h, enc_out.transpose(1, 2))  # (B, T_dec, T_src)
        scores = scores.masked_fill(src_pad_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, enc_out)  # (B, T_dec, D)
        out = torch.tanh(self.combine(torch.cat([dec_h, ctx], dim=-1)))
        return out


class LSTMG2P(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD_ID)
        self.encoder = nn.LSTM(
            d_model, d_model // 2, num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            d_model, d_model, num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = LuongAttention(d_model)
        self.out = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model
        self.n_layers = n_layers

    @staticmethod
    def _combine_directions(state: torch.Tensor, n_layers: int) -> torch.Tensor:
        """(n_layers*2, B, H) -> (n_layers, B, 2H)로 양방향을 concat.

        버그 수정: 이전에는 방향별 hidden(각 크기 H=d_model//2)을 sum()으로 합쳐서
        크기가 H로 줄어들었고, decoder(hidden_size=d_model)와 불일치해
        `RuntimeError: Expected hidden[0] size (..., 128), got [..., 64]`가 발생했다.
        방향은 sum이 아니라 concat해야 d_model(=2H) 크기가 맞는다.
        """
        num_directions_x_layers, batch, hidden = state.size()
        state = state.view(n_layers, 2, batch, hidden)
        state = state.transpose(1, 2).contiguous().view(n_layers, batch, 2 * hidden)
        return state

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_pad_mask = src == PAD_ID
        enc_out, (h, c) = self.encoder(self.src_emb(src))
        # bidirectional -> 방향별 hidden(H=d_model//2)을 concat해 d_model 크기로 복원
        h = self._combine_directions(h, self.n_layers)
        c = self._combine_directions(c, self.n_layers)
        dec_out, _ = self.decoder(self.tgt_emb(tgt_in), (h.contiguous(), c.contiguous()))
        dec_out = self.attn(dec_out, enc_out, src_pad_mask)
        return self.out(dec_out)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@register_model("transformer")
def _transformer(src_vocab: int, tgt_vocab: int, max_len: int) -> nn.Module:
    return TransformerG2P(src_vocab, tgt_vocab, max_len=max_len)


@register_model("lstm")
def _lstm(src_vocab: int, tgt_vocab: int, max_len: int) -> nn.Module:
    return LSTMG2P(src_vocab, tgt_vocab)


def build_model(name: str, src_vocab: int, tgt_vocab: int, max_len: int) -> nn.Module:
    return MODELS.build(name, src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=max_len)
