"""FFLM 모델 등록 지점.

입력: (B, T) int64 토큰 (0..4: w, r, i, 0, 1). 출력: (B, T, 5) 다음 토큰 logits.
생성 함수 인자: smoke(bool, 축소 크기), flash(bool, FlashAttention 사용 여부).
학습 step 예산은 run_fflm.py 가 정한다 (lstm 500, 그 밖의 이름은 10000).
"""
from torch import nn

from common.registry import ModelRegistry
from common_set import NAMES as COMMON_SET, TokenLM

MODELS = ModelRegistry("fflm")
register_model = MODELS.register


class LSTM(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(5, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 5)

    def forward(self, x):
        y, _ = self.rnn(self.embed(x))
        return self.head(y)


@register_model('lstm')
def _lstm(smoke=False, flash=True):
    return LSTM(32 if smoke else 128)


@register_model('transformer')
def _transformer(smoke=False, flash=True):
    from x_transformers import TransformerWrapper, Decoder
    dim, depth, heads = (32, 2, 4) if smoke else (512, 6, 8)
    return TransformerWrapper(
        num_tokens=5, max_seq_len=512, emb_dropout=0., use_abs_pos_emb=True,
        tie_embedding=False,
        attn_layers=Decoder(dim=dim, depth=depth, heads=heads,
            attn_dim_head=dim//heads, ff_mult=4, ff_glu=False,
            ff_dropout=0., attn_dropout=0., pre_norm=True,
            rotary_pos_emb=False, attn_flash=flash))


def _common_set(mixer):
    # Protocol v2 common architecture set: the transformer's width/depth/heads, mixer-only replacement.
    # flash is ignored: attention mixers always use torch SDPA. Every cs_ name gets the 10000-step budget.
    def build(smoke=False, flash=True):
        dim, depth, heads = (32, 2, 4) if smoke else (512, 6, 8)
        return TokenLM(5, mixer, d_model=dim, n_layers=depth, n_heads=heads, l_max=512)
    return build


for _mixer in COMMON_SET:
    register_model('cs_' + _mixer)(_common_set(_mixer))


def model_for(arch, smoke=False, flash=True):
    return MODELS.build(arch, smoke=smoke, flash=flash)
