"""MQAR 모델 등록 지점.

등록하는 것은 시퀀스 mixer 다. 생성 함수는 (d_model, num_heads) 를 받아 SequenceMixer 를 돌려준다.
mixer 계약: 입력 (B, T, D) float32 → 출력 (B, T, D). 반드시 causal 이어야 한다 (뒤 토큰을 보면 과제가 무의미).
위치 정보는 mixer 안에서 처리한다 (임베딩에 위치 인코딩이 없다).
나머지(임베딩, pre-norm LayerNorm, GELU FFN, weight tying, 초기화)는 모든 mixer 에 공통이다.

원본: experiments/mqar/train_mqar_{baseline,linear_attention,gated_linear_attention,gqa}.py.
mixer 소스는 원본에서 그대로 가져왔다 (GQA 의 num_kv_heads 만 CONFIG 대신 인자로 받음).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.registry import ModelRegistry
from common_set import NAMES as COMMON_SET, TokenLM, make_mixer, mixer_width

MODELS = ModelRegistry("mqar")
register_model = MODELS.register


class SequenceMixer(nn.Module):
    def forward(self, x):
        raise NotImplementedError


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10_000):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        length = q.size(2)
        positions = torch.arange(
            length, device=q.device, dtype=torch.float32
        )
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().to(q.dtype)[None, None]
        sin = angles.sin().to(q.dtype)[None, None]

        def rotate(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack(
                (
                    even * cos - odd * sin,
                    even * sin + odd * cos,
                ),
                dim=-1,
            ).flatten(-2)

        return rotate(q), rotate(k)


class CausalSelfAttention(SequenceMixer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        q, k = self.rope(q, k)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(
            batch, length, width
        )
        return self.output(output)


class CausalLinearAttention(SequenceMixer):
    """
    Causal linear attention using:

        phi(x) = ELU(x) + 1

    The cumulative K and KV states ensure that each position
    can only use its current and previous tokens.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        eps=1e-6,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )
        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # Position information remains inside the mixer.
        self.rope = RotaryEmbedding(self.head_dim)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q, k = self.rope(q, k)

        q = self.feature_map(q)
        k = self.feature_map(k)

        # Outer product K^T V at every position.
        kv = torch.einsum(
            "bhtd,bhte->bhtde",
            k,
            v,
        )

        # Causal prefix states.
        kv_state = kv.cumsum(dim=2)
        k_state = k.cumsum(dim=2)

        numerator = torch.einsum(
            "bhtd,bhtde->bhte",
            q,
            kv_state,
        )

        denominator = torch.einsum(
            "bhtd,bhtd->bht",
            q,
            k_state,
        ).unsqueeze(-1)

        mixed = numerator / denominator.clamp_min(
            self.eps
        )

        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                d_model,
            )
        )

        return self.output(mixed)


class GatedLinearAttention(SequenceMixer):
    """
    Pure PyTorch gated recurrent linear attention.

    A learned forget gate controls how much previous KV state
    remains at each position.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        eps=1e-6,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False,
        )
        self.output = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        # One data-dependent retention gate per head.
        self.forget_gate = nn.Linear(
            d_model,
            num_heads,
            bias=True,
        )

        # Start with a gate close to 1 so long-range
        # information is retained early in training.
        nn.init.zeros_(self.forget_gate.weight)
        nn.init.constant_(self.forget_gate.bias, 3.0)

        self.rope = RotaryEmbedding(self.head_dim)

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        batch_size, sequence_length, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        q, k = self.rope(q, k)
        q = self.feature_map(q)
        k = self.feature_map(k)

        gates = torch.sigmoid(
            self.forget_gate(x)
        )
        gates = gates.transpose(1, 2).unsqueeze(-1)

        kv_state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        k_state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )

        outputs = []

        for position in range(sequence_length):
            gate = gates[:, :, position]

            key = k[:, :, position]
            value = v[:, :, position]
            query = q[:, :, position]

            kv_update = torch.einsum(
                "bhd,bhe->bhde",
                key,
                value,
            )

            kv_state = (
                gate.unsqueeze(-1) * kv_state
                + kv_update
            )
            k_state = gate * k_state + key

            numerator = torch.einsum(
                "bhd,bhde->bhe",
                query,
                kv_state,
            )

            denominator = torch.einsum(
                "bhd,bhd->bh",
                query,
                k_state,
            ).unsqueeze(-1)

            output = numerator / denominator.clamp_min(
                self.eps
            )
            outputs.append(output)

        mixed = torch.stack(outputs, dim=2)
        mixed = (
            mixed.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                d_model,
            )
        )

        return self.output(mixed)


class GroupedQueryAttention(SequenceMixer):
    def __init__(self, d_model, num_heads, num_kv_heads=2):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads

        self.q_proj = nn.Linear(
            d_model, num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False
        )
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch, length, width = x.shape

        q = self.q_proj(x).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(
            batch, length, width
        )
        return self.output(output)


class TransformerBlock(nn.Module):
    def __init__(self, mixer, d_model, ffn_dim):
        super().__init__()
        self.mixer_norm = nn.LayerNorm(d_model)
        self.sequence_mixer = mixer
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        x = x + self.sequence_mixer(self.mixer_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MixerLM(nn.Module):
    """v1 모델 (protocol v1 이름). cs_ 이름은 TransformerLM 이 common_set.TokenLM 으로 보낸다."""

    def __init__(self, model, config):
        super().__init__()
        d = config["d_model"]
        self.token_embedding = nn.Embedding(config["vocab_size"], d)
        self.blocks = nn.ModuleList(
            TransformerBlock(MODELS.build(model, d_model=d, num_heads=config["num_heads"]), d, config["ffn_dim"])
            for _ in range(config["num_layers"])
        )
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, config["vocab_size"], bias=False)

        self.apply(self._initialize)

        # Zoology LanguageModel과 동일한 입력-출력 weight tying
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        hidden = self.token_embedding(token_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


@register_model("transformer")
def _transformer(d_model, num_heads):
    return CausalSelfAttention(d_model, num_heads)


@register_model("linear_attention")
def _linear_attention(d_model, num_heads):
    return CausalLinearAttention(d_model, num_heads)


@register_model("gated_linear_attention")
def _gated_linear_attention(d_model, num_heads):
    return GatedLinearAttention(d_model, num_heads)


@register_model("gqa2")
def _gqa2(d_model, num_heads):
    return GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)


# Protocol v2 common architecture set. train.py CONFIG: sequence_length 64 (hyena filter length).
COMMON_SET_L_MAX = 64
# mqar runs on torch 2.2.2 (zoology), where fla chunk_retention cannot load: cs_retnet uses the torch
# parallel form of the same operator (common_set.retention.retention_reference) on every device.
COMMON_SET_RETENTION = "torch"


def _common_set(mixer):
    def build(d_model, num_heads):
        return make_mixer(mixer, d_model, num_heads, COMMON_SET_L_MAX,
                          mixer_width(mixer, d_model, num_heads, COMMON_SET_L_MAX), COMMON_SET_RETENTION)
    return build


for _mixer in COMMON_SET:
    register_model("cs_" + _mixer)(_common_set(_mixer))


def TransformerLM(model, config):
    """train.py 의 생성 지점. cs_ 이름은 공통 뼈대(RMSNorm + SwiGLU, untied head, PyTorch 기본 초기화)를
    쓰고 mixer 만 바꾼다. 그 밖의 이름은 위 MixerLM (v1) 그대로다."""
    if model.startswith("cs_"):
        return TokenLM(config["vocab_size"], model[len("cs_"):], config["d_model"], config["num_layers"],
                       config["num_heads"], COMMON_SET_L_MAX, COMMON_SET_RETENTION)
    return MixerLM(model, config)
