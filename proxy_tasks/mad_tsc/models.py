"""MAD recipe registry (no ML imports). Names map to ordered MAD layer IDs.

Register new layers in third_party/mad-lab/mad/registry.py plus a layer config,
record the change as a separate third_party patch, then add a recipe here.
Input: (B,L) token IDs, output: (B,L,V) causal logits via MAD LanguageModel.
Only mh_attention_swiglu has a completed historical official benchmark.
"""
from common.registry import ModelRegistry

MODELS = ModelRegistry("mad_tsc")
register_model = MODELS.register


@register_model("mh_gla_swiglu")
def _mh_gla_swiglu():
    return ['mh-gated-linear-attention', 'swiglu', 'mh-gated-linear-attention', 'swiglu']


@register_model("mh_gla_moe")
def _mh_gla_moe():
    return ['mh-gated-linear-attention', 'moe-mlp', 'mh-gated-linear-attention', 'moe-mlp']


@register_model("hyena_swiglu")
def _hyena_swiglu():
    return ['hyena', 'swiglu', 'hyena', 'swiglu']


@register_model("hyena_moe")
def _hyena_moe():
    return ['hyena', 'moe-mlp', 'hyena', 'moe-mlp']


@register_model("mamba")
def _mamba():
    return ['mamba', 'mamba', 'mamba', 'mamba']


@register_model("mh_hyena_swiglu")
def _mh_hyena_swiglu():
    return ['mh-hyena', 'swiglu', 'mh-hyena', 'swiglu']


@register_model("mh_attention_moe")
def _mh_attention_moe():
    return ['mh-attention', 'moe-mlp', 'mh-attention', 'moe-mlp']


@register_model("mh_attention_swiglu")
def _mh_attention_swiglu():
    return ['mh-attention', 'swiglu', 'mh-attention', 'swiglu']


@register_model("mh_hyena_moe")
def _mh_hyena_moe():
    return ['mh-hyena', 'moe-mlp', 'mh-hyena', 'moe-mlp']


@register_model("striped_mamba_swiglu")
def _striped_mamba_swiglu():
    return ['mamba', 'swiglu', 'mh-attention', 'swiglu']


@register_model("striped_mamba_moe")
def _striped_mamba_moe():
    return ['mamba', 'moe-mlp', 'mh-attention', 'moe-mlp']


@register_model("striped_hyena_swiglu")
def _striped_hyena_swiglu():
    return ['hyena', 'swiglu', 'mh-attention', 'swiglu']


@register_model("striped_mh_hyena_swiglu")
def _striped_mh_hyena_swiglu():
    return ['mh-hyena', 'swiglu', 'mh-attention', 'swiglu']


@register_model("striped_hyena_moe")
def _striped_hyena_moe():
    return ['hyena', 'moe-mlp', 'mh-attention', 'moe-mlp']


@register_model("striped_mh_hyena_moe")
def _striped_mh_hyena_moe():
    return ['mh-hyena', 'moe-mlp', 'mh-attention', 'moe-mlp']


@register_model("striped_hyena_experts_moe")
def _striped_hyena_experts_moe():
    return ['hyena-experts', 'moe-mlp', 'mh-attention', 'moe-mlp']
