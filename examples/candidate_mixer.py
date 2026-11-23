"""Known-answer candidate: exact frozen Transformer mixer, no extra RNG draws."""
from proxy_tasks.common_set.mixers import CausalSelfAttention

class CandidateMixer(CausalSelfAttention):
    def __init__(self, d_model, num_heads, l_max):
        super().__init__(d_model, num_heads)
