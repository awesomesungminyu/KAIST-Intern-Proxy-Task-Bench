"""RegBench model factories for the nine common sequence mixers."""
from common.registry import ModelRegistry
from common_set import NAMES, TokenLM
MODELS = ModelRegistry('regbench')
for mixer in NAMES:
    MODELS.register('cs_' + mixer)(lambda vocab_size, max_len, mixer=mixer: TokenLM(vocab_size, mixer, d_model=128, n_layers=8, n_heads=2, l_max=max_len))
def build_model(name, vocab_size, max_len):
    return MODELS.build(name, vocab_size=vocab_size, max_len=max_len)
