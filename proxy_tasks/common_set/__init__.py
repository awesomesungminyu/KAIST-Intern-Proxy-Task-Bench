"""Shared causal sequence mixers and model backbones for the sixteen tasks.

Registered names use the cs_ prefix. RetNet uses an FLA CUDA kernel and a PyTorch CPU reference.
"""
from .backbone import (PARAM_TOLERANCE, Backbone, Block, SequenceClassifier, TokenLM, block_parameters,
                       mixer_width)
from .mixers import NAMES, OFFICIAL_WIDTH, make_mixer

__all__ = ["NAMES", "OFFICIAL_WIDTH", "PARAM_TOLERANCE", "Backbone", "Block", "SequenceClassifier", "TokenLM",
           "block_parameters", "make_mixer", "mixer_width"]

# Per-process registration only; mixers.NAMES and protocol.json remain unchanged.
from .plugins import current as _candidate_info
_info = _candidate_info()
if _info:
    NAMES = [*NAMES, _info["model_id"]]
