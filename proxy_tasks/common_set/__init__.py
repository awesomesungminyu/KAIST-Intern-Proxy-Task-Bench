"""Common architecture set (protocol v2): the same mixers, by name and code, in every sequence task.

Task models.py files register them as cs_<name>; the existing per-task model IDs are untouched.
Pure PyTorch except retnet, whose retention kernel is fla on CUDA (retention.py).
This package lives outside proxy_tasks/common/ so adding it does not change any evaluation_sha256,
and it uses only relative imports so the TFL adapter can import it as proxy_tasks.common_set.
"""
from .backbone import (PARAM_TOLERANCE, Backbone, Block, SequenceClassifier, TokenLM, block_parameters,
                       mixer_width)
from .mixers import NAMES, OFFICIAL_WIDTH, make_mixer

__all__ = ["NAMES", "OFFICIAL_WIDTH", "PARAM_TOLERANCE", "Backbone", "Block", "SequenceClassifier", "TokenLM",
           "block_parameters", "make_mixer", "mixer_width"]
