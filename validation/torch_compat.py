"""Process-local compatibility for the pinned legacy formal-language runner."""
from functools import wraps


def install_cpu_sequence_lengths():
    import torch

    original = torch.nn.utils.rnn.pack_padded_sequence
    if getattr(original, "_proxy_cpu_lengths", False):
        return

    @wraps(original)
    def pack(input, lengths, *args, **kwargs):
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu()
        return original(input, lengths, *args, **kwargs)

    pack._proxy_cpu_lengths = True
    torch.nn.utils.rnn.pack_padded_sequence = pack
