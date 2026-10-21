import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from validation.torch_compat import install_cpu_sequence_lengths


@unittest.skipIf(torch is None, "torch unavailable")
class PackedSequenceTests(unittest.TestCase):
    def test_install_is_idempotent_and_preserves_cpu_results(self):
        original = torch.nn.utils.rnn.pack_padded_sequence
        with patch.object(torch.nn.utils.rnn, "pack_padded_sequence", original):
            x = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
            lengths = torch.tensor([3, 2])
            expected = original(x, lengths, enforce_sorted=False)
            install_cpu_sequence_lengths()
            installed = torch.nn.utils.rnn.pack_padded_sequence
            install_cpu_sequence_lengths()
            self.assertIs(installed, torch.nn.utils.rnn.pack_padded_sequence)
            actual = installed(x, lengths, enforce_sorted=False)
            self.assertTrue(torch.equal(actual.data, expected.data))
            self.assertTrue(torch.equal(actual.batch_sizes, expected.batch_sizes))

    def test_cuda_lengths_work_without_changing_packed_values(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        original = torch.nn.utils.rnn.pack_padded_sequence
        with patch.object(torch.nn.utils.rnn, "pack_padded_sequence", original):
            x = torch.arange(24, dtype=torch.float32, device="cuda").reshape(3, 2, 4)
            lengths = torch.tensor([3, 2], device="cuda")
            expected = original(x, lengths.cpu(), enforce_sorted=False)
            install_cpu_sequence_lengths()
            actual = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
            self.assertTrue(torch.equal(actual.data, expected.data))
            self.assertEqual(actual.data.device.type, "cuda")
            self.assertEqual(actual.batch_sizes.device.type, "cpu")
