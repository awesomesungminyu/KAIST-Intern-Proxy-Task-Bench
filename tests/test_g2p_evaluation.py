"""Requires the ML interpreter; skipped in a standard-library-only checkout."""
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class DecodeModeTests(unittest.TestCase):
    def test_greedy_decode_restores_both_training_and_evaluation_mode(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy_tasks"))
        from g2p.train import greedy_decode

        class EOSModel(torch.nn.Module):
            def forward(self, src, tgt):
                result = torch.zeros((src.shape[0], tgt.shape[1], 4))
                result[:, :, 2] = 1
                return result

        model = EOSModel()
        for training in [False, True]:
            model.train(training)
            pred = greedy_decode(model, torch.ones((2, 3), dtype=torch.long), 5, "cpu", "fixture")
            self.assertEqual(model.training, training)
            self.assertEqual(pred.tolist(), [[1, 2], [1, 2]])
