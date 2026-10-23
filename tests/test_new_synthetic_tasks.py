"""iclreg / jelassi_copy data, baselines and model contracts on CPU. No training; not experimental evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from validation import runner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy_tasks"))


def sequential_scan(a, b):
    h, out = torch.zeros_like(b[:, 0]), []
    for t in range(b.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, 1)


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class IclregTests(unittest.TestCase):
    def test_least_squares_oracle_is_exact_once_k_reaches_d_and_zero_baseline_is_one(self):
        from iclreg.data import baselines, eval_set
        xs, ys = eval_set(256, 13, 4)
        ref = baselines(xs, ys, 4)
        self.assertLess(max(ref["least_squares_by_k"][4:]), 1e-8)
        self.assertGreater(min(ref["least_squares_by_k"][:4]), 0.05)
        self.assertAlmostEqual(ref["zero_by_k"][0], 1.0, delta=0.3)
        # The fixed validation prompts must not depend on the global RNG.
        torch.manual_seed(123)
        self.assertTrue(torch.equal(eval_set(8, 13, 4)[0], eval_set(8, 13, 4)[0]))

    def test_curriculum_reaches_full_size(self):
        from iclreg.train import curriculum

        class Args:
            n_dims, n_points, curriculum_start_dims, curriculum_start_points, curriculum_interval = 20, 41, 5, 11, 2000
        self.assertEqual(curriculum(Args, 1), (5, 11))
        self.assertEqual(curriculum(Args, 2001), (6, 13))
        self.assertEqual(curriculum(Args, 10 ** 6), (20, 41))

    def test_models_are_causal_over_the_interleaved_prompt(self):
        from iclreg.models import MODELS
        torch.manual_seed(0)
        xs, ys = torch.randn(2, 9, 5), torch.randn(2, 9)
        for name in MODELS.names():
            model = MODELS.build(name, n_dims=5, n_points=9).eval()
            pred = model(xs, ys)
            self.assertEqual(pred.shape, (2, 9))
            xs2, ys2 = xs.clone(), ys.clone()
            ys2[:, 4] += 3          # y_4 may only affect predictions for points 5.. (index >= 5)
            xs2[:, 5:] += 1
            changed = model(xs2, ys2)
            self.assertTrue(torch.allclose(pred[:, :5], changed[:, :5], atol=1e-6), name)
            self.assertFalse(torch.allclose(pred[:, 5:], changed[:, 5:]), name)


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class JelassiCopyTests(unittest.TestCase):
    def test_training_mask_covers_only_the_copy_and_eos_of_complete_examples(self):
        import numpy as np
        from jelassi_copy.data import BOS, EOS, SEP, train_batch
        tokens, mask = train_batch(np.random.default_rng(0), 4, 80, 5, 10)
        for row, m in zip(tokens.tolist(), mask.tolist()):
            starts = [i for i, t in enumerate(row) if t == BOS]
            for s in starts:
                sep = row.index(SEP, s) if SEP in row[s:] else None
                if sep is None or sep + (sep - s) >= len(row):
                    self.assertEqual(sum(m[s:]), 0)          # truncated example: no loss
                    continue
                L = sep - s - 1
                self.assertEqual(row[s + 1:sep], row[sep + 1:sep + 1 + L])
                self.assertEqual(row[sep + 1 + L], EOS)
                self.assertEqual(m[s:sep + 1], [0] * (L + 2))
                self.assertEqual(m[sep + 1:sep + 2 + L], [1] * (L + 1))

    def test_exact_match_requires_every_copied_character(self):
        import torch.nn.functional as F
        from jelassi_copy.data import VOCAB, eval_strings, pad_batch, score
        examples = eval_strings([("id", 5, 7)], 3)["id"]
        tokens = pad_batch(examples)
        lengths = [(len(e) - 3) // 2 for e in examples]
        logits = F.one_hot(torch.roll(tokens, -1, 1), VOCAB).float()
        self.assertEqual(score(logits, tokens, lengths)[0], [1.0, 1.0, 1.0])
        L = lengths[1]
        logits[1, 2 * L] = 0
        logits[1, 2 * L, (tokens[1, 2 * L + 1] + 1) % 26] = 1   # last copied character wrong
        self.assertEqual(score(logits, tokens, lengths)[0], [1.0, 0.0, 1.0])

    def test_eval_bins_exclude_training_lengths(self):
        from jelassi_copy.train import bins_for

        class Args:
            min_len, max_len, ood_bins = 5, 50, "51-100,101-200"
        self.assertEqual([b[0] for b in bins_for(Args)], ["id", "ood_51_100", "ood_101_200"])
        Args.ood_bins = "50-100"
        with self.assertRaises(ValueError):
            bins_for(Args)

    def test_hard_alibi_windows(self):
        from jelassi_copy.models import hard_alibi_mask
        allowed = torch.isfinite(hard_alibi_mask(4, 2, 6, "cpu"))
        self.assertEqual(allowed[0].sum(1).tolist(), [1] * 6)                 # head 0: itself only
        self.assertEqual(allowed[1].sum(1).tolist(), [1, 2, 2, 2, 2, 2])      # head 1: previous token too
        self.assertEqual(allowed[3].sum(1).tolist(), [1, 2, 3, 4, 5, 6])      # unmasked: full causal

    def test_models_are_causal_and_accept_lengths_beyond_training_context(self):
        from jelassi_copy.data import VOCAB
        from jelassi_copy.models import MODELS
        torch.manual_seed(0)
        x = torch.randint(0, 26, (2, 300))
        y = x.clone()
        y[:, 150:] = (y[:, 150:] + 1) % 26
        for name in MODELS.names():
            model = MODELS.build(name, vocab_size=VOCAB).eval()
            a, b = model(x), model(y)
            self.assertEqual(a.shape, (2, 300, VOCAB))
            self.assertTrue(torch.allclose(a[:, :150], b[:, :150], atol=1e-5), name)
            self.assertFalse(torch.allclose(a[:, 150:], b[:, 150:]), name)


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class LinearScanTests(unittest.TestCase):
    def test_custom_backward_matches_sequential_autograd(self):
        from jelassi_copy.models import LinearScan
        torch.manual_seed(0)
        for scan in [LinearScan]:
            a = torch.rand(2, 7, 3, 4, dtype=torch.float64, requires_grad=True)
            b = torch.randn(2, 7, 3, 4, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(scan.apply, (a, b)))
            ref = sequential_scan(a, b)
            self.assertTrue(torch.allclose(scan.apply(a, b), ref))
            w = torch.randn_like(ref)
            self.assertTrue(all(torch.allclose(x, y) for x, y in zip(
                torch.autograd.grad((scan.apply(a, b) * w).sum(), (a, b)),
                torch.autograd.grad((ref * w).sum(), (a, b)))))


class ProtocolTests(unittest.TestCase):
    def test_step_budget_and_top_level_metric(self):
        for task, metric, steps in [("iclreg", "val_k_ge_d_mse", 45000), ("jelassi_copy", "ood_exact_match", 4000)]:
            spec = runner.load_protocol(task)[1]
            self.assertEqual(spec["budget_rule"]["full"], steps)
            self.assertEqual(spec["metric"], metric)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "transformer_seed0" / "history.json"
                p.parent.mkdir()
                p.write_text(json.dumps({"history": [{"step": steps}], metric: 0.25}))
                self.assertEqual(runner.extract_metrics(task, spec, Path(tmp), "transformer", 0, "full")[0], {metric: 0.25})
                p.write_text(json.dumps({"history": [{"step": steps - 1}], metric: 0.25}))
                with self.assertRaises(ValueError):
                    runner.extract_metrics(task, spec, Path(tmp), "transformer", 0, "full")

    def test_synthetic_tasks_need_no_data_and_get_no_data_dir(self):
        for task in ["iclreg", "jelassi_copy"]:
            spec = runner.load_protocol(task)[1]
            args = runner.parser().parse_args(["run", "--task", task, "--model", "lstm", "--out", "/tmp/unused"])
            self.assertEqual(runner.asset_paths(args, spec), {})
            with tempfile.TemporaryDirectory() as tmp:
                cmd, cwd, _ = runner.build_command(args, spec, 3, Path(tmp))
            self.assertEqual(cmd[2], task + ".train")
            self.assertNotIn("--data_dir", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], "3")


if __name__ == "__main__":
    unittest.main()
