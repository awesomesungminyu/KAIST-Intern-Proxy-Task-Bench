"""iclreg data, baselines and model contracts on CPU. No training; not experimental evidence."""
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


class ProtocolTests(unittest.TestCase):
    def test_step_budget_and_top_level_metric(self):
        for task, metric, steps in [("iclreg", "val_k_ge_d_mse", 45000)]:
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
        for task in ["iclreg"]:
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
