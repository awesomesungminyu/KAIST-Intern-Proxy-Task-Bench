"""Protocol v2 common architecture set (proxy_tasks/common_set). Requires the ML interpreter.

Everything runs on CPU except RetentionKernelTests, which compares fla chunk_retention with the torch
parallel form (mqar's cs_retnet) on CUDA and skips without a GPU or vendored fla. Training smoke runs are not here.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proxy_tasks"))
# Protocol v1 baseline: the commit whose models and hashes v2 must leave untouched.
BASELINE_COMMIT = "b45b67f"
# Task configurations of the v2 registrations: (d_model, n_layers, n_heads, l_max).
TASK_CONFIGS = {"regbench": (128, 8, 2, 256), "mqar": (128, 2, 4, 64), "jelassi_copy": (256, 4, 8, 1024),
                "formal": (32, 2, 4, 128), "scifar": (32, 2, 4, 1024), "fflm": (512, 6, 8, 512)}


def git_show(path):
    proc = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def load_source(name, source):
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location("runner_under_test", ROOT / "validation/runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class MixerTests(unittest.TestCase):
    def test_every_mixer_is_causal_with_finite_gradients(self):
        from common_set import NAMES, make_mixer, mixer_width
        for d, heads in [(32, 4), (64, 8), (64, 2)]:
            for name in NAMES:
                with self.subTest(mixer=name, d=d, heads=heads):
                    torch.manual_seed(0)
                    mixer = make_mixer(name, d, heads, 64, mixer_width(name, d, heads, 64))
                    x = torch.randn(2, 24, d, requires_grad=True)
                    y = mixer(x)
                    self.assertEqual(tuple(y.shape), (2, 24, d))
                    y.pow(2).mean().backward()
                    grads = [p.grad for p in mixer.parameters() if p.grad is not None]
                    self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                    changed = x.detach().clone()
                    changed[:, 12:] += 1.0
                    with torch.no_grad():
                        leak = (mixer(x.detach())[:, :12] - mixer(changed)[:, :12]).abs().max().item()
                    self.assertLess(leak, 1e-5)   # FFT mixers leave ~1e-7 round-off

    def test_width_selection_consumes_no_random_numbers(self):
        from common_set import block_parameters
        state = torch.random.get_rng_state()
        block_parameters("mamba", 96, 4, 64, 40)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_block_parameters_within_tolerance_of_transformer(self):
        from common_set import NAMES, PARAM_TOLERANCE, block_parameters, mixer_width
        for task, (d, _, heads, l_max) in TASK_CONFIGS.items():
            target = block_parameters("transformer", d, heads, l_max)
            for name in NAMES:
                with self.subTest(task=task, mixer=name):
                    count = block_parameters(name, d, heads, l_max, mixer_width(name, d, heads, l_max))
                    self.assertLessEqual(abs(count / target - 1), PARAM_TOLERANCE)

    def test_official_width_kept_when_within_tolerance(self):
        from common_set import mixer_width
        self.assertEqual(mixer_width("s4d", 128, 2, 256), 64)      # regbench: official d_state
        self.assertEqual(mixer_width("hyena", 128, 2, 256), 64)    # regbench: official filter_order
        self.assertEqual(mixer_width("hyena", 32, 4, 128), 8)      # formal: matched
        self.assertIsNone(mixer_width("transformer", 32, 4, 128))

    def test_retention_reference_equals_recurrent_form(self):
        from common_set.retention import retention_reference
        torch.manual_seed(0)
        B, T, H, K, V = 2, 9, 3, 4, 5
        q, k, v = torch.randn(B, T, H, K, dtype=torch.float64), torch.randn(B, T, H, K, dtype=torch.float64), \
            torch.randn(B, T, H, V, dtype=torch.float64)
        gamma = 1 - 2.0 ** (-5.0 - torch.arange(H, dtype=torch.float64))
        state = torch.zeros(B, H, K, V, dtype=torch.float64)
        expected = []
        for t in range(T):
            state = gamma[None, :, None, None] * state + torch.einsum("bhk,bhv->bhkv", k[:, t], v[:, t])
            expected.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * K ** -0.5, state))
        torch.testing.assert_close(retention_reference(q, k, v), torch.stack(expected, 1))

    def test_retention_reference_equals_fla_naive_when_vendored(self):
        from common_set.retention import fla_vendor_root, retention_reference
        naive = fla_vendor_root() / "fla/ops/retention/naive.py"
        if not naive.is_file():
            self.skipTest(f"fla not vendored at {fla_vendor_root()}")
        spec = importlib.util.spec_from_file_location("fla_naive_retention", naive)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 11, 8) for _ in range(3))   # [B,H,T,D]
        ours = retention_reference(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
        torch.testing.assert_close(ours, module.naive_retention(q, k, v), rtol=1e-5, atol=1e-5)

    def test_retention_backend_is_explicit(self):
        from common_set.mixers import RetNetMixer
        from common_set.retention import retention
        q = torch.randn(1, 3, 2, 4)
        with self.assertRaises(ValueError):
            retention(q, q, q, "fla")
        torch.manual_seed(0)
        auto = RetNetMixer(16, 2, 32)
        torch.manual_seed(0)
        torch_form = RetNetMixer(16, 2, 32, retention_backend="torch")
        x = torch.randn(2, 5, 16)
        torch.testing.assert_close(auto(x), torch_form(x), rtol=0, atol=0)     # CPU: both are the parallel form

    def test_linear_scan_matches_autograd_loop(self):
        from common_set.mixers import LinearScan
        torch.manual_seed(0)
        a = torch.rand(2, 7, 3, 4, dtype=torch.float64, requires_grad=True)
        b = torch.randn(2, 7, 3, 4, dtype=torch.float64, requires_grad=True)
        H, h = [], torch.zeros(2, 3, 4, dtype=torch.float64)
        for t in range(7):
            h = a[:, t] * h + b[:, t]
            H.append(h)
        expected = torch.stack(H, 1)
        torch.testing.assert_close(LinearScan.apply(a, b), expected)
        self.assertTrue(torch.autograd.gradcheck(LinearScan.apply, (a, b)))


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "needs CUDA")
class RetentionKernelTests(unittest.TestCase):
    """mqar's cs_retnet (torch parallel form) against the fla kernel the other tasks use, same weights."""

    def setUp(self):
        from common_set.retention import fla_chunk_retention
        try:
            fla_chunk_retention()
        except Exception as exc:
            self.skipTest(f"fla chunk_retention unavailable: {exc}")

    def test_torch_form_matches_fla_kernel(self):
        from common_set import TokenLM
        from common_set.mixers import RetNetMixer
        # mixer level: head dims of every v2 task (k 64/32/8, v 6/21/43/44 among them), then mqar's whole model
        for d, heads, width, T in [(128, 2, 88, 256), (32, 4, 24, 100), (256, 4, 168, 220), (512, 8, 344, 511),
                                   (128, 4, 88, 64), (64, 2, 86, 37)]:
            with self.subTest(d=d, heads=heads, width=width, T=T):
                torch.manual_seed(0)
                kernel = RetNetMixer(d, heads, width).cuda()
                ref = RetNetMixer(d, heads, width, retention_backend="torch").cuda()
                ref.load_state_dict(kernel.state_dict())
                x = torch.randn(3, T, d, device="cuda")
                xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
                ya, yb = kernel(xa), ref(xb)
                torch.testing.assert_close(yb, ya, rtol=1e-4, atol=1e-5)
                ya.square().sum().backward(); yb.square().sum().backward()
                torch.testing.assert_close(xb.grad, xa.grad, rtol=1e-4, atol=1e-4)
                for (name, pa), pb in zip(kernel.named_parameters(), ref.parameters()):
                    torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-4, msg=name)
        torch.manual_seed(0)
        kernel = TokenLM(8192, "retnet", 128, 2, 4, 64).cuda()
        ref = TokenLM(8192, "retnet", 128, 2, 4, 64, retention_backend="torch").cuda()
        ref.load_state_dict(kernel.state_dict())
        tokens = torch.randint(0, 8192, (8, 64), device="cuda")
        torch.testing.assert_close(ref(tokens), kernel(tokens), rtol=1e-4, atol=1e-5)


@unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
class TaskRegistrationTests(unittest.TestCase):
    def test_registered_models_run_and_match_parameters(self):
        from common_set import NAMES, PARAM_TOLERANCE
        from common_set.tfl import CommonSetTFLModel
        import fflm.models as fflm
        import jelassi_copy.models as jelassi
        import mqar.models as mqar
        from mqar.train import CONFIG
        import regbench.models as regbench
        import scifar.models as scifar
        cases = {
            "regbench": (lambda n: regbench.build_model("cs_" + n, 22, 256), torch.randint(0, 22, (2, 16)), (2, 16, 22)),
            "mqar": (lambda n: mqar.TransformerLM("cs_" + n, CONFIG), torch.randint(0, 8192, (2, 16)), (2, 16, 8192)),
            "jelassi_copy": (lambda n: jelassi.MODELS.build("cs_" + n, vocab_size=30), torch.randint(0, 30, (2, 16)),
                             (2, 16, 30)),
            "scifar": (lambda n: scifar.MODELS.build("cs_" + n), torch.randn(2, 1024, 3), (2, 10)),
            "fflm_smoke": (lambda n: fflm.model_for("cs_" + n, smoke=True), torch.randint(0, 5, (2, 16)), (2, 16, 5)),
            "formal": (lambda n: CommonSetTFLModel(3, 2, n, 32, 4, 2), torch.randint(0, 3, (16, 2)), (16, 2, 2)),
        }
        for task, (build, x, shape) in cases.items():
            reference = sum(p.numel() for p in build("transformer").parameters())
            for name in NAMES:
                with self.subTest(task=task, mixer=name):
                    torch.manual_seed(0)
                    model = build(name)
                    with torch.no_grad():
                        y = model(x)
                    self.assertEqual(tuple(y.shape), shape)
                    self.assertTrue(torch.isfinite(y).all())
                    total = sum(p.numel() for p in model.parameters())
                    self.assertLessEqual(abs(total / reference - 1), PARAM_TOLERANCE)
        y = CommonSetTFLModel(3, 2, "gla", 32, 4, 2)(torch.randint(0, 3, (16, 2)))
        self.assertTrue(((y >= 0) & (y <= 1)).all())

    def test_only_mqar_retnet_uses_the_torch_retention(self):
        from common_set.mixers import RetNetMixer
        import mqar.models as mqar
        from mqar.train import CONFIG
        import regbench.models as regbench
        backends = lambda model: {m.retention_backend for m in model.modules() if isinstance(m, RetNetMixer)}
        self.assertEqual(backends(mqar.TransformerLM("cs_retnet", CONFIG)), {"torch"})
        self.assertEqual(backends(mqar.MODELS.build("cs_retnet", d_model=CONFIG["d_model"], num_heads=CONFIG["num_heads"])), {"torch"})
        self.assertEqual(backends(regbench.build_model("cs_retnet", 22, 256)), {"auto"})

    def test_formal_registry_lists_the_common_set(self):
        from common_set import NAMES
        import formal_languages.models as formal
        self.assertEqual(formal.COMMON_SET, NAMES)
        for name in NAMES:
            args = formal.TFL_MODELS["cs_" + name]
            self.assertEqual(args[args.index("-cs_mixer") + 1], name)

    def test_v1_models_are_unchanged(self):
        """Same seed -> identical parameters and outputs as the baseline commit's models.py."""
        cases = {"regbench": (["transformer", "lstm"], lambda M, n: M.build_model(n, 22, 64), torch.randint(0, 22, (2, 12))),
                 "scifar": (["transformer", "lstm"], lambda M, n: M.MODELS.build(n), torch.randn(2, 1024, 3)),
                 "jelassi_copy": (["transformer", "lstm", "mamba"], lambda M, n: M.MODELS.build(n, vocab_size=30),
                                  torch.randint(0, 30, (2, 12))),
                 "fflm": (["lstm"], lambda M, n: M.model_for(n), torch.randint(0, 5, (2, 12)))}
        from mqar.train import CONFIG
        mqar_config = dict(CONFIG, vocab_size=64)
        cases["mqar"] = (["transformer", "linear_attention", "gated_linear_attention", "gqa2"],
                         lambda M, n: M.TransformerLM(n, mqar_config), torch.randint(0, 64, (2, 12)))
        for task, (names, build, x) in cases.items():
            source = git_show(f"proxy_tasks/{task}/models.py")
            if source is None:
                self.skipTest(f"baseline commit {BASELINE_COMMIT} unavailable")
            try:
                baseline = load_source(f"baseline_{task}_models", source)
                current = importlib.import_module(f"{task}.models")
            except ImportError as exc:   # e.g. scifar needs torch.nn.attention (torch 2.5.1 runtime)
                with self.subTest(task=task):
                    self.skipTest(f"{task} models need another runtime: {exc}")
                continue
            for name in names:
                with self.subTest(task=task, model=name):
                    torch.manual_seed(0)
                    old = build(baseline, name).eval()
                    torch.manual_seed(0)
                    new = build(current, name).eval()
                    old_state, new_state = old.state_dict(), new.state_dict()
                    self.assertEqual(list(old_state), list(new_state))
                    self.assertTrue(all(torch.equal(old_state[k], new_state[k]) for k in old_state))
                    with torch.no_grad():
                        self.assertTrue(torch.equal(old(x), new(x)))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.v1 = json.loads((ROOT / "validation/protocols/v1.json").read_text())
        self.v2 = json.loads((ROOT / "validation/protocols/v2.json").read_text())

    def test_v1_protocol_and_evaluation_hashes_unchanged(self):
        source = git_show("validation/protocols/v1.json")
        if source is None:
            self.skipTest(f"baseline commit {BASELINE_COMMIT} unavailable")
        self.assertEqual(json.loads(source), self.v1)
        runner = load_runner()
        # evaluation_sha256 prefixes at the baseline commit (tasks that need no external checkout).
        expected = {"regbench": "4fd5b2298f69", "g2p": "6cf061f52af0", "zinc": "0180f0738086", "fflm": "1153a2c53e36",
                    "ett": "99261617eade", "scifar": "8885a4c524c6", "sc_raw": "5045e108bc4c", "tier0": "133f2161e07e",
                    "iclreg": "8ae8e95e7434", "jelassi_copy": "f1e3822b4013"}
        self.assertEqual(runner.digest(runner.PROTOCOL)[:12], "bb604559925f")
        for task, prefix in expected.items():
            with self.subTest(task=task):
                self.assertEqual(runner.evaluation_identity(types.SimpleNamespace(task=task))[:12], prefix)

    def test_v2_differs_from_v1_only_in_models_limitations_and_fflm_steps(self):
        self.assertEqual(self.v2["version"], "proxy-validation-v2")
        self.assertEqual(sorted(self.v2["tasks"]),
                         sorted(["regbench", "mqar", "jelassi_copy", "parity", "tomita5", "tomita6", "scifar", "fflm"]))
        for task, spec in self.v2["tasks"].items():
            with self.subTest(task=task):
                old = self.v1["tasks"][task]
                changed = {k for k in set(spec) | set(old) if spec.get(k) != old.get(k)}
                self.assertEqual(changed, {"models", "limitations", "steps"} if task == "fflm" else {"models", "limitations"})
                self.assertTrue(all(m.startswith("cs_") for m in spec["models"]))
        self.assertEqual(self.v2["tasks"]["mqar"]["models"], ["cs_" + n for n in
                         ["transformer", "gqa", "linear_attention", "gla", "mamba", "hyena", "lstm", "s4d", "retnet"]])
        fflm = self.v2["tasks"]["fflm"]
        self.assertEqual(fflm["steps"], {m: 10000 for m in fflm["models"]})

    @unittest.skipIf(torch is None, "torch unavailable; run again with the ML interpreter")
    def test_v2_models_are_registered(self):
        import formal_languages.models as formal
        for task, spec in self.v2["tasks"].items():
            names = (formal.TFL_MODELS if task in {"parity", "tomita5", "tomita6"}
                     else importlib.import_module(f"{task}.models").MODELS.names())
            with self.subTest(task=task):
                self.assertTrue(set(spec["models"]) <= set(names))

    def test_v2_entry_point_uses_v2_and_defaults_tfl_root(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/validation_v2.py"), "list"], cwd=ROOT,
                              capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["version"], "proxy-validation-v2")
        code = ("import sys; sys.argv=['x']; sys.path.insert(0, 'scripts'); import validation_v2 as v; "
                "seen=[]; v.runner.main=lambda a: seen.append(a) or 0; v.main(['check','--task','parity','--model','cs_gla']); "
                "print(seen[0][-1])")
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.strip().endswith("tfl-v2"))


if __name__ == "__main__":
    unittest.main()
