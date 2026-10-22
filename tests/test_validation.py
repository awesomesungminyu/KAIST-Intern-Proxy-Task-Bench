"""Contract tests use synthetic records; they are never experimental evidence."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from validation import runner


def measured(seed=0, value=0.5):
    return {"task": "regbench", "model": "transformer", "seed": seed, "status": "done",
            "profile": "full", "evidence": "measured", "protocol_sha256": "test-only",
            "budget": {"steps": 2000}, "selection": "final_step",
            "metric": {"name": "next_token_acc", "direction": "maximize", "split": "reused_test_stream"},
            "metrics": {"next_token_acc": value}, "limitations": ["synthetic contract-test fixture"],
            "provenance": {"data": {"sha256": "same-data"}, "environment": {"gpu": "same-gpu"},
                           "evaluation_sha256": "same-evaluator", "source": {"sha256": "source-a"}}}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def compare(self, a, b):
        for name, rows in [("a", a), ("b", b)]:
            runner.write_json(self.root / (name + ".json"), {"status": "done", "runs": rows})
        return runner.compare(self.root / "a.json", self.root / "b.json")

    def test_paired_three_seed_comparison(self):
        a = [measured(s, 0.5 + s * 0.1) for s in [0, 1, 2]]
        b = [measured(s, 0.55 + s * 0.1) for s in [2, 0, 1]]
        for row in b:
            row["provenance"]["source"]["sha256"] = "candidate-source"
        result = self.compare(a, b)
        self.assertAlmostEqual(result["mean_improvement"], 0.05)
        self.assertEqual(result["evidence"], "repeated_seed_measurement")

    def test_minimize_and_pilot(self):
        a, b = measured(value=0.4), measured(value=0.3)
        for row in [a, b]:
            row["metric"]["direction"] = "minimize"
        result = self.compare([a], [b])
        self.assertAlmostEqual(result["mean_improvement"], 0.1)
        self.assertEqual(result["evidence"], "single_seed_pilot")

    def test_rejects_smoke_failure_timeout_and_unmeasured(self):
        for field, value in [("profile", "smoke"), ("status", "failed"), ("status", "timeout"), ("evidence", "smoke_only")]:
            with self.subTest(field=field, value=value):
                b = measured()
                b[field] = value
                with self.assertRaises(ValueError):
                    self.compare([measured()], [b])

    def test_rejects_protocol_budget_metric_and_selection_changes(self):
        for field in ["task", "protocol_sha256", "budget", "metric", "selection"]:
            with self.subTest(field=field):
                b = measured()
                b[field] = "changed"
                with self.assertRaises(ValueError):
                    self.compare([measured()], [b])

    def test_rejects_data_environment_or_evaluation_changes(self):
        for field in ["data", "environment", "evaluation_sha256"]:
            b = measured()
            b["provenance"][field] = "changed"
            with self.assertRaises(ValueError):
                self.compare([measured()], [b])

    def test_rejects_unmatched_and_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            self.compare([measured(0)], [measured(1)])
        with self.assertRaises(ValueError):
            self.compare([measured(0), measured(0)], [measured(0)])

    def test_rejects_mixed_source_revisions(self):
        a = [measured(0), measured(1)]
        a[1]["provenance"]["source"]["sha256"] = "different"
        with self.assertRaises(ValueError):
            self.compare(a, [measured(0), measured(1)])

    def test_rejects_nan_and_empty_results(self):
        # write_json refuses NaN at creation, too.
        with self.assertRaises(ValueError):
            runner.write_json(self.root / "nan.json", measured(value=float("nan")))
        with self.assertRaises(ValueError):
            self.compare([], [])
        for name, value in [("a", 0.5), ("b", float("nan"))]:
            (self.root / (name + ".json")).write_text(json.dumps({"status": "done", "runs": [measured(value=value)]}))
        with self.assertRaises(ValueError):
            runner.compare(self.root / "a.json", self.root / "b.json")

    def test_rejects_in_progress_or_missing_seed_summary(self):
        for summary in [{"status": "running", "runs": [measured()]},
                        {"status": "done", "requested_seeds": [0, 1, 2], "runs": [measured()]}]:
            p = self.root / "partial.json"
            p.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                runner.load_runs(p)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = runner.parser().parse_args(["run", "--task", "regbench", "--model", "transformer",
                                              "--seeds", "0", "--out", str(self.root / "experiment")])
        self.spec = runner.load_protocol("regbench")[1]

    def test_rejects_incomplete_budget_and_missing_metric(self):
        f = self.root / "transformer_seed0/history.json"
        f.parent.mkdir()
        for data in [{"history": []}, {"history": [{"step": 10, "next_token_acc": 0.5}]},
                     {"history": [{"step": 20}]}]:
            f.write_text(json.dumps(data))
            with self.assertRaises((ValueError, KeyError)):
                runner.extract_metrics("regbench", self.spec, self.root, "transformer", 0, "smoke")

    def test_zinc_search_rejects_test_evaluation_results(self):
        spec = runner.load_protocol("zinc")[1]
        f = self.root / "gine_seed0/history.json"
        f.parent.mkdir()
        raw = {"history": [{"epoch": 1}], "best_val": 0.5,
               "test_mae": 0.4, "args": {"validation_only": True}}
        f.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            runner.extract_metrics("zinc", spec, self.root, "gine", 0, "smoke")
        raw["test_mae"] = None
        f.write_text(json.dumps(raw))
        metrics, _ = runner.extract_metrics("zinc", spec, self.root, "gine", 0, "smoke")
        self.assertEqual(metrics, {"best_val": 0.5})

    def test_fflm_search_rejects_ood_evaluation(self):
        spec = runner.load_protocol("fflm")[1]
        f = self.root / "run/result.json"
        f.parent.mkdir()
        raw = {"status": "done", "config": {"smoke": True, "steps": 3},
               "result": {"id": {"read_error_rate": 0.5}, "sparse": {"read_error_rate": 0.5}}}
        f.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            runner.extract_metrics("fflm", spec, self.root, "lstm", 0, "smoke")
        del raw["result"]["sparse"]
        f.write_text(json.dumps(raw))
        metrics, _ = runner.extract_metrics("fflm", spec, self.root, "lstm", 0, "smoke")
        self.assertEqual(metrics, {"id_read_error_rate": 0.5})

    def run_fake(self, code, *, timeout=10):
        self.args.timeout = timeout
        source = runner.source_identity(self.args)
        checked = {"source": source, "data": runner.fingerprint({}), "environment": {}, "evaluation_sha256": "fixture"}

        def fake_command(args, spec, seed, destination):
            return [sys.executable, "-c", code, str(destination)], runner.ROOT, {}

        with patch.object(runner, "preflight", return_value=checked), \
             patch.object(runner, "gpu_lock", side_effect=nullcontext), \
             patch.object(runner, "require_idle_gpu"), \
             patch.object(runner, "build_command", side_effect=fake_command):
            return runner.run(self.args)

    def test_successful_child_writes_traceable_smoke_result(self):
        code = ("import pathlib,sys,json; p=pathlib.Path(sys.argv[1])/'native/transformer_seed0'; "
                "p.mkdir(parents=True); (p/'history.json').write_text(json.dumps("
                "{'history':[{'step':20,'next_token_acc':0.5}]})); print('params=123')")
        self.assertEqual(self.run_fake(code), 0)
        result = json.loads((self.args.out / "summary.json").read_text())["runs"][0]
        self.assertEqual(result["evidence"], "smoke_only")
        self.assertEqual(result["model_parameters"], 123)
        self.assertTrue(Path(result["raw_artifact"]).is_file())
        with self.assertRaises(FileExistsError):
            runner.run(self.args)

    def test_failed_child_has_no_success_metric(self):
        self.assertEqual(self.run_fake("raise RuntimeError('fixture failure')"), 1)
        result = json.loads((self.args.out / "summary.json").read_text())["runs"][0]
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("metrics", result)

    def test_timeout_is_not_reported_as_success(self):
        self.assertEqual(self.run_fake("import time; time.sleep(10)", timeout=0.05), 1)
        result = json.loads((self.args.out / "summary.json").read_text())["runs"][0]
        self.assertEqual(result["status"], "timeout")

    def test_duplicate_seed_rejected_before_creating_output(self):
        self.args.seeds = [0, 0]
        with self.assertRaises(ValueError):
            runner.run(self.args)
        self.assertFalse(self.args.out.exists())

    def test_fingerprint_depends_on_contents_not_location(self):
        for name in ["one", "two"]:
            (self.root / name).mkdir()
            (self.root / name / "x").write_text("same")
        a = runner.fingerprint({"input": self.root / "one"})
        self.assertEqual(a, runner.fingerprint({"input": self.root / "two"}))
        (self.root / "two/x").write_text("changed")
        self.assertNotEqual(a, runner.fingerprint({"input": self.root / "two"}))

    def test_all_tasks_have_explicit_metric_scope(self):
        catalog = json.loads(runner.PROTOCOL.read_text())["tasks"]
        self.assertEqual(len(catalog), 15)
        for spec in catalog.values():
            self.assertTrue(spec["limitations"])
            self.assertIn(spec["direction"], ["maximize", "minimize", "descriptive"])
            self.assertIn("smoke", spec)
            self.assertIn("full", spec)

    def test_model_edits_allowed_but_evaluation_edits_change_identity(self):
        fake = self.root / "repository"
        task = fake / "proxy_tasks/regbench"
        task.mkdir(parents=True)
        (fake / "proxy_tasks/common").mkdir()
        protocol = fake / "protocol.json"
        protocol.write_bytes(runner.PROTOCOL.read_bytes())
        (task / "models.py").write_text("hidden = 32\n")
        (task / "train.py").write_text("learning_rate = 0.001\n")
        with patch.object(runner, "ROOT", fake), patch.object(runner, "PROTOCOL", protocol):
            original = runner.evaluation_identity(self.args)
            (task / "models.py").write_text("hidden = 64\n")
            self.assertEqual(original, runner.evaluation_identity(self.args))
            (task / "train.py").write_text("learning_rate = 0.01\n")
            self.assertNotEqual(original, runner.evaluation_identity(self.args))


class DefaultPathTests(unittest.TestCase):
    def test_default_asset_paths_follow_common_paths(self):
        data, vendor = runner.proxy_paths.DATA_ROOT, runner.proxy_paths.THIRD_PARTY
        for task, model in [("g2p", "transformer"), ("zinc", "gine"), ("fflm", "lstm")]:
            args = runner.parser().parse_args(["check", "--task", task, "--model", model])
            spec = runner.load_protocol(task)[1]
            for label, p in runner.asset_paths(args, spec).items():
                root = vendor if label == "fflm_vendor" else data
                self.assertEqual(Path(p).parts[:len(root.parts)], root.parts, (task, label))


class RestoreTests(unittest.TestCase):
    def test_checks_entire_source_before_copying_and_refuses_overwrite(self):
        from scripts.restore_fflm_assets import restore
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, target = base / "source", base / "target"
            (source / "data").mkdir(parents=True)
            (source / "data/one").write_text("one")
            manifest = base / "manifest.json"
            entries = [{"path": "data/one", "sha256": runner.digest(source / "data/one")},
                       {"path": "data/two", "sha256": "missing"}]
            manifest.write_text(json.dumps(entries))
            with self.assertRaises(ValueError):
                restore(source, target, manifest)
            self.assertFalse(target.exists())
            manifest.write_text(json.dumps(entries[:1]))
            self.assertEqual(restore(source, target, manifest), 1)
            (target / "data/one").write_text("user change")
            with self.assertRaises(FileExistsError):
                restore(source, target, manifest)

    def test_restores_each_prefix_into_its_own_root(self):
        from scripts.restore_fflm_assets import restore
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            for rel in ["data/one", "vendor/pkg/two"]:
                (source / rel).parent.mkdir(parents=True, exist_ok=True)
                (source / rel).write_text(rel)
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps([{"path": rel, "sha256": runner.digest(source / rel)}
                                            for rel in ["data/one", "vendor/pkg/two"]]))
            roots = {"data": base / "data_root/fflm", "vendor": base / "vendor_root/fflm_vendor"}
            self.assertEqual(restore(source, roots, manifest), 2)
            self.assertEqual((roots["data"] / "one").read_text(), "data/one")
            self.assertEqual((roots["vendor"] / "pkg/two").read_text(), "vendor/pkg/two")


if __name__ == "__main__":
    unittest.main()
