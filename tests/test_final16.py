import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from validation import runner
from proxy_tasks.final16.catalog import build_part

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'T1-R1-05', 'T1-R2-01', 'T1-R2-02', 'T1-R0-07', 'T1-R0-04',
    'T1-R2-03', 'T1-R2-04', 'T1-R2-05', 'T2-R1-11', 'T2-R3-05',
    'T2-R4-02', 'T2-R2-07', 'T2-R1-04', 'T2-R1-09', 'T2-R1-08', 'T2-R2-02',
}
torch.set_num_threads(1)


class FinalSelectionTests(unittest.TestCase):
    def test_catalog_is_exact_and_works_without_site_packages(self):
        result = subprocess.run([sys.executable, '-S', '-m', 'validation', 'list'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        protocol = json.loads(result.stdout)
        self.assertEqual(set(protocol['tasks']), EXPECTED)
        for tier in [1, 2]:
            for label in ['win', 'lose']:
                self.assertEqual(sum(s['tier'] == tier and s['transformer_label'] == label
                                     for s in protocol['tasks'].values()), 4)

    def test_excluded_task_is_rejected(self):
        result = subprocess.run([sys.executable, '-m', 'validation', 'check', '--task', 'zinc',
                                 '--model', 'cs_transformer'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('invalid choice', result.stderr)

    def test_every_part_imports_with_matching_metric(self):
        for cid, spec in runner.load_protocol()['tasks'].items():
            if spec['adapter'] != 'part':
                continue
            with self.subTest(task=cid):
                part = build_part(cid)
                self.assertEqual(part.score_name, spec['metric'])
                self.assertEqual(part.steps, spec['steps'])
                self.assertGreater(part.steps, 3)

    def test_native_command_imports_correct_package(self):
        args = argparse.Namespace(task='T1-R0-07', model='cs_transformer', python=sys.executable,
                                  profile='smoke', data_root=None)
        with tempfile.TemporaryDirectory() as tmp:
            command, cwd, _ = runner.build_command(args, runner.load_protocol(args.task)[1], 0, Path(tmp))
        self.assertEqual(command[command.index('-m') + 1], 'T1_R0_07.train')
        result = subprocess.run([*command[:3], '--help'], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('cs_transformer', result.stdout)

    def test_missing_assets_fail_without_download(self):
        args = argparse.Namespace(task='T2-R4-02', data_root=Path('/does-not-exist'))
        spec = runner.load_protocol(args.task)[1]
        with self.assertRaises(FileNotFoundError):
            runner.fingerprint(runner.asset_paths(args, spec))

    def test_smoke_results_cannot_be_compared_as_performance(self):
        from validation.util import compare
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'summary.json'
            path.write_text(json.dumps(dict(status='done', runs=[dict(seed=0, status='done', profile='smoke')], requested_seeds=[0])))
            with self.assertRaises(ValueError):
                compare(path, path)


class SyntheticExecutionTests(unittest.TestCase):
    def test_selected_generated_tasks_have_finite_forward_and_backward(self):
        for cid in ['T1-R2-01', 'T1-R2-02', 'T1-R2-03', 'T1-R2-04', 'T1-R2-05']:
            with self.subTest(task=cid):
                part = build_part(cid)
                part.batch_size = 2
                part.d_model, part.n_layers = 32, 1
                part.load('cpu')
                batch = part.sample(torch.Generator().manual_seed(7))
                model = part.build('transformer')
                output = model(*part.inputs(batch))
                loss = part.loss(output, batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_mad_cache_loader_and_reconstruction_shape(self):
        from proxy_tasks.final16 import synth
        with tempfile.TemporaryDirectory() as tmp, patch.object(synth, 'DATA', Path(tmp) / 'smoke_v21'):
            part = build_part('T1-R1-05')
            part.path.mkdir(parents=True)
            np.save(part.path / 'inputs.npy', np.random.default_rng(7).integers(0, 16, (12800, 32), dtype=np.int64))
            part.load('cpu')
            self.assertEqual(part.xtr.shape[0], 11520)
            self.assertEqual(part.xva.shape[0], 1280)
            part.batch_size, part.d_model, part.n_layers = 2, 32, 1
            model = part.build('transformer')
            batch = part.sample(torch.Generator().manual_seed(0))
            output = model(*part.inputs(batch))
            self.assertEqual(tuple(output.shape), (2, 32, 16))
            self.assertTrue(torch.isfinite(part.loss(output, batch)))

    def test_modular_arithmetic_labels(self):
        part = build_part('T1-R2-03')
        tokens, labels = part.make(torch.Generator().manual_seed(7), 20, 10)
        operators = {5: '+', 6: '-', 7: '*'}
        for row, label in zip(tokens.tolist(), labels.tolist()):
            expression = ''.join(str(value) if value < 5 else operators[value] for value in row[:-1])
            self.assertEqual(eval(expression, {'__builtins__': {}}) % 5, label)

    def test_cycle_navigation_labels(self):
        part = build_part('T1-R2-04')
        tokens, labels = part.make(torch.Generator().manual_seed(7), 20, 10)
        expected = (tokens[:, :-1] - 1).sum(1) % 5
        self.assertTrue(torch.equal(labels, expected))


class MixerContractTests(unittest.TestCase):
    def test_nine_mixers_are_causal_and_have_finite_gradients(self):
        from proxy_tasks.common_set import NAMES, make_mixer, mixer_width
        self.assertEqual(len(NAMES), 9)
        for name in NAMES:
            with self.subTest(mixer=name):
                torch.manual_seed(0)
                mixer = make_mixer(name, 32, 4, 64, mixer_width(name, 32, 4, 64))
                x = torch.randn(2, 24, 32, requires_grad=True)
                y = mixer(x)
                self.assertEqual(tuple(y.shape), (2, 24, 32))
                y.square().mean().backward()
                grads = [p.grad for p in mixer.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                changed = x.detach().clone()
                changed[:, 12:] += 1
                with torch.no_grad():
                    leak = (mixer(x.detach())[:, :12] - mixer(changed)[:, :12]).abs().max().item()
                self.assertLess(leak, 1e-5)


if __name__ == '__main__':
    unittest.main()
