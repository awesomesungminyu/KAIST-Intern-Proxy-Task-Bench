import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from validation.runner import build_command, data_base, load_protocol
from validation.prepare import prepare_task


class PortableRuntimeTests(unittest.TestCase):
    def test_dataset_root_has_the_same_meaning_for_every_adapter(self):
        from proxy_tasks.final16.catalog import build_part
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task, spec in load_protocol()['tasks'].items():
                args = argparse.Namespace(task=task, data_root=root, python=sys.executable,
                                          model='cs_transformer', profile='smoke')
                self.assertEqual(data_base(args, spec), root / task)
                if spec['adapter'] == 'part':
                    command, cwd, env = build_command(args, spec, 0, root / 'out')
                    with patch.dict('os.environ', {'PROXY_TASK_DATA_DIR': env['PROXY_TASK_DATA_DIR']}):
                        self.assertEqual(build_part(task).data_dir, root / task)

    def test_preparation_preserves_sources_and_reuses_completed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source' / 't2d_protein/stability'
            source.mkdir(parents=True)
            for name in ['train.csv', 'valid.csv']:
                (source / name).write_text('aa_seq,label\nACD,1.0\n')
            destination = root / 'data/T2-R1-04'
            spec = load_protocol('T2-R1-04')[1]
            self.assertEqual(prepare_task(root / 'source', destination, 'T2-R1-04', spec), 'prepared')
            self.assertEqual((source / 'train.csv').read_bytes(), (destination / 'train.csv').read_bytes())
            self.assertEqual(prepare_task(root / 'source', destination, 'T2-R1-04', spec), 'already prepared')

    def test_preparation_failure_leaves_no_partial_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'train.csv').write_text('aa_seq,label\nACD,1.0\n')
            destination = root / 'data/T2-R1-04'
            with self.assertRaises(FileNotFoundError):
                prepare_task(source, destination, 'T2-R1-04', load_protocol('T2-R1-04')[1])
            self.assertFalse(destination.exists())

    def test_data_availability_needs_no_ml_packages(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-S', '-m', 'validation', 'data', '--task', 'T1-R2-02'],
                                cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [dict(task='T1-R2-02', ready=True, missing=[])])


if __name__ == '__main__':
    unittest.main()
