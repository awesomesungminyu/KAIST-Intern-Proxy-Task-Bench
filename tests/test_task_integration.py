"""Protocol/adapter contracts without torch or actual training."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validation import runner
import test_validation as fixtures

measured = fixtures.measured

TASKS = {'regbench','g2p','zinc','parity','tomita5','tomita6','fflm','ett','scifar','sc_raw','mqar','mad_tsc','tier0','iclreg','jelassi_copy'}


class RegistryTests(unittest.TestCase):
    def test_protocol_names_match_registries_without_importing_torch(self):
        catalog = json.loads(runner.PROTOCOL.read_text())['tasks']
        self.assertEqual(set(catalog), TASKS)
        for task, spec in catalog.items():
            if spec['adapter'] == 'formal':
                models = runner._load_file('formal_registry', runner.ROOT/'proxy_tasks/formal_languages/models.py').TFL_MODELS
            else:
                tree = ast.parse((runner.ROOT/'proxy_tasks'/task/'models.py').read_text())
                models = [d.args[0].value for node in ast.walk(tree)
                          if isinstance(node, (ast.FunctionDef, ast.ClassDef)) for d in node.decorator_list
                          if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'register_model']
            # cs_ names are protocol v2's common set (tests/test_common_set.py checks them against v2.json).
            models = [m for m in models if not m.startswith('cs_')]
            self.assertEqual(set(spec['models']), set(models), task)
            self.assertTrue(spec['runtime']['torch'])

    def test_paths_are_callable_for_every_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runner.proxy_paths, 'DATA_ROOT', root/'data'), patch.object(runner.proxy_paths, 'OUTPUT_ROOT', root/'output'):
                for task in TASKS:
                    self.assertTrue(runner.proxy_paths.task_data(task).is_dir())
                    self.assertTrue(runner.proxy_paths.task_output(task, 'candidate', 0).is_dir())


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def args(self, task, model):
        return runner.parser().parse_args(['run', '--task', task, '--model', model, '--out', str(self.root/'out'), '--data-root', str(self.root/'data')])

    def test_native_data_and_upstream_overrides_are_forwarded(self):
        for task, model in [('ett','linear'), ('scifar','lstm'), ('sc_raw','lstm'), ('mqar','gqa2'), ('mad_tsc','mh_attention_swiglu')]:
            args = self.args(task, model)
            spec = runner.load_protocol(task)[1]
            cmd, cwd, _ = runner.build_command(args, spec, 12, self.root/task)
            self.assertEqual(cwd, runner.ROOT/'proxy_tasks')
            self.assertEqual(cmd[cmd.index('--seed')+1], '12')
            if task == 'mqar':
                self.assertEqual(cmd[cmd.index('--zoology_root')+1], str(args.zoology_root))
            else:
                self.assertEqual(cmd[cmd.index('--data_dir')+1], str(args.data_root))
            self.assertNotIn('--final_test', cmd)
            self.assertEqual(cmd[0], '/data/envs/hexa-mad/bin/python' if task in {'mqar','mad_tsc'} else runner.sys.executable)
            args.python = '/explicit/python'
            self.assertEqual(runner.build_command(args,spec,12,self.root/task)[0][0], '/explicit/python')

    def test_formal_stage_uses_copied_task_data(self):
        args = self.args('parity', 'lstm')
        cmd, stage, _ = runner.build_command(args, runner.load_protocol('parity')[1], 0, self.root/'run')
        self.assertEqual((stage/'data/Parity').resolve(), args.data_root)
        self.assertEqual((stage/'src').resolve(), args.tfl_root/'src')

    def test_all_asset_paths_respect_data_root(self):
        for task in TASKS - {'fflm'}:
            spec = runner.load_protocol(task)[1]
            args = self.args(task, spec['models'][0])
            for asset in runner.asset_paths(args, spec).values():
                self.assertTrue(asset.is_relative_to(args.data_root), (task,asset))

    def extract(self, task, raw, *, model=None, profile='full'):
        spec = runner.load_protocol(task)[1]
        model = model or spec['models'][0]
        p = self.root/f'{model}_seed0/history.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(raw))
        return runner.extract_metrics(task, spec, self.root, model, 0, profile)[0]

    def test_new_native_metrics_and_no_training_naive(self):
        for task, key, count, metric in [('ett','epoch',20,'val_mse'),('scifar','epoch',10,'val_accuracy'),('sc_raw','step',2048,'val_accuracy'),('mqar','epoch',30,'best_valid_accuracy'),('mad_tsc','epoch',200,'val_accuracy')]:
            raw = {'history':[{key:count}],metric:0.5,'test':None}
            self.assertEqual(self.extract(task,raw),{metric:0.5})
            raw['history'][-1][key] = count + 1
            with self.assertRaises(ValueError): self.extract(task,raw)
        self.assertEqual(self.extract('ett',{'history':[{'epoch':0}], 'val_mse':0.6},model='naive'),{'val_mse':0.6})

    def test_naive_budget_is_zero_and_not_equal_to_trained_ett(self):
        spec = runner.load_protocol('ett')[1]
        self.assertEqual(runner.training_budget(spec,'full','naive')['maximum_counter'],0)
        self.assertEqual(runner.training_budget(spec,'full','linear')['maximum_counter'],20)

    def test_early_stop_requires_actual_stop_criterion(self):
        raw={'history':[{'epoch':1,'valid_accuracy':0.98}], 'best_valid_accuracy':0.98, 'stop_reason':'early_stopping'}
        with self.assertRaises(ValueError): self.extract('mqar',raw)
        raw['history'][0]['valid_accuracy'] = 0.99
        raw['best_valid_accuracy'] = 0.99
        self.assertEqual(self.extract('mqar',raw),{'best_valid_accuracy':0.99})
        raw={'history':[{'epoch':i+1,'val_mse':v} for i,v in enumerate([0.3,0.4,0.5,0.5])], 'val_mse':0.3,'stop_reason':'early_stopping'}
        self.assertEqual(self.extract('ett',raw),{'val_mse':0.3})
        raw['history'][-1]['val_mse'] = 0.2
        with self.assertRaises(ValueError): self.extract('ett',raw)
        raw={'history':[{'epoch':5}], 'val_accuracy':1.0, 'development_peak_accuracy':0.998, 'stop_reason':'accuracy_threshold'}
        with self.assertRaises(ValueError): self.extract('mad_tsc',raw)
        raw['development_peak_accuracy']=0.999
        self.assertEqual(self.extract('mad_tsc',raw),{'val_accuracy':1.0})

    def test_new_tasks_reject_final_test_leakage(self):
        for task, count, metric in [('ett',20,'val_mse'),('scifar',10,'val_accuracy'),('sc_raw',2048,'val_accuracy')]:
            raw={'history':[{'step' if task=='sc_raw' else 'epoch':count}],metric:0.5,'test':{'accuracy':0.9}}
            with self.assertRaises(ValueError): self.extract(task,raw)

    def test_tier0_seed_budget_and_nonfinite_values(self):
        spec=runner.load_protocol('tier0')[1]
        args=self.args('tier0','lstm')
        command, _, _=runner.build_command(args,spec,7,self.root/'run')
        self.assertEqual(command[command.index('--seed-start')+1],'7')
        self.assertEqual(command[command.index('--seeds')+1],'1')
        raw={'seeds':1,'seed_start':7,'config':{'d_model':32,'n_layers':2,'seq_len':64,'batch':4},'probe_len':8,'results':{'lstm':{'seeds':[{m['name']:0.1 for m in spec['metrics']}]}}}
        p=self.root/'tier0_results.json'
        p.write_text(json.dumps(raw))
        self.assertEqual(len(runner.extract_metrics('tier0',spec,self.root,'lstm',7,'smoke')[0]),8)
        with self.assertRaises(ValueError): runner.extract_metrics('tier0',spec,self.root,'lstm',8,'smoke')
        raw['results']['lstm']['seeds'][0]['snip']=float('nan')
        p.write_text(json.dumps(raw))
        with self.assertRaises(ValueError): runner.extract_metrics('tier0',spec,self.root,'lstm',7,'smoke')

    def test_mad_final_validate_does_not_count_as_an_extra_training_epoch(self):
        data = runner._load_file('mad_data_contract', runner.ROOT/'proxy_tasks/mad_tsc/data.py')
        p = self.root/'metrics.csv'
        p.write_text('epoch,train/Loss_epoch,test/Accuracy_epoch\n0,1.0,0.5\n1,0.5,0.7\n2,,1.0\n2,,0.99\n')
        self.assertEqual(data.read_progress(p), (2,0.7))
        p.write_text('epoch,train/Loss_epoch,test/Accuracy_epoch\n0,,0.5\n')
        with self.assertRaises(ValueError): data.read_progress(p)

    def test_runtime_version_is_task_specific_and_exact(self):
        args=self.args('mqar','gqa2'); spec=runner.load_protocol('mqar')[1]
        commit=(runner.ROOT/'third_party/zoology.commit').read_text().split()[-1]
        def output(cmd, *a, **kw):
            if cmd[0]=='git': return commit
            if 'torch' in cmd[-1]: return json.dumps({'torch':version,'packages':{}})
            return '{}'
        with patch.object(runner,'command_output',side_effect=output), patch.object(runner,'evaluation_identity',return_value='hash'), patch.object(runner,'source_identity',return_value={}), patch.dict(runner.os.environ, {'CUDA_VISIBLE_DEVICES':'0'}):
            version='2.2.2+cu121'
            self.assertTrue(runner.preflight(args,spec)['ready'])
            version='2.5.1+cu124'
            with self.assertRaises(ValueError): runner.preflight(args,spec)
            version='2.2.20'
            with self.assertRaises(ValueError): runner.preflight(args,spec)


class MultiMetricTests(unittest.TestCase):
    setUp = fixtures.CompareTests.setUp
    compare = fixtures.CompareTests.compare
    def test_tier0_compare_is_descriptive_and_checks_definitions(self):
        defs=runner.load_protocol('tier0')[1]['metrics']
        a,b=measured(),measured()
        for r in [a,b]:
            r.update(task='tier0',metric={'name':'multi_metric','direction':'descriptive','split':'random_tokens'},metric_definitions=copy.deepcopy(defs),metrics={m['name']:1.0 for m in defs})
        b['metrics']['snip']=2.0
        out=self.compare([a],[b])
        self.assertNotIn('mean_improvement',out)
        self.assertEqual(out['metrics']['snip']['candidate_mean'],2.0)
        b['metric_definitions'][0]['role']='control'
        with self.assertRaises(ValueError): self.compare([a],[b])
