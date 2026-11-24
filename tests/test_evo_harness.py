import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from proxy_tasks.common_set import plugins
from validation.precheck import check_configuration
from validation.report import make_report, normalized, rank
from validation.runner import load_protocol, PROTOCOL
from validation.util import digest, write_json

ROOT=Path(__file__).resolve().parents[1]
CONFIG=dict(d_model=32,num_heads=4,l_max=64,n_layers=2)
torch.set_num_threads(1)


class PluginTests(unittest.TestCase):
    def test_dynamic_load_hash_and_backbone_without_registry_edit(self):
        from proxy_tasks.common_set.backbone import Backbone
        reference=str(ROOT/'examples/candidate_mixer.py')+':CandidateMixer'
        info=plugins.describe(reference)
        self.assertRegex(info['model_id'],r'^cand_[0-9a-f]{8}$')
        before=PROTOCOL.read_bytes()
        with patch.dict(os.environ,{'PROXY_CANDIDATE_MIXER':reference}):
            model=Backbone(info['model_id'],32,2,4,64)
            x=torch.randn(2,9,32)
            self.assertEqual(tuple(model(x).shape),tuple(x.shape))
        self.assertEqual(before,PROTOCOL.read_bytes())

    def test_transformer_wrapper_has_identical_initial_weights_and_outputs(self):
        from proxy_tasks.common_set.mixers import CausalSelfAttention
        cls=plugins.load(str(ROOT/'examples/candidate_mixer.py')+':CandidateMixer')
        torch.manual_seed(12);a=CausalSelfAttention(32,4)
        torch.manual_seed(12);b=cls(32,4,64)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        x=torch.randn(2,17,32)
        self.assertTrue(torch.equal(a(x),b(x)))

    def test_hash_changes_with_content_or_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'m.py';f.write_text('class X: pass\n')
            a=plugins.describe(str(f)+':X')['sha256'];f.write_text('class X: pass\n# revised\n')
            self.assertNotEqual(a,plugins.describe(str(f)+':X')['sha256'])
            self.assertNotEqual(plugins.describe(str(f)+':Y')['sha256'],plugins.describe(str(f)+':X')['sha256'])

    def test_malformed_plugin_rejected(self):
        with self.assertRaises(ValueError):plugins.describe('no_class.py')


class PrecheckTests(unittest.TestCase):
    def test_good_transformer(self):
        cls=plugins.load(str(ROOT/'examples/candidate_mixer.py')+':CandidateMixer')
        self.assertTrue(check_configuration(cls,CONFIG)['passed'])

    def test_detects_parameter_excess_future_leak_and_nan_separately(self):
        from proxy_tasks.common_set.mixers import CausalSelfAttention
        class Over(CausalSelfAttention):
            def __init__(self,d_model,num_heads,l_max):
                super().__init__(d_model,num_heads);self.extra=torch.nn.Parameter(torch.zeros(d_model*1000))
        class Future(CausalSelfAttention):
            def __init__(self,d_model,num_heads,l_max):super().__init__(d_model,num_heads)
            def forward(self,x):return super().forward(x)+x.mean(1,keepdim=True)
        class Nan(Future):
            def forward(self,x):return super().forward(x)*float('nan')
        for cls,word in [(Over,'parameter_budget'),(Future,'causal'),(Nan,'finite')]:
            with self.subTest(cls=cls.__name__):
                result=check_configuration(cls,CONFIG)
                self.assertFalse(result['passed'])
                self.assertTrue(any(word in k and not v for k,v in result['checks'].items()))

    def test_tokenwise_control_is_causal_and_parameter_matched(self):
        class Tokenwise(torch.nn.Module):
            def __init__(self,d_model,num_heads,l_max):
                super().__init__();self.a=torch.nn.Linear(d_model,2*d_model,bias=False);self.b=torch.nn.Linear(2*d_model,d_model,bias=False)
            def forward(self,x):return self.b(self.a(x))
        self.assertTrue(check_configuration(Tokenwise,CONFIG)['passed'])


class ReportTests(unittest.TestCase):
    def make(self,profile='full',direction='maximize',fail=False):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);folder=Path(tmp.name)
        task='T1-R2-01';metric='accuracy';proto=digest(PROTOCOL)
        models={m:dict(mean=(i+1)/10,sample_std=.01,n_seeds=3,incomplete=False) for i,m in enumerate(load_protocol()['models'])}
        baseline=dict(schema_version=1,protocol_sha256=proto,normalization=dict(formula='(raw_score - baseline) / (ideal - baseline)',ideal='1 maximize, 0 minimize',source='fixture'),tasks={task:dict(name='test',tier=1,metric=metric,direction=direction,baseline=.2,ideal=1 if direction=='maximize' else 0,models=models,incomplete=False)})
        write_json(folder/'baseline.json',baseline)
        suite=dict(status='incomplete' if fail else 'done',profile=profile,precheck='passed',tasks=[task],seeds=[0,1,2],candidate=dict(model_id='cand_fixture'),configuration=dict(protocol_sha256=proto),runs=[])
        for seed,value in enumerate([.55,.6,.65]):
            suite['runs'].append(dict(task=task,seed=seed,status='timeout' if fail and seed==2 else 'done',profile=profile,evidence='measured' if profile=='full' else 'smoke_only',metrics={metric:value},wall_seconds=1))
        write_json(folder/'suite.json',suite)
        return folder,baseline,suite

    def test_independent_rank_normalization_and_sample_std(self):
        folder,b,s=self.make();r=make_report(folder,folder/'baseline.json')['tasks'][0]
        self.assertAlmostEqual(r['mean'],.6);self.assertAlmostEqual(r['sample_std'],.05)
        self.assertAlmostEqual(r['normalized_score'],.5)
        self.assertEqual(r['rank'],4) # .7,.8,.9 strictly better; .6 shares rank

    def test_minimize_direction(self):
        folder,b,s=self.make(direction='minimize');r=make_report(folder,folder/'baseline.json')['tasks'][0]
        self.assertAlmostEqual(r['normalized_score'],-2)
        self.assertEqual(r['rank'],6) # .1 ... .5 better, .6 tied

    def test_smoke_never_has_scores_or_ranks_even_if_metrics_present(self):
        folder,b,s=self.make(profile='smoke');r=make_report(folder,folder/'baseline.json')
        self.assertFalse(r['performance_comparable']);self.assertIsNone(r['summary']['overall_mean'])
        self.assertIsNone(r['tasks'][0]['mean']);self.assertIsNone(r['tasks'][0]['rank'])

    def test_partial_reference_and_candidate_are_provisional(self):
        folder,b,s=self.make(fail=True);b['tasks']['T1-R2-01']['incomplete']=True;write_json(folder/'baseline.json',b)
        r=make_report(folder,folder/'baseline.json')
        self.assertTrue(r['tasks'][0]['ranking_provisional']);self.assertEqual(len(r['timeouts']),1)
        self.assertEqual(r['tasks'][0]['completed_seeds'],[0,1])

    def test_missing_results_and_protocol_mismatch(self):
        folder,b,s=self.make();s['runs']=[];write_json(folder/'suite.json',s)
        r=make_report(folder,folder/'baseline.json');self.assertEqual(len(r['not_run']),3)
        b['protocol_sha256']='different';write_json(folder/'baseline.json',b)
        with self.assertRaises(ValueError):make_report(folder,folder/'baseline.json')

    def test_nonfinite_and_degenerate_normalization_rejected(self):
        with self.assertRaises(ValueError):normalized(float('nan'),0,1)
        with self.assertRaises(ValueError):normalized(.3,1,1)


class SuiteTests(unittest.TestCase):
    def args(self,out):
        return argparse.Namespace(mixer=str(ROOT/'examples/candidate_mixer.py')+':CandidateMixer',profile='smoke',seeds=[0],out=out,data_root=out/'data',tfl_root=out/'tfl',device='cpu',jobs=1,cascade=True,timeout=30,tasks=['T1-R2-01','T2-R1-11'],python=os.sys.executable,allow_busy_gpu=False,stop_starting_at=None)

    def test_bad_precheck_blocks_every_cell(self):
        from validation.suite import run_suite
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(Path(tmp))
            with patch('validation.precheck.run_precheck',return_value=dict(status='failed',candidate={})),patch('validation.suite._cell') as cell:
                self.assertEqual(run_suite(a),1);cell.assert_not_called()
            self.assertEqual(json.loads((a.out/'suite.json').read_text())['status'],'blocked')

    def test_cascade_skips_tier2_after_tier1_failure(self):
        from validation.suite import run_suite
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(Path(tmp))
            with patch('validation.precheck.run_precheck',return_value=dict(status='passed',harness_sha256='x')),patch('validation.util.fingerprint',return_value={'sha256':'fixture'}),patch('validation.suite._cell',return_value=dict(task='T1-R2-01',seed=0,status='timeout')) as cell:
                self.assertEqual(run_suite(a),1);self.assertEqual(cell.call_count,1)
            d=json.loads((a.out/'suite.json').read_text());self.assertEqual(d['runs'][1]['status'],'not_run')

    def test_completed_cell_is_reused_but_changed_identity_is_not(self):
        from validation.suite import _cell
        with tempfile.TemporaryDirectory() as tmp:
            a=self.args(Path(tmp));a.model_id='cand_fixture'
            cell=a.out/'cells/T1-R2-01-s0';cell.mkdir(parents=True)
            native=cell/'native.json';write_json(native,dict(status='done'))
            write_json(cell/'result.json',dict(status='done',suite_identity='fixed',native_result=str(native),native_result_sha256=digest(native)))
            with patch('validation.runner.run') as runner:
                self.assertEqual(_cell(a,'T1-R2-01',{},0,'fixed')['status'],'done');runner.assert_not_called()
            a.stop_starting_at='2000-01-01T00:00:00+00:00'
            self.assertEqual(_cell(a,'T1-R2-01',{},0,'changed')['status'],'not_run')


class FeedbackAndAdapterTests(unittest.TestCase):
    def test_smoke_feedback_does_not_fabricate_ranking(self):
        from scripts.make_feedback import make_feedback
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'report.json',dict(candidate={'model_id':'cand_mock'},profile='smoke',status='done',performance_comparable=False,failures=[],timeouts=[],not_run=[]))
            write_json(root/'precheck.json',dict(status='passed',tasks={}))
            text=make_feedback([root/'report.json'],root/'feedback.md').read_text()
            self.assertIn('Rank: unavailable',text)
            self.assertIn('Strong tasks: unavailable',text)

    def test_mock_generation_does_not_inspect_endpoint_or_make_network_calls(self):
        from integrations.evoscientist.adapter import generate
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);feedback=root/'feedback.md';feedback.write_text('Smoke only; no ranking.')
            with patch('integrations.evoscientist.adapter._endpoint') as endpoint,patch('urllib.request.build_opener') as network:
                result=generate(feedback,root/'gen1',3,'mock',30,root/'budget.json',root)
                endpoint.assert_not_called();network.assert_not_called()
            self.assertEqual(result['backend'],'mock');self.assertEqual(result['llm_calls'],0)
            self.assertEqual(len(result['files']),3)
            self.assertTrue(all('CandidateMixer =' in Path(p).read_text() for p in result['files']))
            self.assertFalse((root/'budget.json').exists())

    def test_call_budget_is_persistent_and_retries_are_counted(self):
        from integrations.evoscientist.adapter import consume_budget
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'budget.json'
            self.assertEqual(consume_budget(path,2),1);self.assertEqual(consume_budget(path,2),2)
            with self.assertRaises(ValueError):consume_budget(path,2)
            self.assertEqual(json.loads(path.read_text())['calls'],2)

if __name__=='__main__':unittest.main()
