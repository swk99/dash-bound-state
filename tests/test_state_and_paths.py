import sys
import types
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import fakeredis
from redis_state import RedisWindowWriter
sys.modules.setdefault('models', types.SimpleNamespace(DASHModelWrapper=object, load_dash_harness=None))
from validate_paths import check_paths
from analyze_latency import summarize

class Wrapper:
    model_type = 'Logistic'
    lookback_w = 30
    tau_conf = .5136591624213244
    def predict_hierarchical_timed(self, x, tau=None, force_stage2=False):
        assert tau is None
        return int(np.asarray(x).sum() > 0), .6, .7, .01, .01

class Tests(unittest.TestCase):
    def test_lua_order_bound_and_script_reload(self):
        client = fakeredis.FakeRedis()
        writer = RedisWindowWriter(client)
        for w in (10,30,60,120):
            key = f'window:{w}'
            for i in range(200):
                writer.push(key, np.full(6,i,dtype=np.float32).tobytes(),w)
            self.assertEqual(client.llen(key),w)
            self.assertEqual(np.frombuffer(client.lindex(key,0),dtype=np.float32)[0],199)
            self.assertEqual(np.frombuffer(client.lindex(key,-1),dtype=np.float32)[0],200-w)
        before = client.lrange('window:10',0,-1)
        for bad in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                writer.push('window:10',b'bad',bad)
        self.assertEqual(before,client.lrange('window:10',0,-1))
        client.script_flush()
        self.assertEqual(writer.push('window:10',b'new',10),10)
    def test_label_agreement_and_cleanup(self):
        client = fakeredis.FakeRedis()
        client.set('unrelated','keep')
        data = np.random.default_rng(7).normal(size=(150,6)).astype(np.float32)
        self.assertEqual(check_paths(data,Wrapper(),client)['label_agreement'],1)
        self.assertEqual(client.keys(),[b'unrelated'])
    def test_statistics(self):
        rows = summarize(np.arange(1.,101.),[10,100],resamples=200)
        self.assertEqual(rows[-2]['value'],.1)
        self.assertEqual(rows[-1]['value'],1)
        self.assertTrue(all(np.isfinite(r['ci_low']) for r in rows[:4]))
        self.assertEqual(summarize([1,1,1],[1])[0]['status'],'constant_sample')

class ExperimentTests(unittest.TestCase):
    def test_e2_sweep_and_gate_streams(self):
        import tempfile
        from unittest.mock import patch
        sys.modules.setdefault('tooling', types.SimpleNamespace(MinioHandler=object))
        import benchmark_2 as bench
        from engine import ProposedStatefulEngine
        class MeasuringFake(fakeredis.FakeRedis):
            def memory_usage(self, key, samples=0):
                # Test harness only: logical payload, not Redis allocator measurement.
                return sum(map(len, self.lrange(key,0,-1)))
        client = MeasuringFake()
        client.set('unrelated','keep')
        data = np.random.default_rng(9).normal(size=(20,6)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp, patch.object(bench,'rds',client), \
             patch.object(bench,'RESULTS_DIR',Path(tmp)), \
             patch.object(bench.psutil,'Process',lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=1024))), \
             patch.object(bench,'_mk_engine',lambda name: ProposedStatefulEngine(Wrapper(),client)), \
             patch.object(bench,'WARMUP',3), patch.object(bench,'N_EVAL',10):
            df = bench.run_memory_vs_nw('stub',data,n_list=(5,20),w_list=(3,10),repeats=2)
            self.assertEqual(len(df),16)
            self.assertTrue((df.redis_list_len == np.minimum(df.N,df.W)).all())
            self.assertTrue((df.redis_mem_bytes == df.redis_list_len*24).all())
            result = bench.run_gate_ablation('stub',data)
            self.assertEqual(result['gate_firing_rate'],1)
        self.assertEqual(client.keys(),[b'unrelated'])
