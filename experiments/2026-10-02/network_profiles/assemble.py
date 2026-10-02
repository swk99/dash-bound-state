"""Assemble dash_paper_results.csv (single header, section-ordered, long format) + validation."""
import os, sys, glob, json, hashlib, platform, re, warnings, io, contextlib, datetime as dt
import numpy as np, pandas as pd
REPO = '/home/claude/dash/dash-bound-state-main'; OUT = '/home/claude/jss_run3'; CK = f'{OUT}/ckpt'
sys.path.insert(0, REPO)
import config as cfg
RUN = 'dash3_20261002T1122Z'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CODE = hashlib.sha256(''.join(sha(f) for f in sorted(glob.glob(f'{REPO}/*.py'))).encode()).hexdigest()
CODE_VERSION = f'nogit-sha16:{CODE[:16]}'
SCRIPTS = {os.path.basename(f): sha(f) for f in [f'{OUT}/bench.py', f'{OUT}/driver.py', f'{OUT}/delay_proxy.py', __file__]}
SCRIPT_SHA = hashlib.sha256(''.join(f'{k}:{v}' for k, v in sorted(SCRIPTS.items())).encode()).hexdigest()
FS = sorted(glob.glob('/home/claude/pq/btcusdt/flush_*.parquet'), key=lambda f: int(re.findall(r'flush_(\d+)', f)[0]))
DFALL = pd.concat([pd.read_parquet(f) for f in FS], ignore_index=True)
XALL = np.ascontiguousarray(DFALL[cfg.FEATURE_COLS].to_numpy(np.float32))
INPUT_SHA = hashlib.sha256(XALL.tobytes()).hexdigest()
MANIFEST_SHA = hashlib.sha256(''.join(f'{os.path.basename(f)}:{sha(f)}\n' for f in FS).encode()).hexdigest()
ART = {n: sha(os.path.join(REPO, n)) for n in ['s1_xgb_H30_a2_L94.json', 's2_xgb_H30_a2_L94.json', 's1_rf_H30_a2_L94.pkl', 's2_rf_H30_a2_L94.pkl',
       'artifacts/s1_lr_H30_a2_L94.pkl', 's2_lr_H30_a2_L94.pkl', 'scaler_H30_a2_L94.pkl', 'thresholds_H30_a2_L94.json']}
MART = {'XGBoost': ('s1_xgb_H30_a2_L94.json', 's2_xgb_H30_a2_L94.json'), 'RandomForest': ('s1_rf_H30_a2_L94.pkl', 's2_rf_H30_a2_L94.pkl'),
        'Logistic': ('artifacts/s1_lr_H30_a2_L94.pkl', 's2_lr_H30_a2_L94.pkl')}
SC_SHA, TH_SHA = ART['scaler_H30_a2_L94.pkl'], ART['thresholds_H30_a2_L94.json']
PROFILES = {'P0': (0, 0), 'P1': (1, 0), 'P2': (1, 0.5), 'P3': (5, 0), 'P4': (5, 2)}
QM = 'numpy.quantile method=linear (Hyndman-Fan type 7)'
DEADLINES = [1, 5, 10, 20, 50, 100, 1000]
COLS = """section record_type experiment run_id provenance timestamp_utc code_version script_sha256 source_file input_sha256
data_origin workload_type symbol logical_stream_id stream_offset stream_local_tick request_sequence model engine policy update_method repeat seed W N K client_concurrency d dtype warmup_requests measured_requests
network_profile impairment_backend redis_topology delay_oneway_ms jitter_parameter_ms jitter_distribution impairment_direction network_seed network_correlation ping_rtt_ms
redis_key redis_key_bytes redis_encoding redis_list_len redis_mem_bytes logical_payload_bytes rss_bytes
tau_s1 tau_s2 stage2_fired prediction reference_prediction label_match artifact_sha256 scaler_sha256 threshold_sha256
latency_ms state_ms stage1_ms stage2_ms deadline_ms success error_type timeout_ms metric value text_value unit n success_count failure_count
aggregation quantile_method ci_low ci_high ci_method bootstrap_unit bootstrap_B block_length confidence_level inference_status
status reason notes""".split()
NOW = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
BASE = dict(run_id=RUN, code_version=CODE_VERSION, script_sha256=SCRIPT_SHA, input_sha256=INPUT_SHA)
ORIGIN = 'btcusdt_real_1s'
NET = dict(impairment_backend='py_delay_proxy', redis_topology='same_host_loopback_via_proxy', impairment_direction='both', network_correlation='none_sampler_fifo', network_seed='rng_20260928_conn_dir')
CODEBOOK = {'data_origin=btcusdt_real_1s': 'real btcusdt 1s-bar features from MinIO flush parquet (user-provided export); not newly collected',
  'impairment_backend=py_delay_proxy': 'experiment-only Python TCP delay proxy (delay_proxy.py) on 127.0.0.1:26379 -> Redis 127.0.0.1:6379; used for ALL profiles incl. P0; kernel lacks sch_netem',
  'redis_topology=same_host_loopback_via_proxy': 'client, proxy and Redis on the same host over loopback TCP',
  'impairment_direction=both': 'client->Redis and Redis->client, delays sampled independently per received chunk',
  'network_correlation=none_sampler_fifo': 'no correlation in sampler; FIFO release (no reordering) can induce positive serial correlation',
  'network_seed=rng_20260928_conn_dir': 'numpy default_rng([20260928, connection_id, direction])',
  'jitter_distribution=normal_clip0_fifo': 'Normal(mean=delay_oneway_ms, sd=jitter_parameter_ms) per chunk, negative values truncated to 0 ms, FIFO release',
  'jitter_distribution=none': 'fixed delay (or none for P0)',
  'raw_rows_artifact_sha': 'artifact/scaler/threshold SHA256 are recorded once per model in 02_artifact_checks and 00_run_manifest; left blank on raw latency rows to keep file size tractable',
  'notes=order=': 'engine execution order within the repeat (seeded permutation)'}
def netcols(p):
    d, j = PROFILES[p]
    return dict(network_profile=p, delay_oneway_ms=d, jitter_parameter_ms=j,
                jitter_distribution=('none' if j == 0 else 'normal_clip0_fifo'), **NET)
SECTIONS = []
def sec(name, rows, **common):
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    for k, v in {**BASE, 'timestamp_utc': NOW, **common}.items():
        if k not in df: df[k] = v
    df['section'] = name; SECTIONS.append(df)
def meta(metric, val, unit=None, **kw):
    d = dict(record_type='metadata', metric=metric, unit=unit, **kw)
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool): d['value'] = float(val)
    else: d['text_value'] = str(val)
    return d
# ================= 00 manifest =================
sec_rows = [meta('run_id', RUN), meta('time_unit', 'ms (latency), bytes (memory)'), meta('timestamp_timezone', 'UTC, ISO 8601'),
    meta('symbol', 'btcusdt'), meta('data_files', ';'.join(os.path.basename(f) for f in FS)), meta('data_file_manifest_sha256', MANIFEST_SHA),
    meta('data_total_rows', len(DFALL), 'rows'), meta('data_timestamp_min', str(DFALL.sec.min())), meta('data_timestamp_max', str(DFALL.sec.max())),
    meta('data_timestamp_monotonic_increasing', bool(DFALL.sec.is_monotonic_increasing)),
    meta('data_nan_count_features', int(np.isnan(XALL).sum()), 'count'), meta('data_inf_count_features', int(np.isinf(XALL).sum()), 'count'),
    meta('feature_order', '|'.join(cfg.FEATURE_COLS)), meta('dtype', 'float32'), meta('input_array_sha256', INPUT_SHA),
    meta('seed', 20260928), meta('seed_rule', 'engine order per repeat: default_rng([20260928, repeat, K, profile_index]).permutation; gating per-tick policy order: default_rng([20260928, repeat, 7, profile_digit]); Lua cost order: default_rng([20260928, repeat, 99]); repeat seed column = 20260928+repeat (label only)'),
    meta('workload_definition', 'controlled replication: K resident logical streams, each a deterministic contiguous non-overlapping segment of the real btcusdt series (offset j*L, L=W+ceil(5200/K)); single client, closed loop, client_concurrency=1; round-robin across streams; not K real symbols; not multi-client concurrency; not open-loop arrival load'),
    meta('warmup_rule', 'per stream W=30 fill requests (stream-major) + 200 global round-robin warmup requests; excluded from samples'),
    meta('measured_requests_per_condition_engine_repeat', 5000, 'requests'),
    meta('planned_primary_matrix', 'XGBoost; W=30; K in {1,10,100}; profiles P0,P1,P2; engines A,B1,B2; 5 repeats'),
    meta('secondary_selection_preregistered', 'only XGBoost K=100 P3 selected before results (pilot-based time estimate); P4, K=500 P0/P2, Logistic/RF K=1/100 P0/P2 not_run (budget)'),
    meta('profile_execution_order', 'P0 then P1 then P2 (primary, all K inside each profile, K order 1,10,100), then Lua cost, gating P0,P2, then secondary P3'),
    meta('scripts_sha256', json.dumps(SCRIPTS))] + [meta('codebook', v, source_file=k) for k, v in CODEBOOK.items()]
for f, h in ART.items(): sec_rows.append(meta('artifact_sha256', h, source_file=f))
sec('00_run_manifest', sec_rows, experiment='manifest', provenance='this_run')
# ================= 01 environment =================
import sklearn, xgboost, redis as rpy, torch
r = rpy.Redis(); info = r.info('server'); mem = r.info('memory')
cpu = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t')
import psutil
env = [('python', platform.python_version()), ('python_readme_target', '3.11 (not met)'), ('redis_server', info['redis_version']), ('redis_py', rpy.__version__),
       ('numpy', np.__version__), ('pandas', pd.__version__), ('sklearn', sklearn.__version__), ('xgboost', xgboost.__version__), ('torch', torch.__version__),
       ('os', platform.platform()), ('cpu_model', cpu), ('vcpu', os.cpu_count()), ('ram_total_bytes', psutil.virtual_memory().total),
       ('shared_cloud_host', 'yes: ephemeral cloud container (Firecracker VM), co-tenancy unknown; not the original experiment PC'),
       ('redis_location', 'same host, loopback; Redis 7.0.15 started inside this container for testing (not the original experiment Redis)'),
       ('redis_maxmemory_policy', r.config_get('maxmemory-policy')['maxmemory-policy']), ('redis_config_changed', 'no'),
       ('redis_client_pool', 'redis-py default ConnectionPool (max_connections=2**31), one connection used sequentially; socket_timeout=1.0s, socket_connect_timeout=1.0s; new client per profile'),
       ('model_threads', 'not set by experiment: XGBoost default nthread (all 2 vCPU; observed driver CPU ~150%); sklearn RF/LR as stored in artifacts; OMP_NUM_THREADS unset'),
       ('omp_num_threads_env', os.environ.get('OMP_NUM_THREADS', 'unset')),
       ('network_impairment_tool', f'delay_proxy.py sha256={SCRIPTS["delay_proxy.py"]}; netem unavailable (CONFIG_NET_SCH_NETEM not set); Toxiproxy v2.12.0 rejected (integer-ms latency, uniform jitter)'),
       ('proxy_cpu_note', 'proxy runs as separate Python process on the same 2 vCPU; adds its own processing/timer delay (~0.5 ms per configured 1 ms each way observed via PING)'),
       ('timeout_ms', 1000), ('W', 30), ('d', 6), ('repeats', 5)]
sec('01_environment', [meta(k, v) for k, v in env], experiment='environment', provenance='this_run')
# ================= 02 artifact checks =================
from models import load_dash_harness
TH = json.load(open(f'{REPO}/thresholds_H30_a2_L94.json'))
REF = {'XGBoost': 0.5002538561820984, 'RandomForest': 0.5032152391133569, 'Logistic': 0.5136591624213244}
KEYMAP = {'XGBoost': 'xgb', 'RandomForest': 'rf', 'Logistic': 'lr'}
rows = []
for m in REF:
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(buf):
        warnings.simplefilter('always')
        try: h = load_dash_harness(m); err = None
        except Exception as e: h = None; err = repr(e)
    vw = [str(x.message)[:200] for x in ws if 'version' in str(x.message).lower() or 'Inconsistent' in type(x.message).__name__]
    a1, a2 = MART[m]
    common = dict(model=m, artifact_sha256=f'{ART[a1]}|{ART[a2]}', scaler_sha256=SC_SHA, threshold_sha256=TH_SHA, record_type='check')
    if h is None:
        rows.append(dict(metric='harness_loaded', text_value='false', status='failed', reason=err, **common)); continue
    x = XALL[100]
    scaled = h._apply_scaler(x) if hasattr(h, '_apply_scaler') else None
    exp_scaled = (x - h._mu) / h._sd if h._mu is not None else None
    js1 = TH['models'][KEYMAP[m]]['s1']['thr']; js2 = TH['models'][KEYMAP[m]]['s2']['thr']
    chk = [('stage1_artifact_loaded', h.s1 is not None, a1), ('stage2_artifact_loaded', h.s2 is not None, a2),
           ('scaler_loaded', h._mu is not None and h._sd is not None, 'scaler_H30_a2_L94.pkl'),
           ('scaler_applied_matches_(x-mu)/sd', scaled is not None and np.allclose(scaled, exp_scaled), 'scaler_H30_a2_L94.pkl'),
           ('tau_s1_equals_json', h.tau_conf == js1, 'thresholds_H30_a2_L94.json'),
           ('tau_s1_equals_reference_value', abs(h.tau_conf - REF[m]) < 1e-15, 'thresholds_H30_a2_L94.json'),
           ('tau_s2_equals_json', h.tau_s2 == js2, 'thresholds_H30_a2_L94.json'),
           ('cfg_TAU_CONF_not_overriding', h.tau_conf != cfg.TAU_CONF, 'config.py'),
           ('no_version_warning', len(vw) == 0, '')]
    for name, ok, src in chk:
        rows.append(dict(metric=name, text_value=str(bool(ok)).lower(), status='pass' if ok else 'fail', source_file=src, tau_s1=h.tau_conf, tau_s2=h.tau_s2,
                         notes=(f'cfg.TAU_CONF={cfg.TAU_CONF} (fallback only)' if 'cfg' in name else ('; '.join(vw) if name == 'no_version_warning' and vw else None)), **common))
    rows.append(dict(metric='tau_s1_applied', value=h.tau_conf, status='recorded', **common)); rows.append(dict(metric='tau_s2_applied_json', value=h.tau_s2, status='recorded', **common))
    rows.append(dict(metric='loader_stdout', text_value=buf.getvalue().strip().replace('\n', ' / '), status='recorded', **common))
sec('02_artifact_checks', rows, experiment='artifact_checks', provenance='this_run')
# ================= 03 lua correctness (prior) =================
R2 = pd.read_csv('/home/claude/jss_run2/dash_paper_results.csv', low_memory=False)
R2RUN = R2.run_id.iloc[0]
lc = R2[R2.record_type == 'state_check'].copy()
rows = []
for _, x in lc.iterrows():
    rows.append(dict(record_type='check', experiment='lua_correctness', provenance='prior_run', run_id=R2RUN, timestamp_utc=x.timestamp_utc, code_version=x.code_version,
        source_file='/home/claude/jss_run2/dash_paper_results.csv', data_origin=ORIGIN, update_method=x.update_method, W=x.W, N=x.N, d=x.d, dtype='float32',
        metric=x.metric, text_value=str(x.value).lower(), n=x.n, status=x.status,
        notes=(x.notes if isinstance(x.notes, str) else None), script_sha256=None, input_sha256=None))
nos = json.load(open('/home/claude/jss_run/lua_noscript.json'))
rows.append(dict(record_type='check', experiment='lua_correctness', provenance='prior_run', run_id='636958a3b7ab', metric='noscript_recovery_exploratory', text_value=nos['noscript_recovery'],
    status='pass_exploratory', source_file='/home/claude/jss_run/lua_noscript.json', script_sha256=None, input_sha256=None,
    notes='SCRIPT FLUSH run on the container-local Redis started for testing (no other users); this is the same Redis instance used for all runs, so it is not an independent test server; run2 recorded skipped'))
rows.append(dict(record_type='check', experiment='lua_correctness', provenance='this_run', metric='prior_result_reuse_criteria', text_value='met',
    status='pass', notes='W=10/30/60/120; sequential/pipeline/Lua final byte lists identical; length min(N,W) every update; newest-first; real float32 payload (24 B); NOSCRIPT: exploratory only (see row)'))
sec('03_lua_correctness', rows)
# ================= 04 memory (prior) =================
mr = R2[R2.record_type == 'memory_raw'].copy(); rows = []
for _, x in mr.iterrows():
    rows.append(dict(record_type='raw', experiment='E2_state_only', provenance='prior_run', run_id=R2RUN, timestamp_utc=x.timestamp_utc, code_version=x.code_version,
        source_file='/home/claude/jss_run2/dash_paper_results.csv', data_origin=ORIGIN, engine='shared_A_B1_layout', update_method='lua', repeat=int(x['repeat']), seed=int(x.seed),
        W=int(x.W), N=int(x.N), d=6, dtype='float32', redis_key=x.redis_key, redis_key_bytes=int(x.redis_key_bytes), redis_encoding=x.redis_encoding,
        redis_list_len=int(x.redis_list_len), redis_mem_bytes=int(x.redis_mem_bytes), logical_payload_bytes=int(x.logical_payload_bytes), rss_bytes=int(float(x.rss_bytes)),
        metric='redis_memory_usage_samples0', value=int(x.redis_mem_bytes), unit='bytes', status='ok', script_sha256=None, input_sha256=None,
        notes='single key per W/repeat written by RedisWindowWriter (the writer A and B1 both call): shared storage layout verification, not a full-engine run'))
m = pd.DataFrame(rows)
for (W, N), g in m.groupby(['W', 'N']):
    v = g.redis_mem_bytes.astype(float)
    for agg, val in (('mean_over_repeats', v.mean()), ('sd_over_repeats', v.std(ddof=1)), ('min_over_repeats', v.min()), ('max_over_repeats', v.max())):
        rows.append(dict(record_type='condition_summary', experiment='E2_state_only', provenance='this_run_recomputed_from_prior_raw', engine='shared_A_B1_layout', W=W, N=N, d=6, dtype='float32',
            metric='redis_mem_bytes', aggregation=agg, value=val, unit='bytes', n=len(v), logical_payload_bytes=int(g.logical_payload_bytes.iloc[0]), status='ok'))
e1 = pd.read_csv('/home/claude/jss_run/e2_state_only.csv')
kl = e1.groupby('engine').key_len.unique().apply(lambda a: ','.join(map(str, a))).to_dict()
eq = e1.pivot_table(index=['W', 'N', 'repeat'], columns='engine', values='redis_memory_usage_bytes')
rows.append(dict(record_type='check', experiment='E2_state_only', provenance='prior_run', run_id='636958a3b7ab', source_file='/home/claude/jss_run/e2_state_only.csv',
    metric='exploratory_A_B1_key_lengths_bytes', text_value=f"A={kl.get('A')}; B1={kl.get('B1')}", status='keys_not_equal_length', script_sha256=None, input_sha256=None,
    notes=f'A and B1 keys differ by 1 byte; MEMORY USAGE equal in {int((eq.A==eq.B1).sum())}/{len(eq)} cells, plausibly allocator size-class rounding; not described as equal-length keys'))
rows.append(dict(record_type='check', experiment='E2_state_only', provenance='this_run', metric='A_B1_same_writer_code_path', text_value='true', status='pass', source_file='engine.py',
    notes='ProposedStatefulEngine and RedisFetchBaselineEngine both call RedisWindowWriter.push(f"dash:state:{symbol}", feat.tobytes(), W); B1 additionally LRANGEs (read only)'))
rows.append(dict(record_type='check', experiment='E2_state_only', provenance='this_run', metric='bounded_in_N_fixed_W', text_value=str(bool(m.groupby('W').redis_mem_bytes.nunique().eq(1).all())).lower(), status='pass'))
sec('04_memory_scaling', rows)
# ================= 05 prediction equivalence (prior raw, reference recomputed as A) =================
p = R2[R2.record_type == 'prediction_tick'].copy()
p['prediction'] = p.prediction.astype(float).astype(int); p['tick_index'] = p.tick_index.astype(int)
refA = p[p.engine == 'A_ProposedStateful'].set_index(['model', 'tick_index']).prediction
p['refA'] = [refA[(a, b)] for a, b in zip(p.model, p.tick_index)]
p['stage2_fired'] = p.stage2_fired.astype(str).str.lower().eq('true')
rows = [dict(record_type='raw', experiment='E0_prediction_equivalence', provenance='prior_run', run_id=R2RUN, timestamp_utc=x.timestamp_utc, code_version=x.code_version,
    source_file='/home/claude/jss_run2/dash_paper_results.csv', data_origin=ORIGIN, model=x.model, engine=x.engine, policy='normal', W=30, d=6, dtype='float32',
    stream_local_tick=x.tick_index, request_sequence=x.tick_index, K=1, client_concurrency=1, tau_s1=float(x.tau_s1), tau_s2=float(x.tau_s2), stage2_fired=bool(x.stage2_fired),
    prediction=int(x.prediction), reference_prediction=int(x.refA), label_match=bool(x.prediction == x.refA), metric='prediction', value=int(x.prediction), status='ok',
    script_sha256=None, input_sha256=None, notes='reference_prediction = A-path label, NOT ground truth; agreement between execution paths only; 1000 contiguous rows = first 1000 of newest 20000 rows')
    for _, x in p.iterrows()]
for (mod, eng), g in p.groupby(['model', 'engine']):
    rows.append(dict(record_type='run_summary', experiment='E0_prediction_equivalence', provenance='this_run_recomputed_from_prior_raw', model=mod, engine=eng, policy='normal', W=30,
        metric='mismatch_count_vs_A', value=int((g.prediction != g.refA).sum()), n=len(g), unit='count', status='ok'))
    rows.append(dict(record_type='run_summary', experiment='E0_prediction_equivalence', provenance='this_run_recomputed_from_prior_raw', model=mod, engine=eng, policy='normal', W=30,
        metric='stage2_fire_rate', value=float(g.stage2_fired.mean()), n=len(g), status='ok',
        notes='low firing in this 1000-tick window; Stage-2 branch exercised on only these ticks'))
rows.append(dict(record_type='check', experiment='E0_prediction_equivalence', provenance='this_run', metric='rerun_needed', text_value='false', status='reused',
    notes='prior run2 has per-tick results and code (dash_paper_run.py) meeting 5.3 spec (separate A/B1 keys, same float32 input, normal gating, W=30); no accuracy/F1/AUC computed (no ground truth)'))
sec('05_prediction_equivalence', rows)
# ================= 06 network calibration =================
rows = []
for f in sorted(glob.glob(f'{CK}/ping__*.parquet')):
    d = pd.read_parquet(f); tag = d.tag.iloc[0]; prof = d.network_profile.iloc[0]
    nc = netcols(prof)
    for _, x in d.iterrows():
        rows.append(dict(record_type='raw', experiment='network_calibration', provenance='this_run', request_sequence=int(x.request_sequence), ping_rtt_ms=x.ping_rtt_ms, latency_ms=x.ping_rtt_ms,
            success=bool(x.success), metric='ping_rtt', value=x.ping_rtt_ms, unit='ms', status='warmup_excluded' if x.request_sequence < 50 else 'ok', policy=tag, timeout_ms=1000, **nc,
            notes=None))
    a = d.ping_rtt_ms[50:].to_numpy()
    for mt, v in [('mean', a.mean()), ('sd', a.std(ddof=1)), ('p50', np.quantile(a, .5)), ('p95', np.quantile(a, .95)), ('p99', np.quantile(a, .99))]:
        rows.append(dict(record_type='run_summary', experiment='network_calibration', provenance='this_run', metric=f'ping_rtt_{mt}', value=v, unit='ms', n=len(a), quantile_method=QM,
            success_count=int(d.success[50:].sum()), failure_count=int((~d.success[50:]).sum()), policy=tag, status='ok', **nc))
rows.append(dict(record_type='check', experiment='network_calibration', provenance='this_run', metric='rtt_not_assumed_2x_delay', text_value='true', status='pass',
    notes='RTT taken from measured PING only; proxy overhead included in measured RTT'))
sec('06_network_calibration', rows)
# ================= 07-10 latency matrix =================
files = sorted(glob.glob(f'{CK}/latency_matrix__*.parquet'))
raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
ENGINE_NOTE = {'A_ProposedStateful': 'state_ms=A_ms: RedisWindowWriter.push incl. feat.tobytes() only (non-LSTM: no LRANGE); _to_feat_vec conversion outside state_ms but inside latency_ms',
               'B1_RedisFetch': 'state_ms=A_ms: push + LRANGE(W) + float32 decode of W entries; _to_feat_vec outside state_ms',
               'B2_InMemoryRecompute': 'state_ms=A_ms: in-memory window slice + z-score/diff recompute; no Redis call'}
if len(raw):
    raw['experiment'] = np.where((raw.network_profile == 'P3'), 'latency_matrix_secondary', 'latency_matrix_primary')
    lr = pd.DataFrame(dict(record_type='raw', experiment=raw.experiment, provenance='this_run', data_origin=ORIGIN, workload_type='controlled_replication_closed_loop',
        symbol='btcusdt', logical_stream_id=raw.logical_stream_id, stream_offset=raw.stream_offset, stream_local_tick=raw.stream_local_tick, request_sequence=raw.request_sequence,
        model=raw.model, engine=raw.engine, policy='normal', repeat=raw.repeat, seed=raw.seed, W=30, K=raw.K, client_concurrency=1, d=6, dtype='float32',
        warmup_requests=200, measured_requests=5000, network_profile=raw.network_profile, tau_s1=raw.tau_s1, tau_s2=raw.tau_s2,
        stage2_fired=raw.stage2_fired, prediction=raw.prediction, latency_ms=raw.latency_ms, state_ms=raw.state_ms, stage1_ms=raw.stage1_ms, stage2_ms=raw.stage2_ms,
        success=raw.success, error_type=raw.error_type.replace('', None), timeout_ms=1000, metric='process_tick_total_latency', value=raw.latency_ms, unit='ms',
        status=np.where(raw.success, 'ok', 'failed'), notes='order=' + raw.engine_order.str.replace('|', '>')))
    for p_ in raw.network_profile.unique():
        for k_, v_ in netcols(p_).items():
            if k_ != 'network_profile': lr.loc[lr.network_profile == p_, k_] = v_
    sec('07_latency_raw', lr)
    # run summaries
    rs = []; cov = []
    def qs(a): return {'p50': np.quantile(a, .5), 'p90': np.quantile(a, .9), 'p95': np.quantile(a, .95), 'p99': np.quantile(a, .99)}
    for (ex, mod, K, prof, eng, rep), g in raw.groupby(['experiment', 'model', 'K', 'network_profile', 'engine', 'repeat']):
        ok = g[g.success]; a = ok.latency_ms.to_numpy(); n = len(g)
        c = dict(record_type='run_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine=eng, repeat=rep, seed=20260928 + rep, W=30, d=6,
                 client_concurrency=1, policy='normal', measured_requests=n, n=n, success_count=len(ok), failure_count=n - len(ok), quantile_method=QM, status='ok', **{k: v for k, v in netcols(prof).items() if k != 'network_profile'})
        vals = {'mean': a.mean(), 'sd': a.std(ddof=1), **qs(a), 'failure_rate': (n - len(ok)) / n, 'stage2_fire_rate': ok.stage2_fired.mean(),
                'state_ms_mean': ok.state_ms.mean(), 'stage1_ms_mean': ok.stage1_ms.mean(), 'stage2_ms_mean_when_fired': ok.stage2_ms[ok.stage2_fired].mean() if ok.stage2_fired.any() else np.nan}
        for mt, v in vals.items():
            rs.append(dict(c, metric=('latency_' + mt if mt in ('mean', 'sd', 'p50', 'p90', 'p95', 'p99') else mt), aggregation='per_repeat', value=v,
                           unit=None if mt in ('failure_rate', 'stage2_fire_rate') else 'ms', notes='success-only latency' if mt in ('mean', 'sd', 'p50', 'p90', 'p95', 'p99') else None,
                           status='ok' if np.isfinite(v) else 'missing', reason=None if np.isfinite(v) else 'stage 2 never fired'))
        for dl in DEADLINES:
            cov.append(dict(c, record_type='run_summary', metric='deadline_coverage', deadline_ms=dl, aggregation='per_repeat',
                            value=float((g.success & (g.latency_ms <= dl)).sum() / n), notes='count(success & latency<=deadline)/all measured requests; 1000 ms = 1 s bar reference, not an SLA'))
    rs = pd.DataFrame(rs); cov = pd.DataFrame(cov)
    sec('08_latency_run_summary', rs)
    # condition summaries
    cs = []; rng = np.random.default_rng(20260928); B = 10000
    for (ex, mod, K, prof, eng, mt), g in rs.groupby(['experiment', 'model', 'K', 'network_profile', 'engine', 'metric']):
        v = g.value.astype(float).to_numpy()
        base = dict(record_type='condition_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine=eng, metric=mt, W=30, d=6, client_concurrency=1,
                    n=len(v), unit=g.unit.iloc[0], quantile_method=QM if 'latency_p' in mt else None, status='ok')
        for agg, val in (('mean_of_repeat_values', np.nanmean(v)), ('sd_of_repeat_values', np.nanstd(v, ddof=1)), ('min_of_repeat_values', np.nanmin(v)), ('max_of_repeat_values', np.nanmax(v))):
            d_ = dict(base, aggregation=agg, value=val)
            if agg == 'mean_of_repeat_values' and mt in ('latency_p99', 'latency_p50', 'latency_mean') and len(v) == 5:
                bs = rng.choice(v, (B, len(v))).mean(1)
                d_.update(ci_low=np.quantile(bs, .025), ci_high=np.quantile(bs, .975), ci_method='percentile bootstrap over repeats', bootstrap_unit='repeat (independent state initialization)',
                          bootstrap_B=B, confidence_level=0.95, inference_status='exploratory: only 5 runs; percentile CI from 5 values is unreliable/too narrow', block_length=None)
            cs.append(d_)
    for (ex, mod, K, prof, eng), g in raw[raw.success].groupby(['experiment', 'model', 'K', 'network_profile', 'engine']):
        a = g.latency_ms.to_numpy()
        for q in (.5, .9, .95, .99):
            cs.append(dict(record_type='condition_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine=eng, W=30, d=6, client_concurrency=1,
                metric=f'latency_p{int(q*100)}', aggregation='pooled_over_repeats', value=np.quantile(a, q), unit='ms', n=len(a), quantile_method=QM, status='ok',
                notes='pooled quantile over 5x5000 requests; differs from mean of repeat quantiles'))
    # paired A/B1
    p99 = rs[rs.metric == 'latency_p99'].pivot_table(index=['experiment', 'model', 'K', 'network_profile', 'repeat'], columns='engine', values='value')
    for idx, x in p99.iterrows():
        ex, mod, K, prof, rep = idx
        for mt, val in (('p99_ratio_B1_over_A', x['B1_RedisFetch'] / x['A_ProposedStateful']), ('p99_diff_B1_minus_A', x['B1_RedisFetch'] - x['A_ProposedStateful'])):
            cs.append(dict(record_type='run_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, repeat=rep, engine='B1_vs_A', metric=mt, value=val,
                aggregation='per_repeat_paired_by_repeat_and_schedule', unit=None if 'ratio' in mt else 'ms', W=30, status='ok',
                notes='same repeat/schedule/input; engines run sequentially, so tick-level network noise is not shared'))
    pr = pd.DataFrame([c for c in cs if c.get('engine') == 'B1_vs_A'])
    for (ex, mod, K, prof, mt), g in pr.groupby(['experiment', 'model', 'K', 'network_profile', 'metric']):
        v = g.value.astype(float)
        for agg, val in (('mean_of_repeat_values', v.mean()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
            cs.append(dict(record_type='condition_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine='B1_vs_A', metric=mt, aggregation=agg, value=val, n=len(v), W=30, status='ok'))
    m99 = rs[rs.metric == 'latency_p99'].groupby(['experiment', 'model', 'K', 'network_profile', 'engine']).value.mean()
    for (ex, mod, K, prof, eng), v in m99.items():
        if prof == 'P0': continue
        b0 = m99.get(('latency_matrix_primary', mod, K, 'P0', eng))
        if b0 is None: continue
        cs.append(dict(record_type='condition_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine=eng, metric='p99_change_vs_P0',
            aggregation='mean_of_repeat_values_minus_P0', value=v - b0, unit='ms', W=30, status='ok', notes='P0 measured through same proxy path'))
    csd = pd.DataFrame(cs)
    SECTIONS.append(None)  # placeholder index fix below
    SECTIONS.pop()
    sec('08_latency_run_summary', csd[csd.record_type == 'run_summary'].reset_index(drop=True))
    sec('09_latency_condition_summary', csd[csd.record_type == 'condition_summary'].reset_index(drop=True))
    cc = []
    for (ex, mod, K, prof, eng, dl), g in cov.groupby(['experiment', 'model', 'K', 'network_profile', 'engine', 'deadline_ms']):
        v = g.value.astype(float)
        for agg, val in (('mean_of_repeat_values', v.mean()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
            cc.append(dict(record_type='condition_summary', experiment=ex, provenance='this_run', model=mod, K=K, network_profile=prof, engine=eng, deadline_ms=dl, metric='deadline_coverage',
                           aggregation=agg, value=val, n=len(v), W=30, status='ok'))
    sec('10_deadline_coverage', pd.concat([cov, pd.DataFrame(cc)], ignore_index=True))
# ================= 11 lua cost =================
f = f'{CK}/lua_cost.parquet'
if os.path.exists(f):
    lc = pd.read_parquet(f); MN = {'seq': 'sequential_lpush_ltrim', 'pip': 'pipeline_nontransactional', 'lua': 'lua_eval'}
    rows = pd.DataFrame(dict(record_type='raw', experiment='lua_update_cost', provenance='this_run', update_method=lc.update_method.map(MN), repeat=lc['repeat'], seed=20260928 + lc['repeat'],
        request_sequence=lc.request_sequence, W=30, d=6, dtype='float32', warmup_requests=200, measured_requests=2000, latency_ms=lc.latency_ms, value=lc.latency_ms, unit='ms',
        metric='update_only_latency', redis_key_bytes=lc.redis_key_bytes, success=True, status='ok', notes='order=' + lc.order.str.replace('|', '>'), **netcols('P0')))
    s = []
    for (meth, rep), g in lc.groupby(['update_method', 'repeat']):
        a = g.latency_ms.to_numpy()
        for mt, v in [('mean', a.mean()), ('sd', a.std(ddof=1)), ('p50', np.quantile(a, .5)), ('p95', np.quantile(a, .95)), ('p99', np.quantile(a, .99))]:
            s.append(dict(record_type='run_summary', experiment='lua_update_cost', provenance='this_run', update_method=MN[meth], repeat=rep, metric=f'latency_{mt}', aggregation='per_repeat',
                          value=v, unit='ms', n=len(a), quantile_method=QM, W=30, d=6, status='ok'))
    sd = pd.DataFrame(s)
    for (meth, mt), g in sd.groupby(['update_method', 'metric']):
        v = g.value
        for agg, val in (('mean_of_repeat_values', v.mean()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
            s.append(dict(record_type='condition_summary', experiment='lua_update_cost', provenance='this_run', update_method=meth, metric=mt, aggregation=agg, value=val, unit='ms', n=len(v), W=30, d=6, status='ok',
                          notes='round trips: sequential=2, pipeline=1, lua=1; Lua provides atomic push+trim, pipeline(transaction=False) does not'))
    sec('11_lua_update_cost', pd.concat([rows, pd.DataFrame(s)], ignore_index=True))
# ================= 12 gating =================
gf = sorted(glob.glob(f'{CK}/gating__*.parquet'))
if gf:
    rows = []
    for f in gf:
        prof = f.split('__')[-1].split('.')[0]; g = pd.read_parquet(f); nc = netcols(prof)
        rows.append(pd.DataFrame(dict(record_type='raw', experiment='gating_ablation', provenance='this_run', model='XGBoost', engine='A_ProposedStateful', policy=g.policy, repeat=g['repeat'],
            seed=20260928 + g['repeat'], request_sequence=g.request_sequence, stream_local_tick=g.stream_local_tick, W=30, K=1, client_concurrency=1, d=6, dtype='float32', warmup_requests=200,
            measured_requests=2000, tau_s1=g.tau_s1, tau_s2=g.tau_s2, stage2_fired=g.stage2_fired, prediction=g.prediction, latency_ms=g.latency_ms, state_ms=g.state_ms, stage1_ms=g.stage1_ms,
            stage2_ms=g.stage2_ms, success=g.success, timeout_ms=1000, metric='process_tick_total_latency', value=g.latency_ms, unit='ms', status=np.where(g.success, 'ok', 'failed'),
            notes='first=' + g.first_policy, **nc)))
        for rep, gr in g.groupby('repeat'):
            pv = gr.pivot(index='request_sequence', columns='policy', values='prediction')
            for pol, gp in gr.groupby('policy'):
                a = gp[gp.success].latency_ms.to_numpy()
                for mt, v in [('stage2_fire_rate', gp.stage2_fired.mean()), ('latency_mean', a.mean()), ('latency_sd', a.std(ddof=1)), ('latency_p50', np.quantile(a, .5)), ('latency_p90', np.quantile(a, .9)),
                              ('latency_p95', np.quantile(a, .95)), ('latency_p99', np.quantile(a, .99)), ('failure_rate', (~gp.success).mean())]:
                    rows.append(pd.DataFrame([dict(record_type='run_summary', experiment='gating_ablation', provenance='this_run', model='XGBoost', policy=pol, repeat=rep, K=1, W=30, metric=mt, value=v,
                        aggregation='per_repeat', n=len(gp), unit=None if mt in ('stage2_fire_rate', 'failure_rate') else 'ms', quantile_method=QM if 'latency_p' in mt else None, status='ok', **nc)]))
                for dl in DEADLINES:
                    rows.append(pd.DataFrame([dict(record_type='run_summary', experiment='gating_ablation', provenance='this_run', model='XGBoost', policy=pol, repeat=rep, K=1, W=30, metric='deadline_coverage',
                        deadline_ms=dl, value=float((gp.success & (gp.latency_ms <= dl)).mean()), aggregation='per_repeat', n=len(gp), status='ok', **nc)]))
            rows.append(pd.DataFrame([dict(record_type='run_summary', experiment='gating_ablation', provenance='this_run', model='XGBoost', policy='normal_vs_force_stage2', repeat=rep, K=1, W=30,
                metric='prediction_differs_rate', value=float((pv['normal'] != pv['force_stage2']).mean()), aggregation='per_repeat', n=len(pv), status='ok',
                notes='fraction of ticks where force_stage2 prediction differs from normal gating; not accuracy', **nc)]))
    gd = pd.concat(rows, ignore_index=True)
    s = []
    for (prof, pol, mt, dl), g in gd[gd.record_type == 'run_summary'].fillna({'deadline_ms': -1}).groupby(['network_profile', 'policy', 'metric', 'deadline_ms']):
        v = g.value.astype(float)
        for agg, val in (('mean_of_repeat_values', v.mean()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
            s.append(dict(record_type='condition_summary', experiment='gating_ablation', provenance='this_run', model='XGBoost', policy=pol, K=1, W=30, metric=mt, aggregation=agg, value=val, n=len(v),
                          deadline_ms=None if dl == -1 else dl, status='ok', **netcols(prof)))
    sec('12_gating_ablation', pd.concat([gd, pd.DataFrame(s)], ignore_index=True))
# ================= 13 historical =================
h6 = R2[(R2.experiment == 'E6_deadline_coverage')].copy()
rows = [dict(record_type='run_summary' if x.record_type == 'summary' else 'status', experiment='E6_historical_deadline_coverage', provenance='historical_log_reanalysis', run_id=R2RUN,
    timestamp_utc=x.timestamp_utc, source_file=x.source_file, model=x.model, engine=x.engine, K=x.K, W=x.W, tau_s1=x.tau_s1, logical_stream_id=(re.search(r'logical_symbol=([^;]+);', x.notes).group(1) if isinstance(x.notes, str) and 'logical_symbol=' in x.notes else None), aggregation='per_condition_in_log', deadline_ms=x.deadline_ms, metric=x.metric,
    value=pd.to_numeric(x.value, errors='coerce') if x.metric != 'duplicate_excluded' or True else None, n=x.n, status=x.status, script_sha256=None, input_sha256=None,
    notes=(x.notes if isinstance(x.notes, str) else '') + ' | historical: logs produced before threshold recalibration with older code; not comparable to this run (different host/code)')
    for _, x in h6.iterrows()]
sec('13_historical_results', rows)
# ================= 14 status =================
done = {os.path.basename(f) for f in glob.glob(f'{CK}/*.parquet')}
st = []
for prof in ['P0', 'P1', 'P2']:
    for K in (1, 10, 100):
        n = sum(f'latency_matrix__XGBoost__K{K}__{prof}__r{r_}.parquet' in done for r_ in range(5))
        st.append(dict(experiment='latency_matrix_primary', model='XGBoost', K=K, network_profile=prof, status='done' if n == 5 else ('incomplete' if n else 'not_run'), value=n, metric='repeats_completed'))
n = sum(f'latency_matrix__XGBoost__K100__P3__r{r_}.parquet' in done for r_ in range(5))
st.append(dict(experiment='latency_matrix_secondary', model='XGBoost', K=100, network_profile='P3', status='done' if n == 5 else ('incomplete' if n else 'not_run'), value=n, metric='repeats_completed',
               reason=None if n == 5 else json.load(open(f'{OUT}/status_secondary.json')).get('reason') if os.path.exists(f'{OUT}/status_secondary.json') else 'not reached'))
for txt, K, prof, model in [('XGBoost K=100 P4', 100, 'P4', 'XGBoost'), ('XGBoost K=500 P0', 500, 'P0', 'XGBoost'), ('XGBoost K=500 P2', 500, 'P2', 'XGBoost')] + \
        [(f'{m_} K={k_} {p_}', k_, p_, m_) for m_ in ('Logistic', 'RandomForest') for k_ in (1, 100) for p_ in ('P0', 'P2')]:
    st.append(dict(experiment='latency_matrix_secondary', model=model, K=K, network_profile=prof, status='not_run', reason='excluded before results: 90-min budget (pilot estimate)'))
st += [dict(experiment='lua_update_cost', status='done' if 'lua_cost.parquet' in done else 'deferred'),
       dict(experiment='gating_ablation', network_profile='P0', status='done' if 'gating__XGBoost__K1__P0.parquet' in done else 'not_run'),
       dict(experiment='gating_ablation', network_profile='P2', status='done' if 'gating__XGBoost__K1__P2.parquet' in done else 'not_run'),
       dict(experiment='network_calibration', status='done'), dict(experiment='lua_correctness', status='reused', reason='prior_run met criteria'),
       dict(experiment='E2_state_only', status='reused', reason='prior_run met criteria'), dict(experiment='E0_prediction_equivalence', status='reused', reason='prior per-tick results exist'),
       dict(experiment='netem_namespace_impairment', status='failed', reason='kernel CONFIG_NET_SCH_NETEM not set; tc qdisc netem unknown; test netns/veth removed'),
       dict(experiment='noscript_independent_server', status='skipped', reason='no independent Redis test server'),
       dict(experiment='block_bootstrap_ci', status='deferred', reason='block length not validated; no block-length sensitivity'),
       dict(experiment='E1_capacity_T2_T3', status='not_run', reason='excluded by scope'),
       dict(experiment='open_loop_arrival_rate', status='not_run', reason='excluded by scope'),
       dict(experiment='breakpoint_capacity_law_pvalue', status='not_run', reason='not computed; none claimed')]
for x in st: x.update(record_type='status', provenance='this_run')
sec('14_experiment_status', st)
# ================= 15 limitations =================
L = ['Shared ephemeral cloud container (2 vCPU) and container-local Redis, not the original experiment PC/Redis; absolute latencies not transferable.',
     'Python 3.13 instead of README target 3.11 (sklearn 1.6.1 met).',
     'Network impairment via user-space Python TCP proxy (netem unavailable); proxy adds its own CPU/timer delay (~0.5 ms per 1 ms configured, see PING) and shares CPUs with client and Redis. Synthetic controlled impairment, not real network.',
     'Jitter = Normal(1 ms, 0.5 ms) per chunk, clipped at 0, FIFO release (no reordering) which correlates successive delays; distribution of observed RTT differs from configured parameters.',
     'XGBoost used default multithreading (observed ~150% CPU) competing with Redis/proxy on 2 vCPU; thread count not pinned.',
     'Primary workload is closed-loop, client_concurrency=1, K resident replicated streams from one symbol; not multi-client concurrency, not production arrival load, not K real symbols; no queueing/SLA or capacity inference.',
     'P99 from 5000 requests depends on ~50 largest observations per repeat; 5 repeats give limited uncertainty quantification; bootstrap CIs over repeats are exploratory.',
     'No dependence-aware (block) CI; block-length sensitivity not done.',
     'Engines within a repeat run sequentially; A/B1 pairing is by repeat and schedule, not shared tick-level noise.',
     'state_ms (A_ms) excludes _to_feat_vec conversion; for A it covers push only (no LRANGE for non-LSTM models), for B1 push+LRANGE+decode; component times do not sum exactly to total.',
     'E0 window (1000 ticks) rarely opened the Stage-2 gate (Logistic/RF 0, XGB 3 ticks); agreement evidence for the Stage-2 branch is weak.',
     'No ground truth used: no accuracy/F1/AUC/deadline_correct_rate; force_stage2 prediction changes are not accuracy changes.',
     'T4: observed Stage-2 timings are noisy wall-clock samples; they do not establish a deterministic bound delta.',
     'Jitter-induced P99 degradation is not evidence for T2 (cache-miss knee) or T3 (capacity law); no saturation observed means no scalability limit claimed.',
     'Memory: state is bounded in N for fixed W,d and grows with W (logical payload 4*d*min(N,W)); not W-independent.',
     'Historical E6 logs (tau=0.6731, older code) are reported separately and not comparable to this run.',
     'Progress logging is per engine-run; long engine runs (e.g. P3 B1, ~3 min) had no intermediate 30 s log line.']
sec('15_limitations', [dict(record_type='limitation', experiment='limitations', provenance='this_run', metric=f'limitation_{i+1:02d}', text_value=t, status='recorded') for i, t in enumerate(L)])
# ================= write + validate =================
out = pd.concat([s.reindex(columns=COLS) for s in SECTIONS], ignore_index=True)
out['_o'] = out.section.str[:2].astype(int); out = out.sort_values('_o', kind='stable').drop(columns='_o').reset_index(drop=True)
for c in ['success', 'stage2_fired', 'label_match']:
    out[c] = out[c].map(lambda v: None if pd.isna(v) else ('true' if bool(v) else 'false'))
tmp = f'{OUT}/dash_paper_results.tmp.csv'
out.to_csv(tmp, index=False, encoding='utf-8'); NOUT = len(out); del out, SECTIONS, raw, lr; import gc; gc.collect()
chk = pd.read_csv(tmp, low_memory=False, keep_default_na=False, na_values=[''], dtype={'success': str, 'stage2_fired': str, 'label_match': str, 'text_value': str})
V = []
V.append(('columns_match', list(chk.columns) == COLS)); V.append(('row_count_match', len(chk) == NOUT))
V.append(('section_order', list(dict.fromkeys(chk.section)) == sorted(set(chk.section))))
PK = ['section', 'record_type', 'experiment', 'provenance', 'run_id', 'source_file', 'model', 'engine', 'policy', 'update_method', 'repeat', 'W', 'N', 'K', 'network_profile',
      'logical_stream_id', 'request_sequence', 'stream_local_tick', 'deadline_ms', 'metric', 'aggregation', 'redis_key']
V.append(('no_duplicate_pk', not chk.duplicated(PK).any()))
lat = chk[(chk.section == '07_latency_raw')]
V.append(('raw_latency_finite_nonneg', bool(np.isfinite(lat.latency_ms).all() and (lat.latency_ms >= 0).all())))
V.append(('bool_fields_consistent', all(set(chk[c].dropna().unique()) <= {'true', 'false'} for c in ['success', 'stage2_fired', 'label_match'])))
cv = chk[chk.metric == 'deadline_coverage'].value
V.append(('coverage_in_0_1', bool(((cv >= 0) & (cv <= 1)).all())))
mm = chk.redis_mem_bytes.dropna(); V.append(('memory_nonneg_int', bool(((mm >= 0) & (mm == mm.round())).all())))
V.append(('timestamps_iso_utc', bool(chk.timestamp_utc.dropna().str.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$').all())))
# recompute from raw
rsum = chk[(chk.section == '08_latency_run_summary')]
bad = 0
for (ex, mod, K, prof, eng, rep), g in lat.groupby(['experiment', 'model', 'K', 'network_profile', 'engine', 'repeat']):
    a = g[g.success == 'true'].latency_ms.to_numpy()
    s = rsum[(rsum.experiment == ex) & (rsum.K == K) & (rsum.network_profile == prof) & (rsum.engine == eng) & (rsum.repeat == rep)].set_index('metric').value
    bad += (len(g) != 5000) + (not np.isclose(s['latency_p99'], np.quantile(a, .99))) + (not np.isclose(s['latency_p50'], np.quantile(a, .5)))
    dc = chk[(chk.section == '10_deadline_coverage') & (chk.record_type == 'run_summary') & (chk.experiment == ex) & (chk.K == K) & (chk.network_profile == prof) & (chk.engine == eng) & (chk.repeat == rep)]
    for _, x in dc.iterrows(): bad += not np.isclose(x.value, ((g.success == 'true') & (g.latency_ms <= x.deadline_ms)).mean())
V.append(('recompute_counts_quantiles_coverage_from_raw', bad == 0))
for nme, ok in V: print(('PASS ' if ok else 'FAIL ') + nme)
if all(ok for _, ok in V): os.replace(tmp, f'{OUT}/dash_paper_results.csv'); print('written', len(chk), 'rows')
json.dump(dict(V), open(f'{OUT}/validation.json', 'w'), indent=1)
