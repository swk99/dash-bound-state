"""Assemble dash_replication_results.csv from ckpt_<env>/ dirs (one or two environments). Never pools environments."""
import os, sys, glob, json, hashlib, platform, datetime as dt
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('DASH_OUT', HERE)
import run_replication as R   # reuses data/manifest/hash definitions (no experiment is run on import)
COLS = open(f'{HERE}/columns.txt').read().split()
QM = 'numpy.quantile method=linear (Hyndman-Fan type 7)'; NOW = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
PRIOR = os.environ.get('DASH_PRIOR_PROXY_CSV', '/home/claude/jss_run3/dash_paper_results.csv')
envs = sorted(d.split('ckpt_')[1] for d in glob.glob(f'{HERE}/ckpt_*') if glob.glob(f'{d}/lat__*.parquet'))
S = []
def sec(name, rows, **c):
    df = pd.DataFrame(rows)
    if not len(df): return
    for k, v in dict(code_version=R.CODE_VERSION, script_sha256=R.SCRIPT_SHA, input_sha256=R.INPUT_SHA, timestamp_utc=NOW, **c).items():
        if k not in df: df[k] = v
    df['section'] = name; S.append(df)
def meta(metric, val, **kw):
    d = dict(record_type='metadata', metric=metric, **kw)
    if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool): d['value'] = float(val)
    else: d['text_value'] = str(val)
    return d
NOTE_TOPO = 'direct TCP to Redis, no proxy, no impairment (P0_direct)'
# ---------- 00 manifest ----------
rows = [meta('environments_present', ','.join(envs)), meta('symbol', 'btcusdt'), meta('data_files_count', len(R.FS)), meta('data_file_manifest_sha256', R.MANIFEST_SHA),
        meta('data_total_rows', len(R.DFALL)), meta('data_timestamp_min', str(R.DFALL.sec.min())), meta('data_timestamp_max', str(R.DFALL.sec.max())),
        meta('data_nan_count', int(np.isnan(R.XALL).sum())), meta('data_inf_count', int(np.isinf(R.XALL).sum())), meta('timestamp_monotonic', bool(R.DFALL.sec.is_monotonic_increasing)),
        meta('feature_order', '|'.join(R.cfg.FEATURE_COLS)), meta('input_array_sha256', R.INPUT_SHA), meta('time_unit', 'ms'), meta('timestamp_timezone', 'UTC ISO 8601'),
        meta('design', 'XGBoost; W=30; d=6; K in {1,100}; network_profile=P0_direct (no proxy); engines A/B1/B2; 5 repeats; 200 warmup + 5000 measured per engine/K/repeat; engine order seeded permutation default_rng([20260928, repeat, K, 0, 4]); model inference threads pinned to 1'),
        meta('workload_definition', 'controlled replication: K resident logical streams, deterministic contiguous non-overlapping segments of one real series; single client, closed loop, client_concurrency=1; not K real symbols, not multi-client, not arrival load'),
        meta('relation_to_prior_P0', 'prior P0 (run dash3_20261002T1122Z) went through the experiment delay proxy with default XGBoost threading; this run is direct with pinned threads: different path and thread setting, reported separately, never pooled')]
for k, v in R.PREREG.items(): rows.append(meta(f'preregistered_{k}', v, notes='written to preregistered_rules_<env>.json before measurement'))
for f, h in R.ART.items(): rows.append(meta('artifact_sha256', h, source_file=f))
sec('00_run_manifest', rows, record_type='metadata', experiment='manifest', provenance='this_run')
# ---------- 01 environment ----------
rows = []
for env in envs:
    m = json.load(open(f'{HERE}/ckpt_{env}/run_meta.json'))
    rid = m['run']
    import sklearn, xgboost, redis as rp, psutil, torch
    here = env == os.environ.get('DASH_ENV_LABEL', 'cloud')
    E = dict(python=m.get('python', platform.python_version() if here else None), redis_server=m.get('redis_version', '7.0.15' if env == 'cloud' else None), redis_py=rp.__version__ if here else None,
             numpy=np.__version__ if here else None, pandas=pd.__version__ if here else None, sklearn=sklearn.__version__ if here else None, xgboost=xgboost.__version__ if here else None,
             torch=torch.__version__ if here else None, os=m.get('platform', platform.platform() if here else None), cpu_model=m.get('cpu_model', 'Intel(R) Xeon(R) Processor @ 2.10GHz' if env == 'cloud' else None),
             vcpu=m.get('vcpu', os.cpu_count() if here else None), ram_total_bytes=m.get('ram_total_bytes', psutil.virtual_memory().total if here else None), sha256_check=m.get('sha_check', 'reference environment (defines expected SHA256)' if env == 'cloud' else None),
             redis_location=f'{R.RH}:{R.RP}, same host, loopback TCP, direct (no proxy)', redis_client='redis-py default ConnectionPool, one connection used sequentially, socket_timeout=1.0s',
             model_threads=f"XGBoost booster nthread readback={json.dumps(m['thread_check'])}; OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1; torch.set_num_threads(1)",
             version_warnings=json.dumps(m['version_warnings']), loader_stdout=m['loader_stdout'], run_end_utc=m['end'])
    if env == 'cloud':
        E.update(shared_cloud_host='yes: ephemeral cloud container (Firecracker VM, 2 vCPU), co-tenancy unknown',
                 redis_restart='container rebooted at ~14:06Z; Redis 7.0.15 restarted with default config in an empty dir before this run (no keys, noeviction, maxmemory unset); not the original experiment Redis',
                 python_readme_target='3.11 (not met; 3.13 used)', host_cpu_monitor='psutil.cpu_percent sampled every 1 s in a background thread during each engine run (small interference)')
    for k, v in E.items():
        if v is not None: rows.append(dict(meta(k, v), run_id=rid, notes=f'env={env}'))
sec('01_environment', rows, experiment='environment', provenance='this_run')
# ---------- 02 thread / artifact check ----------
rows = []
for env in envs:
    m = json.load(open(f'{HERE}/ckpt_{env}/run_meta.json'))
    ok = all(str(v) == '1' for v in m['thread_check'].values())
    rows.append(dict(record_type='check', run_id=m['run'], model='XGBoost', metric='inference_threads_pinned_to_1', text_value=str(ok).lower(), status='pass' if ok else 'fail', notes=f"env={env}; readback {m['thread_check']}"))
    for mdl in ('Logistic', 'RandomForest', 'XGBoost'):
        p = f'{HERE}/ckpt_{env}/e0__{mdl}.json'
        if os.path.exists(p):
            j = json.load(open(p)); th = json.loads(j['threads'])
            rows.append(dict(record_type='check', run_id=m['run'], model=mdl, metric='e0_threads_pinned', text_value=json.dumps(th), status='recorded', tau_s1=j['tau_s1'], tau_s2=j['tau_s2'], notes=f'env={env}'))
sec('02_artifact_checks', rows, experiment='thread_checks', provenance='this_run')
# ---------- 05 Stage-2 E0 ----------
rows = []
for env in envs:
    rid = json.load(open(f'{HERE}/ckpt_{env}/run_meta.json'))['run']
    for mdl in ('Logistic', 'RandomForest', 'XGBoost'):
        p = f'{HERE}/ckpt_{env}/e0__{mdl}.parquet'; j = json.load(open(f'{HERE}/ckpt_{env}/e0__{mdl}.json'))
        base = dict(run_id=rid, experiment='E0_stage2_window', model=mdl, policy='normal', W=30, d=6, dtype='float32', K=1, client_concurrency=1, tau_s1=j['tau_s1'], tau_s2=j['tau_s2'], network_profile='P0_direct')
        rows.append(dict(base, record_type='check', metric='window_selection_rule', text_value=R.PREREG['e0_window_rule'], status='applied'))
        rows.append(dict(base, record_type='run_summary', metric='window_start_row', value=j['window_start'], status='ok' if j['window_start'] is not None else 'not_found'))
        rows.append(dict(base, record_type='run_summary', metric='window_stage1_open_fraction_pi1', value=j['window_fire_rate_pi1'], status='ok', notes='SELECTED validation window; not representative firing rate'))
        rows.append(dict(base, record_type='run_summary', metric='dataset_stage1_open_fraction_pi1', value=j['dataset_fire_rate'], n=len(R.XALL), status='ok', notes='whole array (74,855 rows), Stage-1 only'))
        if not os.path.exists(p): continue
        d = pd.read_parquet(p)
        for _, x in d.iterrows():
            rows.append(dict(base, record_type='raw', engine=x.engine, stream_local_tick=int(x.tick_index), request_sequence=int(x.tick_index), stream_offset=int(x.global_row) - int(x.tick_index),
                             prediction=int(x.prediction), reference_prediction=int(x.reference_prediction), label_match=bool(x.label_match), stage2_fired=bool(x.stage2_fired),
                             metric='prediction', value=int(x.prediction), status='ok', notes='reference_prediction = A-path label, not ground truth'))
        for eng, g in d.groupby('engine'):
            rows.append(dict(base, record_type='run_summary', engine=eng, metric='mismatch_count_vs_A', value=int((~g.label_match).sum()), n=len(g), status='ok'))
            rows.append(dict(base, record_type='run_summary', engine=eng, metric='stage2_fire_rate', value=float(g.stage2_fired.mean()), n=len(g), status='ok', notes='in selected window only'))
            rows.append(dict(base, record_type='run_summary', engine=eng, metric='stage2_fired_ticks', value=int(g.stage2_fired.sum()), n=len(g), status='ok'))
            for lab, c in g[g.stage2_fired].prediction.value_counts().items():
                rows.append(dict(base, record_type='run_summary', engine=eng, metric=f'stage2_prediction_count_label_{lab}', value=int(c), status='ok', notes='both Stage-2 output labels exercised if two labels appear'))
sec('05_prediction_equivalence', rows, provenance='this_run')
# ---------- 07-10 latency ----------
raws = []
for env in envs:
    rid = json.load(open(f'{HERE}/ckpt_{env}/run_meta.json'))['run']
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{HERE}/ckpt_{env}/lat__*.parquet'))]); d['env'] = env; d['run_id'] = rid; raws.append(d)
raw = pd.concat(raws, ignore_index=True)
cpu = pd.concat([pd.read_csv(f'{HERE}/ckpt_{e}/cpu.csv').assign(env=e) for e in envs])
lr = pd.DataFrame(dict(record_type='raw', experiment='P0_direct_replication', run_id=raw.run_id, provenance='this_run', data_origin='btcusdt_real_1s', workload_type='controlled_replication_closed_loop',
    symbol='btcusdt', logical_stream_id=raw.logical_stream_id, stream_offset=raw.stream_offset, stream_local_tick=raw.stream_local_tick, request_sequence=raw.request_sequence, model='XGBoost',
    engine=raw.engine, policy='normal', repeat=raw['repeat'], seed=20260928 + raw['repeat'], W=30, K=raw.K, client_concurrency=1, d=6, dtype='float32', warmup_requests=200, measured_requests=5000,
    network_profile='P0_direct', impairment_backend='none', redis_topology='same_host_loopback_direct', tau_s1=raw.tau_s1, tau_s2=raw.tau_s2, stage2_fired=raw.stage2_fired, prediction=raw.prediction,
    latency_ms=raw.latency_ms, state_ms=raw.state_ms, stage1_ms=raw.stage1_ms, stage2_ms=raw.stage2_ms, success=raw.success, error_type=raw.error_type.replace('', None), timeout_ms=1000,
    metric='process_tick_total_latency', value=raw.latency_ms, unit='ms', status=np.where(raw.success, 'ok', 'failed'), notes='env=' + raw.env + '; order=' + raw.engine_order))
sec('07_latency_raw', lr)
rs, cov = [], []
for (env, rid, K, eng, rep), g in raw.groupby(['env', 'run_id', 'K', 'engine', 'repeat']):
    ok = g[g.success]; a = ok.latency_ms.to_numpy(); n = len(g)
    c = dict(record_type='run_summary', experiment='P0_direct_replication', run_id=rid, provenance='this_run', model='XGBoost', engine=eng, K=K, repeat=rep, seed=20260928 + rep, W=30, d=6,
             network_profile='P0_direct', redis_topology='same_host_loopback_direct', client_concurrency=1, n=n, success_count=len(ok), failure_count=n - len(ok), quantile_method=QM, status='ok', notes=f'env={env}')
    cr = cpu[(cpu.env == env) & (cpu.K == K) & (cpu.engine == eng) & (cpu['repeat'] == rep)]
    vals = dict(latency_mean=a.mean(), latency_sd=a.std(ddof=1), latency_p50=np.quantile(a, .5), latency_p90=np.quantile(a, .9), latency_p95=np.quantile(a, .95), latency_p99=np.quantile(a, .99),
                failure_rate=(n - len(ok)) / n, stage2_fire_rate=ok.stage2_fired.mean(), state_ms_mean=ok.state_ms.mean(), stage1_ms_mean=ok.stage1_ms.mean(),
                host_cpu_percent_mean=cr.cpu_mean.iloc[0] if len(cr) else np.nan, host_cpu_percent_max=cr.cpu_max.iloc[0] if len(cr) else np.nan)
    for mt, v in vals.items():
        rs.append(dict(c, metric=mt, aggregation='per_repeat', value=v, unit='percent' if 'cpu' in mt else (None if mt in ('failure_rate', 'stage2_fire_rate') else 'ms')))
    for dl in R.DEADLINES:
        cov.append(dict(c, metric='deadline_coverage', deadline_ms=dl, aggregation='per_repeat', value=float((g.success & (g.latency_ms <= dl)).sum() / n), notes=f'env={env}; success & latency<=deadline over all measured; not an SLA'))
rs = pd.DataFrame(rs)
p99 = rs[rs.metric == 'latency_p99'].assign(env=rs.notes.str[4:]).pivot_table(index=['env', 'run_id', 'K', 'repeat'], columns='engine', values='value')
pair = []
for (env, rid, K, rep), x in p99.iterrows():
    for mt, v in (('p99_ratio_B1_over_A', x['B1_RedisFetch'] / x['A_ProposedStateful']), ('p99_diff_B1_minus_A', x['B1_RedisFetch'] - x['A_ProposedStateful'])):
        pair.append(dict(record_type='run_summary', experiment='P0_direct_replication', run_id=rid, provenance='this_run', model='XGBoost', engine='B1_vs_A', K=K, repeat=rep, W=30, network_profile='P0_direct',
                         metric=mt, aggregation='per_repeat_paired_by_repeat_and_schedule', value=v, unit=None if 'ratio' in mt else 'ms', status='ok', notes=f'env={env}; engines run sequentially, noise not shared'))
pair = pd.DataFrame(pair)
sec('08_latency_run_summary', pd.concat([rs, pair], ignore_index=True))
cs = []
allrs = pd.concat([rs, pair], ignore_index=True); allrs['env'] = allrs.notes.str.extract(r'env=(\w+)')[0]
for (env, rid, K, eng, mt), g in allrs.groupby(['env', 'run_id', 'K', 'engine', 'metric']):
    v = g.value.astype(float)
    for agg, val in (('mean_of_repeat_values', v.mean()), ('median_of_repeat_values', v.median()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
        cs.append(dict(record_type='condition_summary', experiment='P0_direct_replication', run_id=rid, provenance='this_run', model='XGBoost', engine=eng, K=K, metric=mt, aggregation=agg, value=val, n=len(v),
                       W=30, network_profile='P0_direct', status='ok', unit=g.unit.iloc[0], quantile_method=QM if 'latency_p' in mt else None,
                       inference_status='descriptive over 5 repeats; no CI (5 runs)', notes=f'env={env}'))
for (env, rid, K, eng), g in raw[raw.success].groupby(['env', 'run_id', 'K', 'engine']):
    cs.append(dict(record_type='condition_summary', experiment='P0_direct_replication', run_id=rid, provenance='this_run', model='XGBoost', engine=eng, K=K, metric='latency_p99', aggregation='pooled_over_repeats_within_env',
                   value=np.quantile(g.latency_ms, .99), n=len(g), unit='ms', W=30, network_profile='P0_direct', quantile_method=QM, status='ok', notes=f'env={env}; pooled within one environment only'))
# pre-registered decisions
dec = []
for env in envs:
    for K in R.KS:
        rat = pair[(pair.notes.str.startswith(f'env={env};')) & (pair.K == K) & (pair.metric == 'p99_ratio_B1_over_A')].value
        b2 = rs[(rs.notes == f'env={env}') & (rs.K == K) & (rs.engine == 'B2_InMemoryRecompute') & (rs.metric == 'latency_p99')].value
        cv = b2.std(ddof=1) / b2.mean(); med = rat.median()
        base = dict(record_type='check', experiment='preregistered_decision', provenance='this_run', model='XGBoost', K=K, network_profile='P0_direct', notes=f'env={env}')
        dec += [dict(base, metric='repeats_with_B1_over_A_gt_1', value=int((rat > 1).sum()), n=len(rat), status='computed'),
                dict(base, metric='median_ratio_B1_over_A', value=med, n=len(rat), status='computed'),
                dict(base, metric='B2_p99_cv_noise_gate', value=cv, n=len(b2), status='noisy_inconclusive' if cv > 0.25 else 'ok'),
                dict(base, metric='scope_rule_benefit_limited', text_value=str(bool(med <= 1.10)).lower(), status='limited' if med <= 1.10 else 'not_limited')]
if len(envs) < 2:
    for K in R.KS:
        dec.append(dict(record_type='check', experiment='preregistered_decision', provenance='this_run', model='XGBoost', K=K, network_profile='P0_direct', metric='cross_environment_direction_maintained',
                        status='pending_second_environment', reason='only one environment run so far; rule requires both'))
else:
    for K in R.KS:
        meds = [pd.DataFrame(dec)[(pd.DataFrame(dec).metric == 'median_ratio_B1_over_A') & (pd.DataFrame(dec).K == K) & (pd.DataFrame(dec).notes == f'env={e}')].value.iloc[0] for e in envs]
        same = len({np.sign(m - 1) for m in meds}) == 1
        dec.append(dict(record_type='check', experiment='preregistered_decision', provenance='this_run', model='XGBoost', K=K, network_profile='P0_direct', metric='cross_environment_direction_maintained',
                        text_value=str(same).lower(), status='maintained' if same else 'not_maintained', notes=';'.join(f'{e}={m:.3f}' for e, m in zip(envs, meds))))
sec('09_latency_condition_summary', pd.DataFrame(cs + dec))
cc = []
cov = pd.DataFrame(cov)
for (rid, K, eng, dl, nt), g in cov.groupby(['run_id', 'K', 'engine', 'deadline_ms', cov.notes.str.extract(r'(env=\w+)')[0]]):
    v = g.value
    for agg, val in (('mean_of_repeat_values', v.mean()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
        cc.append(dict(record_type='condition_summary', experiment='P0_direct_replication', run_id=rid, provenance='this_run', model='XGBoost', engine=eng, K=K, deadline_ms=dl, metric='deadline_coverage',
                       aggregation=agg, value=val, n=len(v), W=30, network_profile='P0_direct', status='ok', notes=nt))
sec('10_deadline_coverage', pd.concat([cov, pd.DataFrame(cc)], ignore_index=True))
# ---------- 13 prior proxy-path P0 reference (separate provenance) ----------
rows = []
if os.path.exists(PRIOR):
    pr = pd.read_csv(PRIOR, usecols=['section', 'record_type', 'experiment', 'run_id', 'model', 'engine', 'K', 'network_profile', 'metric', 'aggregation', 'value'], low_memory=False)
    pr = pr[(pr.section == '09_latency_condition_summary') & (pr.network_profile == 'P0') & (pr.K.isin([1, 100])) & (pr.metric.isin(['latency_p99', 'p99_ratio_B1_over_A', 'latency_p50'])) & (pr.aggregation.isin(['mean_of_repeat_values', 'sd_of_repeat_values']))]
    for _, x in pr.iterrows():
        rows.append(dict(record_type='condition_summary', experiment='prior_P0_via_proxy_reference', run_id=x.run_id, provenance='prior_run_different_path', model=x.model, engine=x.engine, K=x.K,
                         network_profile='P0_via_proxy', impairment_backend='py_delay_proxy (0 ms)', metric=x.metric, aggregation=x.aggregation, value=x.value, status='reference_only',
                         source_file=os.path.basename(PRIOR), script_sha256=None, input_sha256=None,
                         notes='different path (through delay proxy) and default XGBoost threading; not pooled or directly compared as same condition'))
sec('13_historical_results', rows)
# ---------- 14 status ----------
st = [dict(experiment='P0_direct_replication', notes=f'env={e}', status='done') for e in envs]
if 'labpc' not in envs: st.append(dict(experiment='P0_direct_replication', notes='env=labpc', status='not_run', reason='requires experiment PC; run package provided'))
st += [dict(experiment='E0_stage2_window', status='done', notes=f'env={e}') for e in envs]
st += [dict(experiment='jitter_profiles_P1_P4', status='not_run', reason='out of scope for this replication; P0 replication does not validate prior jitter results'),
       dict(experiment='dependence_aware_ci', status='deferred', reason='5 runs; descriptive repeat statistics only')]
for x in st: x.update(record_type='status', provenance='this_run')
sec('14_experiment_status', st)
# ---------- 15 limitations ----------
L = ['Cloud environment is a shared 2 vCPU container with a container-local Redis restarted for this run; not the original experiment Redis.',
     'Python 3.13 instead of README target 3.11 (sklearn 1.6.1 met).',
     'Host CPU monitor samples in a background thread (1 s); small measurement interference, applied identically to all engines.',
     'Closed-loop single client, K replicated streams from one symbol; no multi-client, arrival-load or queueing inference.',
     'P99 of 5000 requests rests on ~50 largest observations; 5 repeats; no confidence intervals reported.',
     'Engines run sequentially within a repeat; A/B1 pairing is by repeat and schedule only.',
     'E0 windows are rule-selected to exercise Stage 2; their firing rates are not representative of the dataset (dataset rates reported separately).',
     'E0 labels compare execution paths only; no ground truth, no accuracy.',
     'This P0 direct replication does not validate the earlier proxy-based jitter results.',
     'Cross-environment conclusion pending until the experiment-PC run is added (same script, same SHA256).']
sec('15_limitations', [dict(record_type='limitation', experiment='limitations', provenance='this_run', metric=f'limitation_{i+1:02d}', text_value=t, status='recorded') for i, t in enumerate(L)])
# ---------- write + validate ----------
out = pd.concat([s.reindex(columns=COLS) for s in S], ignore_index=True)
out['_o'] = out.section.str[:2].astype(int); out = out.sort_values('_o', kind='stable').drop(columns='_o').reset_index(drop=True)
for c in ['success', 'stage2_fired', 'label_match']: out[c] = out[c].map(lambda v: None if pd.isna(v) else ('true' if bool(v) else 'false'))
tmp = f'{HERE}/dash_replication_results.tmp.csv'; out.to_csv(tmp, index=False, encoding='utf-8'); N = len(out); del out
chk = pd.read_csv(tmp, low_memory=False, keep_default_na=False, na_values=[''], dtype={'success': str, 'stage2_fired': str, 'label_match': str, 'text_value': str})
PK = ['section', 'record_type', 'experiment', 'provenance', 'run_id', 'source_file', 'model', 'engine', 'policy', 'repeat', 'K', 'network_profile', 'logical_stream_id', 'request_sequence', 'stream_local_tick', 'deadline_ms', 'metric', 'aggregation', 'notes']
lat = chk[chk.section == '07_latency_raw']; bad = 0
for (rid, K, eng, rep), g in lat.groupby(['run_id', 'K', 'engine', 'repeat']):
    s = chk[(chk.section == '08_latency_run_summary') & (chk.run_id == rid) & (chk.K == K) & (chk.engine == eng) & (chk.repeat == rep)].set_index('metric').value
    a = g[g.success == 'true'].latency_ms
    bad += (len(g) != 5000) + (not np.isclose(s['latency_p99'], np.quantile(a, .99))) + (not np.isclose(s['latency_p50'], np.quantile(a, .5)))
V = dict(columns_match=list(chk.columns) == COLS, row_count_match=len(chk) == N, section_order=list(dict.fromkeys(chk.section)) == sorted(set(chk.section)),
         no_duplicate_pk=not chk.duplicated(PK).any(), raw_latency_finite_nonneg=bool(np.isfinite(lat.latency_ms).all() and (lat.latency_ms >= 0).all()),
         bool_fields_consistent=all(set(chk[c].dropna()) <= {'true', 'false'} for c in ['success', 'stage2_fired', 'label_match']),
         coverage_in_0_1=bool(chk[chk.metric == 'deadline_coverage'].value.between(0, 1).all()),
         timestamps_iso_utc=bool(chk.timestamp_utc.dropna().str.match(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$').all()), recompute_from_raw=bad == 0)
for k, v in V.items(): print(('PASS ' if v else 'FAIL ') + k)
json.dump(V, open(f'{HERE}/validation_replication.json', 'w'), indent=1)
if all(V.values()): os.replace(tmp, f'{HERE}/dash_replication_results.csv'); print('written', N)
