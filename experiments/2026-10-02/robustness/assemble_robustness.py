import os, sys, glob, json, datetime as dt
import numpy as np, pandas as pd
os.environ.setdefault('DASH_OUT', '/tmp/claude-0/-home-claude/5ed89c38-be66-59f3-b1fa-10abac770c57/scratchpad'); sys.path.insert(0, '/home/claude/jss_run4')
import run_replication as R
HERE = '/home/claude/jss_run5'; COLS = open('/home/claude/jss_run4/columns.txt').read().split()
QM = 'numpy.quantile method=linear (Hyndman-Fan type 7)'; NOW = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
SCRIPT_SHA = R.sha(f'{HERE}/run_robustness.py'); RUN = 'robust_cloud_20261002'
pre = json.load(open(f'{HERE}/preregistered_rules_robustness.json')); meta = json.load(open(f'{HERE}/ckpt/run_meta.json'))
raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{HERE}/ckpt/*.parquet'))], ignore_index=True)
raw['experiment'] = raw.exp.map({'RW': 'robustness_W_sweep', 'RC': 'robustness_cpu_allocation'})
S = []
def sec(n, rows):
    d = pd.DataFrame(rows)
    for k, v in dict(run_id=RUN, provenance='this_run', code_version=R.CODE_VERSION, script_sha256=SCRIPT_SHA, input_sha256=R.INPUT_SHA, timestamp_utc=NOW).items():
        if k not in d: d[k] = v
    d['section'] = n; S.append(d)
M = [dict(record_type='metadata', experiment='manifest', metric=f'preregistered_{k}', text_value=v) for k, v in pre.items()]
M += [dict(record_type='metadata', experiment='manifest', metric=k, text_value=str(v)) for k, v in dict(
    design='XGBoost; K=1; P0 direct; inference threads 1; A/B1/B2; 5 repeats; 200 warmup + 5000 measured; Exp RW: W in {10,30,60,120} (client 2 cores); Exp RC: W=30, client affinity 1 core {cpu0} vs 2 cores {cpu0,cpu1}',
    scope='single host, controlled conditions; not cross-environment replication; Redis affinity unchanged',
    thread_check=json.dumps(meta['thread_check']), redis_affinity=meta['redis_affinity'], run_start=meta['start'], run_end=meta['end'], leftover_keys=meta['remaining'],
    host='cloud container, 2 vCPU Intel Xeon 2.10 GHz, Redis 7.0.15 same host loopback, Python 3.13.15, sklearn 1.6.1, xgboost 3.4.1').items()]
sec('00_run_manifest', M)
lr = pd.DataFrame(dict(record_type='raw', experiment=raw.experiment, data_origin='btcusdt_real_1s', workload_type='controlled_replication_closed_loop', symbol='btcusdt',
    logical_stream_id=raw.logical_stream_id, stream_offset=raw.stream_offset, stream_local_tick=raw.stream_local_tick, request_sequence=raw.request_sequence, model='XGBoost', engine=raw.engine,
    policy='normal', repeat=raw['repeat'], seed=20260928 + raw['repeat'], W=raw.W, K=1, client_concurrency=1, d=6, dtype='float32', warmup_requests=200, measured_requests=5000,
    network_profile='P0_direct', impairment_backend='none', redis_topology='same_host_loopback_direct', tau_s1=raw.tau_s1, tau_s2=raw.tau_s2, stage2_fired=raw.stage2_fired, prediction=raw.prediction,
    latency_ms=raw.latency_ms, state_ms=raw.state_ms, stage1_ms=raw.stage1_ms, stage2_ms=raw.stage2_ms, success=raw.success, error_type=raw.error_type.replace('', None), timeout_ms=1000,
    metric='process_tick_total_latency', value=raw.latency_ms, unit='ms', status=np.where(raw.success, 'ok', 'failed'), notes='cpu_alloc=' + raw.cpu_alloc + '; order=' + raw.engine_order))
sec('07_latency_raw', lr)
rs = []
for (ex, W, ca, eng, rep), g in raw.groupby(['experiment', 'W', 'cpu_alloc', 'engine', 'repeat']):
    ok = g[g.success]; a = ok.latency_ms.to_numpy(); n = len(g)
    c = dict(record_type='run_summary', experiment=ex, model='XGBoost', engine=eng, W=W, K=1, repeat=rep, d=6, network_profile='P0_direct', client_concurrency=1, n=n, success_count=len(ok),
             failure_count=n - len(ok), aggregation='per_repeat', quantile_method=QM, status='ok', notes=f'cpu_alloc={ca}')
    for mt, v in dict(latency_mean=a.mean(), latency_sd=a.std(ddof=1), latency_p50=np.quantile(a, .5), latency_p90=np.quantile(a, .9), latency_p95=np.quantile(a, .95), latency_p99=np.quantile(a, .99),
                      state_ms_mean=ok.state_ms.mean(), state_ms_p50=ok.state_ms.median(), stage1_ms_mean=ok.stage1_ms.mean(), failure_rate=(n - len(ok)) / n, stage2_fire_rate=ok.stage2_fired.mean(),
                      host_cpu_percent_mean=g.host_cpu_mean.iloc[0]).items():
        rs.append(dict(c, metric=mt, value=v, unit=None if mt in ('failure_rate', 'stage2_fire_rate') else ('percent' if 'cpu' in mt else 'ms')))
    for dl in R.DEADLINES: rs.append(dict(c, metric='deadline_coverage', deadline_ms=dl, value=float((g.success & (g.latency_ms <= dl)).mean())))
rs = pd.DataFrame(rs); rs['ca'] = rs.notes.str[10:]
p = rs[rs.metric == 'latency_p99'].pivot_table(index=['experiment', 'W', 'ca', 'repeat'], columns='engine', values='value')
pr = [dict(record_type='run_summary', experiment=ex, model='XGBoost', engine='B1_vs_A', W=W, K=1, repeat=rep, metric=mt, value=v, aggregation='per_repeat_paired_by_repeat_and_schedule', status='ok', notes=f'cpu_alloc={ca}')
      for (ex, W, ca, rep), x in p.iterrows() for mt, v in (('p99_ratio_B1_over_A', x.B1_RedisFetch / x.A_ProposedStateful), ('p99_diff_B1_minus_A', x.B1_RedisFetch - x.A_ProposedStateful))]
pr = pd.DataFrame(pr); pr['ca'] = pr.notes.str[10:]
allr = pd.concat([rs, pr], ignore_index=True)
sec('08_latency_run_summary', allr.drop(columns='ca'))
cs = []
for (ex, W, ca, eng, mt, dl), g in allr.fillna({'deadline_ms': -1}).groupby(['experiment', 'W', 'ca', 'engine', 'metric', 'deadline_ms']):
    v = g.value.astype(float)
    for agg, val in (('mean_of_repeat_values', v.mean()), ('median_of_repeat_values', v.median()), ('sd_of_repeat_values', v.std(ddof=1)), ('min_of_repeat_values', v.min()), ('max_of_repeat_values', v.max())):
        cs.append(dict(record_type='condition_summary', experiment=ex, model='XGBoost', engine=eng, W=W, K=1, metric=mt, deadline_ms=None if dl == -1 else dl, aggregation=agg, value=val, n=len(v), status='ok',
                       inference_status='descriptive over 5 repeats; no CI', notes=f'cpu_alloc={ca}'))
# pre-registered checks
cdf = pd.DataFrame(cs); mean = lambda ex, eng, mt, ca='2core': cdf[(cdf.experiment == ex) & (cdf.engine == eng) & (cdf.metric == mt) & (cdf.aggregation == 'mean_of_repeat_values') & (cdf.notes == f'cpu_alloc={ca}')].set_index('W').value.sort_index()
chk = []
b1 = mean('robustness_W_sweep', 'B1_RedisFetch', 'state_ms_mean'); a = mean('robustness_W_sweep', 'A_ProposedStateful', 'state_ms_mean'); b2 = mean('robustness_W_sweep', 'B2_InMemoryRecompute', 'state_ms_mean')
med = cdf[(cdf.experiment == 'robustness_W_sweep') & (cdf.metric == 'p99_ratio_B1_over_A') & (cdf.aggregation == 'median_of_repeat_values')].set_index('W').value.sort_index()
from scipy.stats import spearmanr
chk += [dict(metric='B1_state_ms_monotonic_increasing_in_W', text_value=str(bool(b1.is_monotonic_increasing)).lower(), notes='means by W: ' + ', '.join(f'{w}:{v:.4f}' for w, v in b1.items())),
        dict(metric='A_state_ms_max_over_min_across_W', value=a.max() / a.min(), text_value=str(bool(a.max() / a.min() <= 1.25)).lower(), notes='rule <=1.25; means: ' + ', '.join(f'{w}:{v:.4f}' for w, v in a.items())),
        dict(metric='B2_state_ms_monotonic_increasing_in_W', text_value=str(bool(b2.is_monotonic_increasing)).lower(), notes='means: ' + ', '.join(f'{w}:{v:.4f}' for w, v in b2.items())),
        dict(metric='median_p99_ratio_spearman_vs_W', value=spearmanr(med.index, med.values).statistic, notes='descriptive only (4 points); medians: ' + ', '.join(f'{w}:{v:.3f}' for w, v in med.items()))]
for ca in ('1core', '2core'):
    m = cdf[(cdf.experiment == 'robustness_cpu_allocation') & (cdf.metric == 'p99_ratio_B1_over_A') & (cdf.aggregation == 'median_of_repeat_values') & (cdf.notes == f'cpu_alloc={ca}')].value.iloc[0]
    chk.append(dict(metric=f'cpu_{ca}_median_p99_ratio_gt_1', value=m, text_value=str(bool(m > 1)).lower()))
for ex in ('robustness_W_sweep', 'robustness_cpu_allocation'):
    for (W, ca), g in rs[(rs.experiment == ex) & (rs.engine == 'B2_InMemoryRecompute') & (rs.metric == 'latency_p99')].groupby(['W', 'ca']):
        cv = g.value.std(ddof=1) / g.value.mean(); chk.append(dict(metric='B2_p99_cv_noise_gate', W=W, value=cv, text_value='noisy' if cv > .25 else 'ok', notes=f'{ex}; cpu_alloc={ca}'))
for x in chk: x.update(record_type='check', experiment='preregistered_decision', model='XGBoost', K=1, status='computed')
sec('09_latency_condition_summary', cs + chk)
sec('14_experiment_status', [dict(record_type='status', experiment=e, status='done') for e in ('robustness_W_sweep', 'robustness_cpu_allocation')] +
    [dict(record_type='status', experiment='independent_instance_replication', status='not_run', reason='requires separate VMs prepared by the author (same image/package)')])
sec('15_limitations', [dict(record_type='limitation', experiment='limitations', metric=f'limitation_{i+1:02d}', text_value=t) for i, t in enumerate([
    'Single host (2 vCPU shared cloud container); conditions are controlled variations, not independent environments.',
    '1-core condition restricts only the client process; Redis may run on the other core.',
    'Closed-loop single client; 5 repeats; descriptive statistics only, no CI.',
    'W changes only the state window; models are unchanged (non-LSTM models do not read the window in A).'])])
out = pd.concat([s.reindex(columns=COLS) for s in S], ignore_index=True)
for c in ['success', 'stage2_fired', 'label_match']: out[c] = out[c].map(lambda v: None if pd.isna(v) else ('true' if bool(v) else 'false'))
tmp = f'{HERE}/dash_robustness_results.tmp.csv'; out.to_csv(tmp, index=False); N = len(out)
k = pd.read_csv(tmp, low_memory=False, dtype={'success': str, 'stage2_fired': str, 'label_match': str, 'text_value': str})
PK = ['section', 'record_type', 'experiment', 'engine', 'repeat', 'W', 'request_sequence', 'deadline_ms', 'metric', 'aggregation', 'notes']
lat = k[k.section == '07_latency_raw']; bad = 0
for (ex, W, nt, eng, rep), g in lat.assign(ca=lat.notes.str.extract(r'(cpu_alloc=\w+)')[0]).groupby(['experiment', 'W', 'ca', 'engine', 'repeat']):
    s = k[(k.section == '08_latency_run_summary') & (k.experiment == ex) & (k.W == W) & (k.engine == eng) & (k.repeat == rep) & (k.notes == nt) & (k.metric == 'latency_p99')].value
    bad += (len(g) != 5000) + (not np.isclose(s.iloc[0], np.quantile(g[g.success == 'true'].latency_ms, .99)))
V = dict(columns=list(k.columns) == COLS, rows=len(k) == N, no_dup_pk=not k.duplicated(PK).any(), lat_ok=bool((lat.latency_ms >= 0).all()), recompute=bad == 0)
print(V)
if all(V.values()): os.replace(tmp, f'{HERE}/dash_robustness_results.csv'); print('written', N)
