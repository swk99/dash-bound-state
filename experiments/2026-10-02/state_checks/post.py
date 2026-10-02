import sys, glob, os, hashlib, json, numpy as np, pandas as pd, warnings
sys.path.insert(0, '/home/claude/dash/dash-bound-state-main'); sys.path.insert(0, '.')
from analyze_latency import summarize
df = pd.read_csv('dash_paper_results.csv', low_memory=False)
RUN = df.run_id.iloc[0]; CV = df.code_version.iloc[0]; TS = pd.Timestamp.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
cols = list(df.columns); new = []
def add(**kw): new.append(dict(dict(run_id=RUN, code_version=CV, timestamp_utc=TS), **kw))
NODEP = 'dependence-aware CI not computed (tick latencies are serially dependent; no block-bootstrap block length validated in this run)'
# --- E2 summary
m = df[df.record_type == 'memory_raw']
for (W, N), g in m.groupby(['W', 'N']):
    v = g.redis_mem_bytes.astype(float)
    for st, val in (('mean', v.mean()), ('sd', v.std(ddof=1)), ('min', v.min()), ('max', v.max())):
        add(record_type='summary', experiment='E2_state_only', engine='shared_A_B1_layout', W=W, N=N, d=6, dtype='float32',
            metric=f'redis_mem_bytes_across_repeats_{st}', value=val, unit='bytes', n=len(v), status='ok',
            logical_payload_bytes=int(g.logical_payload_bytes.iloc[0]), redis_list_len=int(g.redis_list_len.iloc[0]),
            redis_encoding=g.redis_encoding.iloc[0], redis_key_bytes=int(g.redis_key_bytes.iloc[0]))
# --- gating summary
g0 = df[df.experiment == 'gating_ablation']
for (mod, pol), g in g0.groupby(['model', 'policy']):
    x = g.latency_ms.astype(float).to_numpy()
    add(record_type='summary', experiment='gating_ablation', model=mod, engine='A_ProposedStateful', policy=pol, W=30, metric='stage2_fire_rate',
        value=g.stage2_fired.astype(str).eq('True').mean(), n=len(g), status='ok', tau_s1=g.tau_s1.iloc[0], tau_s2=g.tau_s2.iloc[0])
    for row in summarize(x, [1, 5, 10, 20, 50, 100, 1000], seed=20260928):
        if row['metric'] == 'deadline_coverage':
            add(record_type='summary', experiment='gating_ablation', model=mod, engine='A_ProposedStateful', policy=pol, W=30, metric='deadline_coverage',
                value=row['value'], deadline_ms=row['deadline_ms'], n=row['n'], status='ok', notes='empirical; shared cloud container')
        else:
            add(record_type='summary', experiment='gating_ablation', model=mod, engine='A_ProposedStateful', policy=pol, W=30, metric=row['metric'], value=row['value'],
                unit='ms', n=row['n'], ci_low=row['ci_low'], ci_high=row['ci_high'], ci_method='IID_BCa (exploratory)', status=row['status'],
                notes='IID assumption not verified; exploratory only. ' + NODEP)
    add(record_type='summary', experiment='gating_ablation', model=mod, engine='A_ProposedStateful', policy=pol, W=30, metric='mean', value=x.mean(), unit='ms', n=len(x), status='ok')
# --- Lua cost dependence note
add(record_type='experiment_status', experiment='lua_update_cost', metric='dependence_aware_ci', status='not_computed', notes=NODEP)
# --- E6 on existing per-tick logs
R = '/home/claude/dash/dash-bound-state-main/results/'
cand = ['latency_k_Logistic.csv', 'latency_k_RandomForest.csv', 'latency_k_XGBoost.csv', 'latency_k_Logistic_2025_11.csv',
        'latency_k_RandomForest_2025_11.csv', 'latency_k_XGBoost_2025_11.csv', 'latency_k_sweep_FULL.csv', 'latency_multisymbol_mixed.csv']
data = {f: pd.read_csv(R + f) for f in cand}
key = lambda d: pd.util.hash_pandas_object(d.reset_index(drop=True), index=False)
hashes = {f: set(key(d)) for f, d in data.items()}
TAU_NOW = float(df[(df.experiment == 'E0_label_agreement') & (df.model == 'Logistic')].tau_s1.dropna().iloc[0])
used = []
for f in cand:
    others = set().union(*[hashes[o] for o in used]) if used else set()
    ov = len(hashes[f] & others) / len(hashes[f])
    sha = hashlib.sha256(open(R + f, 'rb').read()).hexdigest()
    if ov > 0.999:
        add(record_type='experiment_status', experiment='E6_deadline_coverage', source_file=f, metric='duplicate_excluded', value=ov, status='excluded_duplicate',
            notes=f'{ov:.4f} of rows already present in previously counted files {used}; sha256={sha}'); continue
    if ov > 0:
        add(record_type='experiment_status', experiment='E6_deadline_coverage', source_file=f, metric='partial_overlap', value=ov, status='excluded_partial_overlap',
            notes=f'partial row overlap with {used}; excluded to avoid double counting; sha256={sha}'); continue
    used.append(f); d = data[f]
    gb = [c for c in ['model', 'engine', 'k', 'lookback_w', 'tau', 'logical_symbol'] if c in d]
    for keyv, g in d.groupby(gb):
        meta = dict(zip(gb, keyv if isinstance(keyv, tuple) else (keyv,)))
        hist = 'historical' if abs(meta.get('tau', np.nan) - TAU_NOW) > 1e-6 else 'current'
        for dl in [1, 5, 10, 20, 50, 100, 1000]:
            add(record_type='summary', experiment='E6_deadline_coverage', source_file=f, data_origin='existing per-tick log (not regenerated)',
                model=meta.get('model'), engine=meta.get('engine'), K=meta.get('k'), W=meta.get('lookback_w'), metric='deadline_coverage',
                value=float((g.total_ms <= dl).mean()), deadline_ms=dl, n=len(g), tau_s1=meta.get('tau'), status=hist,
                notes=(f"log tau={meta.get('tau')} != current calibrated tau_s1={TAU_NOW:.6f} (pre-recalibration log); " if hist == 'historical' else '')
                      + (f"logical_symbol={meta['logical_symbol']}; " if 'logical_symbol' in meta else '') + 'latency_column=total_ms; Coverage x Accuracy not claimed as utility')
    add(record_type='experiment_status', experiment='E6_deadline_coverage', source_file=f, metric='included', status='done', n=len(d), notes=f'sha256={sha}')
# --- status rows
for exp, st, note in [('lua_correctness', 'done', 'W=10/30/60/120, 2W inputs, 3 methods'), ('E2_state_only', 'done', 'W=10/30/60/120 x 5 repeats, N checkpoints W/1000/5000/10000/20000 in one pass'),
                      ('E0_label_agreement', 'done', 'Logistic/RF/XGB, 1000 contiguous real ticks, W=30, normal gating'),
                      ('lua_update_cost', 'done', 'W=30 d=6, 5 repeats, 200 warmup + 2000 measured per method, randomized order'),
                      ('E6_deadline_coverage', 'done', 'existing logs only; duplicates excluded'), ('gating_ablation', 'done', '1000 ticks/model, separate state per policy, randomized per-tick order'),
                      ('E1_capacity', 'not_run', 'excluded by scope (T2/T3)'), ('noscript_recovery', 'skipped', 'no independent test server designated'),
                      ('breakpoint_or_pvalue', 'not_run', 'not computed; none claimed')]:
    add(record_type='experiment_status', experiment=exp, metric='status', status=st, notes=note)
out = pd.concat([df, pd.DataFrame(new)], ignore_index=True)[cols]
out.to_csv('dash_paper_results.csv', index=False)
print(len(df), len(new), 'used E6 files:', used)
