"""DASH JSS paper evidence -> dash_paper_results.csv (long format). Single process, sequential."""
import sys, os, time, json, uuid, hashlib, glob, re, platform, subprocess, datetime as dt
REPO = '/home/claude/dash/dash-bound-state-main'; sys.path.insert(0, REPO)
import numpy as np, pandas as pd, redis, psutil
import config as cfg
from redis_state import RedisWindowWriter
from engine import _to_feat_vec, ProposedStatefulEngine, RedisFetchBaselineEngine, InMemoryRecomputeEngine
OUT = '/home/claude/jss_run2'; SEED = 20260928
RUN = 'dash_' + dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
PFX = f"jss:{RUN}:"
r = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CODE = hashlib.sha256(b''.join(sha(f).encode() for f in sorted(glob.glob(f'{REPO}/*.py')) + [__file__])).hexdigest()[:16]
CODE_VERSION = f"no-git; sha256(sorted repo *.py + this script)[:16]={CODE}"
COLS = "record_type experiment run_id timestamp_utc code_version source_file data_origin model engine update_method policy repeat seed W N K d dtype tick_index metric value unit n ci_low ci_high ci_method status notes redis_key redis_key_bytes redis_encoding redis_list_len redis_mem_bytes logical_payload_bytes rss_bytes tau_s1 tau_s2 stage2_fired prediction reference_prediction label_match latency_ms deadline_ms".split()
ROWS = []
def now(): return dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
def add(record_type, experiment, **kw):
    row = dict(record_type=record_type, experiment=experiment, run_id=RUN, timestamp_utc=now(), code_version=CODE_VERSION)
    row.update(kw); ROWS.append(row)
def save():
    pd.DataFrame(ROWS, columns=COLS).to_csv(f'{OUT}/dash_paper_results.csv', index=False)
LOG = open(f'{OUT}/progress.log', 'a'); T0 = time.time(); _last = [0.0]
def log(msg, force=False):
    if force or time.time() - _last[0] >= 30:
        _last[0] = time.time(); LOG.write(f"{now()} +{time.time()-T0:.0f}s {msg}\n"); LOG.flush()
created = set()
def K(*a):
    k = PFX + ':'.join(map(str, a)); created.add(k); return k
D = len(cfg.FEATURE_COLS)
fs = sorted(glob.glob('/home/claude/pq/btcusdt/flush_*.parquet'), key=lambda f: int(re.findall(r'flush_(\d+)', f)[0]))
DF = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
X = DF[cfg.FEATURE_COLS].tail(20000).to_numpy(np.float32)
ORIGIN = f"btcusdt flush_{re.findall(r'flush_(\d+)',fs[0])[0]}..{re.findall(r'flush_(\d+)',fs[-1])[0]} ({len(DF)} rows, 1s bars), newest 20000 rows, cols={cfg.FEATURE_COLS}"

# ---------------- environment ----------------
import sklearn, xgboost, torch
info = r.info('server')
env = {'python': platform.python_version(), 'redis_server': info['redis_version'], 'redis_py': redis.__version__,
       'numpy': np.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__, 'torch': torch.__version__,
       'pandas': pd.__version__, 'os': platform.platform(), 'cpu': open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t'),
       'cpu_count': os.cpu_count(), 'mem_available_bytes': psutil.virtual_memory().available, 'mem_total_bytes': psutil.virtual_memory().total,
       'redis_location': f'same host ({cfg.REDIS_HOST}:{cfg.REDIS_PORT}, loopback TCP); Redis started inside this cloud container for testing, not the original experiment server',
       'redis_maxmemory_policy': r.config_get('maxmemory-policy')['maxmemory-policy'],
       'host_note': 'shared cloud container (2 vCPU); latency numbers not equivalent to the original experiment PC; README specifies Python 3.11',
       'seed': SEED, 'seed_rule': 'repeat r uses seed+r (Lua-cost method order shuffle; gating per-tick order); E2/E0 inputs are deterministic real rows',
       'redis_key_prefix': PFX, 'data_origin': ORIGIN, 'input_sha256': hashlib.sha256(X.tobytes()).hexdigest()}
for k_, v in env.items(): add('environment', 'environment', metric=k_, value=v, status='recorded')
for f in sorted(glob.glob(f'{REPO}/*.pkl') + glob.glob(f'{REPO}/*.json') + glob.glob(f'{REPO}/*.pth') + glob.glob(f'{REPO}/artifacts/*')):
    add('environment', 'environment', source_file=os.path.relpath(f, REPO), metric='sha256', value=sha(f), status='recorded')
save(); log(f"start {RUN}", True)
w = RedisWindowWriter(r)
WS = (10, 30, 60, 120)
try:
    # ================= 1) state correctness (W=10..120) =================
    for W in WS:
        n = 2 * W
        ks = {m: K('state', m, W) for m in ('sequential', 'pipeline_nontx', 'lua')}
        ok_len = ok_head = True
        for t in range(n):
            p = _to_feat_vec(X[t]).tobytes()
            r.lpush(ks['sequential'], p); r.ltrim(ks['sequential'], 0, W - 1)
            pl = r.pipeline(transaction=False); pl.lpush(ks['pipeline_nontx'], p); pl.ltrim(ks['pipeline_nontx'], 0, W - 1); pl.execute()
            w.push(ks['lua'], p, W)
            exp = min(t + 1, W)
            ok_len &= all(r.llen(k) == exp for k in ks.values())
            ok_head &= all(r.lindex(k, 0) == p for k in ks.values())
        lists = {m: r.lrange(k, 0, -1) for m, k in ks.items()}
        expected = [_to_feat_vec(X[t]).tobytes() for t in range(n - 1, n - 1 - W, -1)]
        checks = {'len_eq_min_t_W_every_update': ok_len, 'newest_first_head_every_update': ok_head,
                  'final_list_identical_3_methods': lists['sequential'] == lists['pipeline_nontx'] == lists['lua'],
                  'final_list_eq_expected_newest_first_bytes': lists['lua'] == expected,
                  'entry_payload_bytes_eq_4d': all(len(b) == 4 * D for b in lists['lua'])}
        for m_, v in checks.items():
            add('state_check', 'lua_correctness', data_origin=ORIGIN, update_method='sequential|pipeline_nontx|lua', W=W, N=n, d=D, dtype='float32',
                metric=m_, value=bool(v), status='pass' if v else 'fail', n=n)
    add('state_check', 'lua_correctness', metric='script_re_register', value=RedisWindowWriter(r).push(K('state', 'rereg'), X[0].tobytes(), 10) == 1, status='pass')
    add('state_check', 'lua_correctness', metric='noscript_recovery', status='skipped',
        notes='not run: no separate independent test server designated; SCRIPT FLUSH on this Redis not performed in this run (earlier exploratory run 636958a3b7ab did flush on the dedicated local server and recovered)')
    save(); log("state correctness done", True)

    # ================= 2) state-only E2 =================
    # A and B1 call the same RedisWindowWriter.push(f"dash:state:{sym}", feat.tobytes(), W) -> one shared layout
    tE = time.time(); total = len(WS) * 5 * 20000; done = 0
    for W in WS:
        cps = sorted({W, 1000, 5000, 10000, 20000})
        for rep in range(5):
            k = K('e2', f'W{W:03d}', f'r{rep}')
            for t in range(20000):
                w.push(k, _to_feat_vec(X[t]).tobytes(), W); done += 1
                if t + 1 in cps:
                    mu = r.memory_usage(k, samples=0); enc = r.object('encoding', k)
                    add('memory_raw', 'E2_state_only', data_origin=ORIGIN, engine='shared_A_B1_layout', update_method='lua', repeat=rep, seed=SEED + rep,
                        W=W, N=t + 1, d=D, dtype='float32', metric='redis_memory_usage_samples0', value=int(mu), unit='bytes', status='ok',
                        redis_key=k, redis_key_bytes=len(k.encode()), redis_encoding=enc.decode() if isinstance(enc, bytes) else enc,
                        redis_list_len=r.llen(k), redis_mem_bytes=int(mu), logical_payload_bytes=4 * D * min(t + 1, W),
                        rss_bytes=psutil.Process().memory_info().rss, notes='no model/scaler/inference/LRANGE; MEMORY USAGE includes key-name and object overhead')
                if done % 5000 == 0:
                    el = time.time() - tE; log(f"E2 W={W} repeat={rep} N={t+1} ticks={done}/{total} elapsed={el:.0f}s eta={el/done*(total-done):.0f}s")
            r.delete(k)
        save(); log(f"E2 W={W} saved", True)

    # ================= 3) E0 real artifacts, 1000 contiguous ticks, W=30 =================
    from models import load_dash_harness
    X0 = X[:1000]
    for m in ('Logistic', 'RandomForest', 'XGBoost'):
        h = load_dash_harness(m); h.lookback_w = 30
        engs = {'A_ProposedStateful': (ProposedStatefulEngine(h, r), K('e0', m, 'A')),
                'B1_RedisFetch': (RedisFetchBaselineEngine(h, r), K('e0', m, 'B1'))}
        b2 = InMemoryRecomputeEngine(h, X0)
        mism = 0
        for i in range(1000):
            outs = {}
            for name, (e, k) in engs.items():
                outs[name] = e.process_tick(k.removeprefix('dash:state:'), X0[i])
            outs['B2_InMemoryRecompute'] = b2.process_tick(i)
            ref = outs['B2_InMemoryRecompute']['yhat']
            allm = len({o['yhat'] for o in outs.values()}) == 1; mism += (not allm)
            for name, o in outs.items():
                add('prediction_tick', 'E0_label_agreement', data_origin=ORIGIN, model=m, engine=name, policy='normal', W=30, d=D, dtype='float32',
                    tick_index=i, metric='yhat', value=o['yhat'], tau_s1=h.tau_conf, tau_s2=h.tau_s2, stage2_fired=o['stage2_fired'],
                    prediction=o['yhat'], reference_prediction=ref, label_match=o['yhat'] == ref, status='ok' if allm else 'mismatch',
                    notes='reference = B2_InMemoryRecompute; agreement between paths, not accuracy (no ground truth)')
        for name in list(engs) + ['B2_InMemoryRecompute']:
            fired = np.mean([x['stage2_fired'] for x in ROWS if x.get('experiment') == 'E0_label_agreement' and x['model'] == m and x['engine'] == name])
            add('summary', 'E0_label_agreement', model=m, engine=name, policy='normal', W=30, metric='stage2_fire_rate', value=float(fired), n=1000, tau_s1=h.tau_conf, tau_s2=h.tau_s2, status='ok')
        add('summary', 'E0_label_agreement', model=m, engine='A|B1|B2', policy='normal', W=30, metric='label_agreement_rate', value=1 - mism / 1000, n=1000,
            status='pass' if mism == 0 else 'fail', notes=f'mismatch_ticks={mism}; Redis state keys under prefix {PFX}')
        for _, k in engs.values(): r.delete('dash:state:' + k.removeprefix('dash:state:'))
        save(); log(f"E0 {m} mism={mism}", True)

    # ================= 4) Lua update cost: W=30,d=6, 5 repeats, warmup 200, measure 2000, random order =================
    W = 30; methods = ['sequential', 'pipeline_nontx', 'lua']
    RedisWindowWriter(r)  # script registered before measuring (redis-py loads on first call; warmup covers it)
    for rep in range(5):
        order = list(np.random.default_rng(SEED + rep).permutation(methods))
        for meth in order:
            k = K('cost', rep, meth)
            def upd(p):
                if meth == 'sequential': r.lpush(k, p); r.ltrim(k, 0, W - 1)
                elif meth == 'pipeline_nontx':
                    pl = r.pipeline(transaction=False); pl.lpush(k, p); pl.ltrim(k, 0, W - 1); pl.execute()
                else: w.push(k, p, W)
            for t in range(200): upd(_to_feat_vec(X[t]).tobytes())
            lat = []
            for t in range(2000):
                p = _to_feat_vec(X[200 + t]).tobytes(); t0 = time.perf_counter_ns(); upd(p); lat.append((time.perf_counter_ns() - t0) / 1e6)
            for t, l in enumerate(lat):
                add('latency_tick', 'lua_update_cost', data_origin=ORIGIN, update_method=meth, repeat=rep, seed=SEED + rep, W=W, d=D, dtype='float32',
                    tick_index=t, metric='update_latency', value=l, unit='ms', latency_ms=l, status='ok', notes=f'order={"|".join(order)}; no inference')
            a = np.array(lat)
            for q in (50, 95, 99):
                add('summary', 'lua_update_cost', update_method=meth, repeat=rep, seed=SEED + rep, W=W, d=D, metric=f'p{q}', value=float(np.percentile(a, q)), unit='ms', n=len(a), status='ok', notes='per-repeat quantile')
            add('summary', 'lua_update_cost', update_method=meth, repeat=rep, W=W, d=D, metric='mean', value=float(a.mean()), unit='ms', n=len(a), status='ok', notes='per-repeat')
            r.delete(k)
        save(); log(f"cost repeat {rep} order={order}", True)
    # across-repeat stats of per-repeat quantiles
    cs = pd.DataFrame([x for x in ROWS if x['experiment'] == 'lua_update_cost' and x['record_type'] == 'summary'])
    for (meth, met), g in cs.groupby(['update_method', 'metric']):
        v = g['value'].astype(float)
        for st, val in (('mean', v.mean()), ('sd', v.std(ddof=1)), ('min', v.min()), ('max', v.max())):
            add('summary', 'lua_update_cost', update_method=meth, W=30, d=D, metric=f'{met}_across_repeats_{st}', value=float(val), unit='ms', n=len(v), status='ok',
                notes='statistic over 5 repeat-level values')
    save()

    # ================= 5) gating ablation =================
    for m in ('Logistic', 'RandomForest', 'XGBoost'):
        h = load_dash_harness(m); h.lookback_w = 30
        eN, eF = ProposedStatefulEngine(h, r), ProposedStatefulEngine(h, r)
        sN, sF = K('gate', m, 'normal').removeprefix('dash:state:'), K('gate', m, 'force').removeprefix('dash:state:')
        rng = np.random.default_rng(SEED)
        for i in range(1000):
            first = rng.random() < 0.5
            pairs = [('normal', eN, sN, False), ('force', eF, sF, True)]
            if not first: pairs.reverse()
            for pol, e, s, f in pairs:
                o = e.process_tick(s, X0[i], force_stage2=f)
                add('latency_tick', 'gating_ablation', data_origin=ORIGIN, model=m, engine='A_ProposedStateful', policy=pol, W=30, d=D, dtype='float32',
                    tick_index=i, metric='total_latency', value=o['total_ms'], unit='ms', latency_ms=o['total_ms'], tau_s1=h.tau_conf, tau_s2=h.tau_s2,
                    stage2_fired=o['stage2_fired'], prediction=o['yhat'], status='ok', notes=f'first_policy={pairs[0][0]}')
        r.delete(*['dash:state:' + s for s in (sN, sF)])
        save(); log(f"gating {m} done", True)
finally:
    left_before = list(r.scan_iter(PFX + '*')) + list(r.scan_iter('dash:state:' + PFX + '*'))
    if left_before: r.delete(*left_before)
    for k in created: r.delete(k)
    left = len(list(r.scan_iter(PFX + '*'))) + len(list(r.scan_iter('dash:state:' + PFX + '*')))
    add('experiment_status', 'cleanup', metric='leftover_experiment_keys', value=left, status='ok' if left == 0 else 'fail', notes='deleted only keys under run prefix')
    save(); log(f"cleanup leftover={left}", True)
