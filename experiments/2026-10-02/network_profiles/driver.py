import os, sys, time, json, numpy as np, pandas as pd
os.environ.setdefault('RUN_ID', 'dash3_20261002T1122Z')
sys.path.insert(0, '/home/claude/jss_run3'); import bench as b
from redis_state import RedisWindowWriter
deadline = time.time() + 88 * 60   # hard budget guard
def ping_cal(prof, tag, n=500):
    f = f'{b.CK}/ping__{tag}.parquet'
    if os.path.exists(f): return
    px = b.Proxy(prof, b.SEED); r = b.client(); L = []
    try:
        for i in range(n):
            t = time.perf_counter_ns(); ok = True
            try: r.ping()
            except Exception: ok = False
            L.append((i, (time.perf_counter_ns() - t) / 1e6, ok))
    finally: px.close()
    d = pd.DataFrame(L, columns=['request_sequence', 'ping_rtt_ms', 'success']); d['network_profile'] = prof; d['tag'] = tag
    d.to_parquet(f); a = d.ping_rtt_ms[50:]; b.log(f'ping {tag} mean={a.mean():.3f} p99={a.quantile(.99):.3f}', True)
def lua_cost():
    f = f'{b.CK}/lua_cost.parquet'
    if os.path.exists(f): return
    px = b.Proxy('P0', b.SEED); r = b.client(); w = RedisWindowWriter(r); rows = []
    X = b.XALL
    try:
        w.push(f'{b.PFX}luacost:reg', X[0].tobytes(), 30); r.delete(f'{b.PFX}luacost:reg')   # register before measuring
        for rep in range(5):
            order = list(np.random.default_rng([b.SEED, rep, 99]).permutation(['seq', 'pip', 'lua']))
            for meth in order:
                k = f'{b.PFX}luacost:r{rep}:{meth}'   # same length (3-char method codes)
                def upd(p):
                    if meth == 'seq': r.lpush(k, p); r.ltrim(k, 0, 29)
                    elif meth == 'pip':
                        pl = r.pipeline(transaction=False); pl.lpush(k, p); pl.ltrim(k, 0, 29); pl.execute()
                    else: w.push(k, p, 30)
                for t in range(200): upd(X[t].tobytes())
                for t in range(2000):
                    p = X[200 + t].tobytes(); t0 = time.perf_counter_ns(); upd(p)
                    rows.append((rep, meth, '|'.join(order), t, (time.perf_counter_ns() - t0) / 1e6, len(k)))
                r.delete(k)
    finally: px.close()
    pd.DataFrame(rows, columns=['repeat', 'update_method', 'order', 'request_sequence', 'latency_ms', 'redis_key_bytes']).to_parquet(f)
    b.log('lua cost done', True)
def gating(prof):
    f = f'{b.CK}/gating__XGBoost__K1__{prof}.parquet'
    if os.path.exists(f): return
    h = b.load_dash_harness('XGBoost'); h.lookback_w = 30
    px = b.Proxy(prof, b.SEED); r = b.client(); rows = []
    X = b.XALL; L = 30 + 200 + 2000
    try:
        for rep in range(5):
            eN, eF = b.ProposedStatefulEngine(h, r), b.ProposedStatefulEngine(h, r)
            sN, sF = f'{b.PFX}gate:{prof}:r{rep}:nor', f'{b.PFX}gate:{prof}:r{rep}:frc'
            rng = np.random.default_rng([b.SEED, rep, 7, int(prof[1])])
            for t in range(L):
                pairs = [('normal', eN, sN, False), ('force_stage2', eF, sF, True)]
                if rng.random() < 0.5: pairs.reverse()
                for pol, e, s, fr in pairs:
                    t0 = time.perf_counter_ns(); ok = True; err = ''
                    try: o = e.process_tick(s, X[t], force_stage2=fr)
                    except Exception as ex: o = {}; ok = False; err = type(ex).__name__
                    wall = (time.perf_counter_ns() - t0) / 1e6
                    if t >= 230:
                        rows.append((rep, pol, t - 230, t, ok, err, o.get('total_ms', wall), o.get('A_ms'), o.get('B1_ms'), o.get('B2_ms'), o.get('stage2_fired'), o.get('yhat'), pairs[0][0]))
            r.delete(*[f'dash:state:{s}' for s in (sN, sF)])
            b.log(f'gating {prof} repeat {rep} done', True)
    finally: px.close()
    d = pd.DataFrame(rows, columns=['repeat', 'policy', 'request_sequence', 'stream_local_tick', 'success', 'error_type', 'latency_ms', 'state_ms', 'stage1_ms', 'stage2_ms', 'stage2_fired', 'prediction', 'first_policy'])
    d['tau_s1'] = h.tau_conf; d['tau_s2'] = h.tau_s2; d.to_parquet(f)
b.log(f'driver start run={b.RUN}', True)
for prof in ['P0', 'P1', 'P2', 'P3', 'P4']:
    ping_cal(prof, prof)
    if prof != 'P0': ping_cal('P0', f'P0_restore_after_{prof}')
h = b.load_dash_harness('XGBoost'); h.lookback_w = b.W
for prof in ['P0', 'P1', 'P2']:
    px = b.Proxy(prof, b.SEED); r = b.client()
    try:
        for K in (1, 10, 100):
            b.run_condition('XGBoost', K, prof, h, r)
            b.log(f'condition done XGBoost K={K} {prof} remaining_budget={(deadline-time.time())/60:.1f}min', True)
    finally: px.close()
json.dump({'primary_done': True, 't': b.utc()}, open(f'{b.OUT}/status_primary.json', 'w'))
lua_cost(); gating('P0'); gating('P2')
est = 23 * 60
if deadline - time.time() > est * 1.1:
    px = b.Proxy('P3', b.SEED); r = b.client()
    try: b.run_condition('XGBoost', 100, 'P3', h, r)
    finally: px.close()
    json.dump({'secondary_P3': 'done'}, open(f'{b.OUT}/status_secondary.json', 'w'))
else:
    json.dump({'secondary_P3': 'not_run', 'reason': 'remaining budget < estimate'}, open(f'{b.OUT}/status_secondary.json', 'w'))
left = list(redis.Redis().scan_iter('*jss3:*')) if (redis := __import__('redis')) else []
if left: redis.Redis().delete(*left)
b.log(f'driver end leftover_keys_deleted={len(left)}', True)
