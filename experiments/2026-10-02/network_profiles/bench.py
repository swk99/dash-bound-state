"""DASH JSS integrated latency benchmark: K resident logical streams, single client (concurrency=1), closed loop.
Redis traffic via experiment-only delay proxy (all profiles incl. P0). Raw rows -> parquet checkpoint per condition."""
import sys, os, time, json, glob, re, uuid, hashlib, subprocess, datetime as dt
REPO = '/home/claude/dash/dash-bound-state-main'; sys.path.insert(0, REPO)
import numpy as np, pandas as pd, redis
import config as cfg
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine, InMemoryRecomputeEngine
from models import load_dash_harness
OUT = '/home/claude/jss_run3'; CK = f'{OUT}/ckpt'; os.makedirs(CK, exist_ok=True)
SEED = 20260928; W = 30; WARM = 200; MEAS = 5000; PROXY_PORT = 26379; TIMEOUT_S = 1.0
PROFILES = {'P0': (0, 0), 'P1': (1, 0), 'P2': (1, 0.5), 'P3': (5, 0), 'P4': (5, 2)}
RUN = os.environ.get('RUN_ID') or ('dash3_' + dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'))
PFX = f'jss3:{RUN}:'
LOG = open(f'{OUT}/progress.log', 'a'); T0 = time.time(); _last = [0.0]
def utc(): return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
def log(msg, force=False):
    if force or time.time() - _last[0] >= 30:
        _last[0] = time.time(); LOG.write(f'{utc()} +{time.time()-T0:.0f}s {msg}\n'); LOG.flush()
# ---- data (prepared outside timing) ----
FS = sorted(glob.glob('/home/claude/pq/btcusdt/flush_*.parquet'), key=lambda f: int(re.findall(r'flush_(\d+)', f)[0]))
DFALL = pd.concat([pd.read_parquet(f) for f in FS], ignore_index=True)
XALL = np.ascontiguousarray(DFALL[cfg.FEATURE_COLS].to_numpy(np.float32))
def schedule(K):
    """stream j uses contiguous rows [off_j, off_j+L); request order: fill (W per stream, stream-major), then global
    round-robin over streams for WARM + MEAS requests. Non-overlapping offsets if they fit, else overlapping (recorded)."""
    per = int(np.ceil((WARM + MEAS) / K)); L = W + per
    overlap = K * L > len(XALL)
    stride = L if not overlap else max(1, (len(XALL) - L) // max(1, K - 1))
    offs = [j * stride for j in range(K)]
    rr = [(j, W + i) for i in range(per) for j in range(K)][:WARM + MEAS]   # (stream, local tick)
    return offs, L, rr, overlap
class Proxy:
    def __init__(self, prof, seed):
        d, j = PROFILES[prof]
        self.p = subprocess.Popen([sys.executable, f'{OUT}/delay_proxy.py', str(PROXY_PORT), str(cfg.REDIS_PORT), str(d), str(j), str(seed)], stdout=subprocess.PIPE)
        assert self.p.stdout.readline().strip() == b'ready'
    def close(self): self.p.terminate(); self.p.wait(5)
def client(): return redis.Redis(host='127.0.0.1', port=PROXY_PORT, socket_timeout=TIMEOUT_S, socket_connect_timeout=TIMEOUT_S)
def run_engine(eng_name, h, r, K, offs, L, rr, tag):
    """returns raw rows for measured requests only"""
    streams = [XALL[o:o + L] for o in offs]
    if eng_name == 'A': engs = [ProposedStatefulEngine(h, r)] * 1
    elif eng_name == 'B1': engs = [RedisFetchBaselineEngine(h, r)] * 1
    else: engs = [InMemoryRecomputeEngine(h, s) for s in streams]
    e = engs[0]
    sym = lambda j: f'{PFX}{tag}:{eng_name:>2}:s{j:04d}'   # same-length per engine (A padded to 2 chars)
    def tick(j, t):
        if eng_name == 'B2': return engs[j].process_tick(t)
        return e.process_tick(sym(j), streams[j][t])
    for j in range(K):
        for t in range(W): tick(j, t)
    rows = []
    for seq, (j, t) in enumerate(rr):
        t0 = time.perf_counter_ns(); err = ''
        try: o = tick(j, t); ok = True
        except Exception as ex: o = {}; ok = False; err = type(ex).__name__
        wall = (time.perf_counter_ns() - t0) / 1e6
        if seq >= WARM:
            rows.append((seq - WARM, j, offs[j], t, ok, err, o.get('total_ms', wall) if ok else wall, wall,
                         o.get('A_ms'), o.get('B1_ms'), o.get('B2_ms'), o.get('stage2_fired'), o.get('yhat')))
    keys = [f'dash:state:{sym(j)}' for j in range(K)] if eng_name != 'B2' else []
    return rows, keys
COLS = ['request_sequence', 'logical_stream_id', 'stream_offset', 'stream_local_tick', 'success', 'error_type', 'latency_ms', 'outer_wall_ms',
        'state_ms', 'stage1_ms', 'stage2_ms', 'stage2_fired', 'prediction']
def run_condition(model, K, prof, h, r, repeats=5, engines=('A', 'B1', 'B2'), exp='latency_matrix'):
    offs, L, rr, overlap = schedule(K)
    for rep in range(repeats):
        f = f'{CK}/{exp}__{model}__K{K}__{prof}__r{rep}.parquet'
        if os.path.exists(f): continue
        order = list(np.random.default_rng([SEED, rep, K, list(PROFILES).index(prof)]).permutation(list(engines)))
        parts = []; created = []
        try:
            for i, eng in enumerate(order):
                tE = time.time()
                rows, keys = run_engine(eng, h, r, K, offs, L, rr, f'{exp}:{model}:K{K}:{prof}:r{rep}'); created += keys
                d = pd.DataFrame(rows, columns=COLS); d['engine'] = {'A': 'A_ProposedStateful', 'B1': 'B1_RedisFetch', 'B2': 'B2_InMemoryRecompute'}[eng]
                d['engine_order'] = '|'.join(order); d['engine_position'] = i
                parts.append(d)
                log(f'{exp} model={model} K={K} W={W} profile={prof} repeat={rep} engine={eng} done {len(d)} req in {time.time()-tE:.1f}s', True)
                if created: r.delete(*created); created = []
        finally:
            if created: r.delete(*created)
        d = pd.concat(parts); d['model'] = model; d['K'] = K; d['network_profile'] = prof; d['repeat'] = rep
        d['seed'] = SEED + rep; d['overlapping_replication'] = overlap; d['tau_s1'] = h.tau_conf; d['tau_s2'] = h.tau_s2
        d.to_parquet(f + '.tmp'); os.replace(f + '.tmp', f)
