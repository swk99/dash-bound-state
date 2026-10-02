"""DASH two-environment P0 replication + rule-selected Stage-2 E0.  Portable: same file runs on cloud and experiment PC.
Env vars: DASH_REPO (repo dir), DASH_DATA_GLOB (flush_*.parquet glob), DASH_ENV_LABEL (e.g. cloud / labpc), REDIS_HOST, REDIS_PORT, DASH_OUT.
Fixed: XGBoost, W=30, d=6, K in {1,100}, P0 DIRECT Redis (no proxy), A/B1/B2, 5 repeats, 200 warmup + 5000 measured per engine/condition/repeat,
model inference threads pinned to 1. Output: dash_replication_results.csv (single header, sectioned long format)."""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[v] = '1'
import sys, time, json, glob, re, hashlib, platform, threading, io, contextlib, warnings, datetime as dt, socket
REPO = os.environ.get('DASH_REPO', '/home/claude/dash/dash-bound-state-main'); sys.path.insert(0, REPO)
DATA_GLOB = os.environ.get('DASH_DATA_GLOB', '/home/claude/pq/btcusdt/flush_*.parquet')
ENV_LABEL = os.environ.get('DASH_ENV_LABEL', 'cloud'); OUT = os.environ.get('DASH_OUT', os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, redis, psutil, torch
torch.set_num_threads(1)
import config as cfg
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine, InMemoryRecomputeEngine
from models import load_dash_harness
SEED = 20260928; W = 30; WARM = 200; MEAS = 5000; KS = (1, 100); REPS = 5; DEADLINES = [1, 5, 10, 20, 50, 100, 1000]; TIMEOUT_S = 1.0
RH, RP = os.environ.get('REDIS_HOST', cfg.REDIS_HOST), int(os.environ.get('REDIS_PORT', cfg.REDIS_PORT))
RUN = f"rep_{ENV_LABEL}_" + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'); PFX = f'jss4:{RUN}:'
CK = f'{OUT}/ckpt_{ENV_LABEL}'; os.makedirs(CK, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
SCRIPT_SHA = sha(os.path.abspath(__file__))
CODE = hashlib.sha256(''.join(sha(f) for f in sorted(glob.glob(f'{REPO}/*.py'))).encode()).hexdigest()
CODE_VERSION = f'nogit-sha16:{CODE[:16]}'
LOG = open(f'{OUT}/progress_{ENV_LABEL}.log', 'a'); T0 = time.time()
def utc(): return dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
def log(m): LOG.write(f'{utc()} +{time.time()-T0:.0f}s {m}\n'); LOG.flush(); print(m, flush=True)
# ---------------- data ----------------
FS = sorted(glob.glob(DATA_GLOB), key=lambda f: int(re.findall(r'flush_(\d+)', f)[0]))
DFALL = pd.concat([pd.read_parquet(f) for f in FS], ignore_index=True)
XALL = np.ascontiguousarray(DFALL[cfg.FEATURE_COLS].to_numpy(np.float32)); INPUT_SHA = hashlib.sha256(XALL.tobytes()).hexdigest()
MANIFEST_SHA = hashlib.sha256(''.join(f'{os.path.basename(f)}:{sha(f)}\n' for f in FS).encode()).hexdigest()
ART_FILES = ['s1_xgb_H30_a2_L94.json', 's2_xgb_H30_a2_L94.json', 's1_rf_H30_a2_L94.pkl', 's2_rf_H30_a2_L94.pkl', 'artifacts/s1_lr_H30_a2_L94.pkl', 's2_lr_H30_a2_L94.pkl', 'scaler_H30_a2_L94.pkl', 'thresholds_H30_a2_L94.json']
ART = {n: sha(os.path.join(REPO, n)) for n in ART_FILES}
# ---------------- SHA256 check against the cloud run (expected_sha256.json shipped with the package) ----------------
SHA_CHECK = 'no expected_sha256.json'
_exp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expected_sha256.json')
if os.path.exists(_exp):
    E = json.load(open(_exp)); mism = [k for k, v in E['artifacts'].items() if ART.get(k) != v]
    mism += [k for k in ('input_array_sha256', 'data_file_manifest_sha256', 'code_sha256') if E[k] != {'input_array_sha256': INPUT_SHA, 'data_file_manifest_sha256': MANIFEST_SHA, 'code_sha256': CODE}[k]]
    if E['script_sha256'] != SCRIPT_SHA: mism.append('run_replication.py')
    SHA_CHECK = 'all_match' if not mism else 'MISMATCH:' + ','.join(mism)
    print('SHA256 check:', SHA_CHECK, flush=True)
    if mism and os.environ.get('DASH_ALLOW_MISMATCH') != '1': raise SystemExit('SHA256 mismatch vs cloud run; set DASH_ALLOW_MISMATCH=1 to run anyway (will be recorded)')
# ---------------- pre-registered decision rules (written before any measurement) ----------------
PREREG = {
 'direction_rule': 'per environment and K: count repeats with P99_B1/P99_A > 1 (out of 5) and sign of median ratio minus 1; direction "maintained" across environments iff median-ratio sign agrees in both environments for that K',
 'noise_gate_rule': 'B2 (no Redis) control: coefficient of variation of per-repeat P99 across 5 repeats; if CV > 0.25 the environment is flagged noisy and A/B1 differences there are reported as inconclusive',
 'scope_rule': 'if the median ratio is <= 1.10 or signs disagree, report "benefit limited without added network delay"',
 'e0_window_rule': 'per model: on the full array, compute Stage-1 pi_1 only (scaler + s1.predict_proba); choose the FIRST contiguous 1000-row window starting at offset multiple of 100 (start >= 0) whose fraction pi_1 >= tau_s1 is >= 0.20; then run A/B1/B2 once on it; labels are not inspected before selection',
 'pooling_rule': 'raw data of different environments are never pooled into one quantile'}
# ---------------- helpers ----------------
def pin_threads(h):
    out = {}
    for nm in ('s1', 's2'):
        m = getattr(h, nm)
        if hasattr(m, 'get_booster'):
            m.set_params(n_jobs=1); m.get_booster().set_param({'nthread': 1})
            out[nm] = json.loads(m.get_booster().save_config())['learner']['generic_param'].get('nthread')
        elif hasattr(m, 'n_jobs'):
            m.n_jobs = 1; out[nm] = m.n_jobs
        else: out[nm] = 'n/a'
    return out
def load(model):
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(buf):
        warnings.simplefilter('always'); h = load_dash_harness(model)
    h.lookback_w = W; th = pin_threads(h)
    return h, th, buf.getvalue().strip().replace('\n', ' / '), [str(x.message)[:200] for x in ws if 'version' in str(x.message).lower()]
def client(): return redis.Redis(host=RH, port=RP, socket_timeout=TIMEOUT_S, socket_connect_timeout=TIMEOUT_S)
class CpuMon:
    """samples host CPU% every 1 s in a background thread (outside the timed code path; small interference)"""
    def __init__(self): self.v = []; self.stop = False; self.t = threading.Thread(target=self.run, daemon=True); psutil.cpu_percent(None); self.t.start()
    def run(self):
        while not self.stop: self.v.append(psutil.cpu_percent(interval=1.0))
    def close(self): self.stop = True; self.t.join(2); return self.v
def schedule(K):
    per = int(np.ceil((WARM + MEAS) / K)); L = W + per
    offs = [j * L for j in range(K)]; assert K * L <= len(XALL)
    rr = [(j, W + i) for i in range(per) for j in range(K)][:WARM + MEAS]
    return offs, L, rr
def run_engine(eng, h, r, K, offs, L, rr, tag):
    streams = [XALL[o:o + L] for o in offs]
    if eng == 'B2': engs = [InMemoryRecomputeEngine(h, s) for s in streams]
    else: e = (ProposedStatefulEngine if eng == 'A' else RedisFetchBaselineEngine)(h, r)
    sym = lambda j: f'{PFX}{tag}:{eng:>2}:s{j:04d}'
    def tick(j, t): return engs[j].process_tick(t) if eng == 'B2' else e.process_tick(sym(j), streams[j][t])
    for j in range(K):
        for t in range(W): tick(j, t)
    rows = []
    for seq, (j, t) in enumerate(rr):
        t0 = time.perf_counter_ns(); err = ''
        try: o = tick(j, t); ok = True
        except Exception as ex: o = {}; ok = False; err = type(ex).__name__
        wall = (time.perf_counter_ns() - t0) / 1e6
        if seq >= WARM: rows.append((seq - WARM, j, offs[j], t, ok, err, o.get('total_ms', wall) if ok else wall, o.get('A_ms'), o.get('B1_ms'), o.get('B2_ms'), o.get('stage2_fired'), o.get('yhat')))
    return rows, ([f'dash:state:{sym(j)}' for j in range(K)] if eng != 'B2' else [])
RCOLS = ['request_sequence', 'logical_stream_id', 'stream_offset', 'stream_local_tick', 'success', 'error_type', 'latency_ms', 'state_ms', 'stage1_ms', 'stage2_ms', 'stage2_fired', 'prediction']
def main():
    log(f'start {RUN} env={ENV_LABEL} redis={RH}:{RP}')
    json.dump(PREREG, open(f'{OUT}/preregistered_rules_{ENV_LABEL}.json', 'w'), indent=1)
    r = client(); r.ping(); created = set(); cpu_rows = []
    h, th, lout, vw = load('XGBoost'); log(f'XGBoost threads {th}')
    try:
        # ---- P0 direct latency ----
        for K in KS:
            offs, L, rr = schedule(K)
            for rep in range(REPS):
                f = f'{CK}/lat__K{K}__r{rep}.parquet'
                if os.path.exists(f): continue
                order = list(np.random.default_rng([SEED, rep, K, 0, 4]).permutation(['A', 'B1', 'B2'])); parts = []
                for i, eng in enumerate(order):
                    mon = CpuMon(); tE = time.time()
                    rows, keys = run_engine(eng, h, r, K, offs, L, rr, f'K{K}:r{rep}'); created |= set(keys)
                    cv = mon.close()
                    if keys: r.delete(*keys); created -= set(keys)
                    d = pd.DataFrame(rows, columns=RCOLS); d['engine'] = {'A': 'A_ProposedStateful', 'B1': 'B1_RedisFetch', 'B2': 'B2_InMemoryRecompute'}[eng]
                    d['engine_order'] = '>'.join(order); parts.append(d)
                    cpu_rows.append(dict(K=K, repeat=rep, engine=d.engine.iloc[0], cpu_mean=float(np.mean(cv)) if cv else np.nan, cpu_max=float(np.max(cv)) if cv else np.nan, n=len(cv)))
                    log(f'P0direct K={K} repeat={rep} engine={eng} {len(d)} req {time.time()-tE:.1f}s hostcpu_mean={np.mean(cv) if cv else float("nan"):.0f}%')
                d = pd.concat(parts); d['K'] = K; d['repeat'] = rep; d['tau_s1'] = h.tau_conf; d['tau_s2'] = h.tau_s2
                d.to_parquet(f + '.tmp'); os.replace(f + '.tmp', f)
                pd.DataFrame(cpu_rows).to_csv(f'{CK}/cpu.csv', index=False)
        # ---- Stage-2 E0 (rule-selected window) ----
        for model in ('Logistic', 'RandomForest', 'XGBoost'):
            f = f'{CK}/e0__{model}.parquet'
            if os.path.exists(f): continue
            hm, thm, _, _ = load(model)
            xs = (XALL - hm._mu) / hm._sd
            p1 = hm.s1.predict_proba(xs)[:, 1]
            fire = p1 >= hm.tau_conf
            start = next((s for s in range(0, len(XALL) - 1000 + 1, 100) if fire[s:s + 1000].mean() >= 0.20), None)
            meta = dict(model=model, tau_s1=hm.tau_conf, tau_s2=hm.tau_s2, dataset_fire_rate=float(fire.mean()), window_start=start,
                        window_fire_rate_pi1=None if start is None else float(fire[start:start + 1000].mean()), threads=json.dumps(thm))
            json.dump(meta, open(f'{CK}/e0__{model}.json', 'w'))
            if start is None: log(f'E0 {model}: no window met rule'); continue
            X0 = XALL[start:start + 1000]; ka, kb = f'{PFX}e0:{model}: A', f'{PFX}e0:{model}:B1'; created |= {f'dash:state:{ka}', f'dash:state:{kb}'}
            eA, eB, eC = ProposedStatefulEngine(hm, r), RedisFetchBaselineEngine(hm, r), InMemoryRecomputeEngine(hm, X0)
            rows = []
            for i in range(1000):
                oa, ob, oc = eA.process_tick(ka, X0[i]), eB.process_tick(kb, X0[i]), eC.process_tick(i)
                for nm, o in (('A_ProposedStateful', oa), ('B1_RedisFetch', ob), ('B2_InMemoryRecompute', oc)):
                    rows.append((model, nm, i, start + i, o['yhat'], oa['yhat'], o['yhat'] == oa['yhat'], o['stage2_fired']))
            r.delete(f'dash:state:{ka}', f'dash:state:{kb}')
            pd.DataFrame(rows, columns=['model', 'engine', 'tick_index', 'global_row', 'prediction', 'reference_prediction', 'label_match', 'stage2_fired']).to_parquet(f)
            log(f'E0 {model} window={start} fire={meta["window_fire_rate_pi1"]:.3f}')
    finally:
        left = list(r.scan_iter(f'*{PFX}*'))
        if left: r.delete(*left)
        log(f'cleanup leftover_deleted={len(left)} remaining={len(list(r.scan_iter(f"*{PFX}*")))}')
    try: rinfo = redis.Redis(host=RH, port=RP).info('server').get('redis_version')
    except Exception: rinfo = None
    try: cpu = open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': \t')
    except Exception: cpu = platform.processor()
    json.dump(dict(run=RUN, thread_check=th, loader_stdout=lout, version_warnings=vw, end=utc(), redis_version=rinfo, cpu_model=cpu, platform=platform.platform(),
                   python=platform.python_version(), vcpu=os.cpu_count(), ram_total_bytes=psutil.virtual_memory().total, sha_check=SHA_CHECK), open(f'{CK}/run_meta.json', 'w'))
if __name__ == '__main__':
    main()
