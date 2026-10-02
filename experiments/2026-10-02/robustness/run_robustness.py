"""Controlled-condition robustness on ONE host (not cross-environment replication).
Exp RW: W in {10,30,60,120}; Exp RC: client CPU affinity {1 core, 2 cores} at W=30. XGBoost, K=1, P0 direct, threads=1, A/B1/B2, 5 repeats, 200 warmup + 5000 measured."""
import os, sys, json, time
os.environ.setdefault('DASH_OUT', '/home/claude/jss_run5'); os.environ.setdefault('DASH_ENV_LABEL', 'cloud_robust')
sys.path.insert(0, '/home/claude/jss_run4'); import run_replication as R
import numpy as np, pandas as pd
OUT = os.environ['DASH_OUT']; CK = f'{OUT}/ckpt'; os.makedirs(CK, exist_ok=True)
WS = (10, 30, 60, 120); CORES = {'1core': {0}, '2core': {0, 1}}; K = 1; REPS = 5
PREREG = dict(
  design='per repeat r: Exp RW runs W in permuted order default_rng([20260928,r,11]); Exp RC runs {1core,2core} in permuted order default_rng([20260928,r,12]); within each condition engines in order default_rng([20260928,r,W_or_coreidx,13]); fresh keys per engine',
  prediction_RW='B1 state_ms (push+LRANGE+decode) increases monotonically with W; A state_ms (push only, non-LSTM) does not depend on W (max/min of mean state_ms across W <= 1.25); B2 state_ms increases with W (O(W) recompute)',
  prediction_RW_tail='P99_B1/P99_A median across repeats increases with W (Spearman sign over 4 W values, descriptive only)',
  prediction_RC='restricting the client to 1 core does not reverse the A/B1 direction at W=30 (median ratio > 1 in both)',
  noise_gate='B2 P99 CV across repeats > 0.25 => condition flagged noisy',
  affinity_scope='only the client process affinity is changed (os.sched_setaffinity); Redis server affinity unchanged (all CPUs)',
  label='single host, controlled conditions; not cross-environment replication')
def main():
    json.dump(PREREG, open(f'{OUT}/preregistered_rules_robustness.json', 'w'), indent=1)
    r = R.client(); r.ping(); h, th, lout, vw = R.load('XGBoost'); R.log(f'robust start threads={th}')
    meta = dict(thread_check=th, loader_stdout=lout, version_warnings=vw, redis_affinity=os.popen(f'taskset -p $(pgrep -o redis-server)').read().strip(), start=R.utc())
    try:
        for rep in range(REPS):
            os.sched_setaffinity(0, {0, 1})
            for W in np.random.default_rng([R.SEED, rep, 11]).permutation(WS):
                f = f'{CK}/RW__W{W}__r{rep}.parquet'
                if os.path.exists(f): continue
                run_cond('RW', int(W), '2core', rep, h, r, f, int(W))
            for ci in np.random.default_rng([R.SEED, rep, 12]).permutation(2):
                cname = list(CORES)[ci]; f = f'{CK}/RC__{cname}__r{rep}.parquet'
                if os.path.exists(f): continue
                os.sched_setaffinity(0, CORES[cname]); run_cond('RC', 30, cname, rep, h, r, f, 100 + ci)
            os.sched_setaffinity(0, {0, 1})
    finally:
        left = list(r.scan_iter(f'*{R.PFX}*'))
        if left: r.delete(*left)
        meta.update(end=R.utc(), leftover_deleted=len(left), remaining=len(list(r.scan_iter(f'*{R.PFX}*'))))
        json.dump(meta, open(f'{CK}/run_meta.json', 'w')); R.log(f'robust end {meta["leftover_deleted"]} {meta["remaining"]}')
def run_cond(exp, W, cname, rep, h, r, f, tag):
    R.W = W; h.lookback_w = W
    offs, L, rr = R.schedule(K)
    order = list(np.random.default_rng([R.SEED, rep, tag, 13]).permutation(['A', 'B1', 'B2'])); parts = []; cpu = []
    for eng in order:
        mon = R.CpuMon(); tE = time.time()
        rows, keys = R.run_engine(eng, h, r, K, offs, L, rr, f'{exp}:W{W}:{cname}:r{rep}')
        cv = mon.close()
        if keys: r.delete(*keys)
        d = pd.DataFrame(rows, columns=R.RCOLS); d['engine'] = {'A': 'A_ProposedStateful', 'B1': 'B1_RedisFetch', 'B2': 'B2_InMemoryRecompute'}[eng]
        d['engine_order'] = '>'.join(order); d['host_cpu_mean'] = float(np.mean(cv)) if cv else np.nan; parts.append(d)
        R.log(f'{exp} W={W} cpu={cname} repeat={rep} engine={eng} {len(d)} req {time.time()-tE:.1f}s affinity={sorted(os.sched_getaffinity(0))}')
    d = pd.concat(parts); d['exp'] = exp; d['W'] = W; d['cpu_alloc'] = cname; d['repeat'] = rep; d['tau_s1'] = h.tau_conf; d['tau_s2'] = h.tau_s2
    d.to_parquet(f + '.tmp'); os.replace(f + '.tmp', f)
if __name__ == '__main__':
    main()
