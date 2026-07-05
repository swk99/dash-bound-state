"""
tau_sensitivity.py
------------------
tau_conf 값을 변화시키면서 SLA miss rate 변화 측정.

실행:
  python tau_sensitivity.py
"""

import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')

import time
import numpy as np
import pandas as pd
import redis
from pathlib import Path

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine
from tooling import MinioHandler

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# tau_conf sweep 범위
TAU_LIST = [0.50, 0.55, 0.60, 0.6731, 0.70, 0.75, 0.80]
SLA_MS   = 10.0
N_WARMUP = 200
N_EVAL   = 2000
K        = 10
MODEL    = "XGBoost"

rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)

# 데이터 로드
handler = MinioHandler()
needed  = K * 50 + N_WARMUP + N_EVAL + 64
needed  = max(needed, 3000)
data = handler.load_historical_features(
    symbol=cfg.SYMBOL, n_rows=needed, allow_dummy=False
)
if isinstance(data, pd.DataFrame):
    data = data.values
data = np.asarray(data, dtype=np.float32)

print(f"[*] tau_conf sensitivity | model={MODEL} K={K} SLA={SLA_MS}ms")
print(f"{'tau_conf':>10}  {'InvRate':>9}  {'P50_ms':>8}  {'P99_ms':>8}  {'MissRate':>9}")
print("-" * 55)

rows = []
for tau in TAU_LIST:
    harness = load_dash_harness(
        model_name=MODEL,
        lambda_val=cfg.LAMBDA,
        alpha=cfg.ALPHA,
        h=cfg.HORIZON_H,
        tau_conf=tau,
        lookback_w=cfg.LOOKBACK_W,
    )
    engine = ProposedStatefulEngine(harness, rds)
    symbols = [f"sym_tau_{tau:.4f}_{i}" for i in range(K)]

    # flush redis
    keys = [f"dash:state:{s}" for s in symbols]
    if keys:
        rds.delete(*keys)

    # warmup
    for t in range(N_WARMUP):
        sym = symbols[t % K]
        engine.process_tick(sym, data[t], tau)

    # measure
    lat = []
    invoked = 0
    for t in range(N_EVAL):
        sym = symbols[t % K]
        idx = N_WARMUP + (t // K)
        t0 = time.perf_counter_ns()
        res = engine.process_tick(sym, data[idx], tau)
        t1 = time.perf_counter_ns()
        ms = (t1 - t0) / 1e6
        lat.append(ms)
        if res.get("B2_ms", 0) > 0:
            invoked += 1

    lat = np.array(lat)
    inv_rate = invoked / N_EVAL
    p50 = float(np.percentile(lat, 50))
    p99 = float(np.percentile(lat, 99))
    miss = float((lat > SLA_MS).mean())

    print(f"{tau:>10.4f}  {inv_rate:>9.4f}  {p50:>8.3f}  {p99:>8.3f}  {miss:>9.4f}")
    rows.append({
        "tau_conf": tau, "inv_rate": inv_rate,
        "p50_ms": p50, "p99_ms": p99, "miss_rate": miss,
        "model": MODEL, "K": K, "SLA_ms": SLA_MS,
    })

df = pd.DataFrame(rows)
out = RESULTS_DIR / "tau_sensitivity.csv"
df.to_csv(out, index=False)
print(f"\n[OK] saved {out}")
