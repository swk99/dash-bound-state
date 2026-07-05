import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')

import time
import numpy as np
import pandas as pd
import redis

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine
from tooling import MinioHandler

rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)

SYMBOLS_POOL = ['btcusdt_1', 'ethusdt_2', 'solusdt_2']
K_LIST = [3, 6, 9, 15, 30, 50]
N_WARMUP = 200
N_EVAL = 1000
MODEL = 'XGBoost'

handler = MinioHandler()
data_by_symbol = {}
for sym in SYMBOLS_POOL:
    needed = N_WARMUP + N_EVAL + 64
    d = handler.load_historical_features(symbol=sym, n_rows=needed, allow_dummy=False)
    if isinstance(d, pd.DataFrame):
        d = d.values
    data_by_symbol[sym] = np.asarray(d, dtype=np.float32)
    print(f"{sym}: loaded {len(d)} rows")

harness = load_dash_harness(
    model_name=MODEL, lambda_val=cfg.LAMBDA, alpha=cfg.ALPHA,
    h=cfg.HORIZON_H, tau_conf=cfg.TAU_CONF, lookback_w=cfg.LOOKBACK_W,
)
engine_A = ProposedStatefulEngine(harness, rds)
engine_B1 = RedisFetchBaselineEngine(harness, rds)

print("\n=== Multi-Symbol Heterogeneity Test ===")
header = "{:>4}  {:>9} {:>9}  {:>10} {:>10}  {:>6}".format("K", "DASH_P50", "DASH_P99", "Redis_P50", "Redis_P99", "Gain")
print(header)

results = []
for K in K_LIST:
    symbols = [f"multisym_{K}_{i}" for i in range(K)]
    base_symbols = [SYMBOLS_POOL[i % len(SYMBOLS_POOL)] for i in range(K)]

    keys = [f"dash:state:{s}" for s in symbols]
    if keys:
        rds.delete(*keys)

    for t in range(N_WARMUP):
        idx = t
        for sym, base in zip(symbols, base_symbols):
            feat = data_by_symbol[base][idx % len(data_by_symbol[base])]
            engine_A.process_tick(sym, feat, cfg.TAU_CONF)
            engine_B1.process_tick(sym, feat, cfg.TAU_CONF)

    lat_a = []
    for t in range(N_EVAL):
        sym_idx = t % K
        sym = symbols[sym_idx]
        base = base_symbols[sym_idx]
        idx = (N_WARMUP + t // K) % len(data_by_symbol[base])
        feat = data_by_symbol[base][idx]
        t0 = time.perf_counter_ns()
        engine_A.process_tick(sym, feat, cfg.TAU_CONF)
        t1 = time.perf_counter_ns()
        lat_a.append((t1 - t0) / 1e6)

    lat_b1 = []
    for t in range(N_EVAL):
        sym_idx = t % K
        sym = symbols[sym_idx]
        base = base_symbols[sym_idx]
        idx = (N_WARMUP + t // K) % len(data_by_symbol[base])
        feat = data_by_symbol[base][idx]
        t0 = time.perf_counter_ns()
        engine_B1.process_tick(sym, feat, cfg.TAU_CONF)
        t1 = time.perf_counter_ns()
        lat_b1.append((t1 - t0) / 1e6)

    lat_a = np.array(lat_a)
    lat_b1 = np.array(lat_b1)
    dash_p50, dash_p99 = np.percentile(lat_a, [50, 99])
    redis_p50, redis_p99 = np.percentile(lat_b1, [50, 99])
    gain = redis_p99 / dash_p99

    row = "{:>4}  {:>9.3f} {:>9.3f}  {:>10.3f} {:>10.3f}  {:>5.2f}x".format(K, dash_p50, dash_p99, redis_p50, redis_p99, gain)
    print(row)
    results.append({"K": K, "dash_p50": dash_p50, "dash_p99": dash_p99,
                     "redis_p50": redis_p50, "redis_p99": redis_p99, "gain": gain})

df = pd.DataFrame(results)
df.to_csv("results/multisymbol_real.csv", index=False)
print("\n[OK] saved results/multisymbol_real.csv")
