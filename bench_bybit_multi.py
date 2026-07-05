import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')
import config as cfg
import numpy as np
import pandas as pd
import redis
import time
from models import load_dash_harness
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine
from tooling import MinioHandler

rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)

SYMBOLS_POOL = ['btcusdt_bybit', 'ethusdt_bybit', 'solusdt_bybit']
K_LIST = [3, 6, 9, 15, 30, 50]
N_WARMUP = 200
N_EVAL = 1000
N_TRIALS = 5
MODEL = 'XGBoost'

handler = MinioHandler()
data_by_symbol = {}
for sym in SYMBOLS_POOL:
    needed = N_WARMUP + N_EVAL + 64
    d = handler.load_historical_features(symbol=sym, n_rows=needed, allow_dummy=False)
    if isinstance(d, pd.DataFrame):
        d = d.values
    data_by_symbol[sym] = np.asarray(d, dtype=np.float32)
    print(f'{sym}: loaded {len(d)} rows')

harness = load_dash_harness(
    model_name=MODEL, lambda_val=cfg.LAMBDA, alpha=cfg.ALPHA,
    h=cfg.HORIZON_H, tau_conf=cfg.TAU_CONF, lookback_w=cfg.LOOKBACK_W,
)

print('\n=== Bybit Multi-Asset (BTC/ETH/SOL) K-sweep ===')
all_results = []

for trial in range(N_TRIALS):
    print(f'\n[Trial {trial+1}/{N_TRIALS}]')
    engine_A = ProposedStatefulEngine(harness, rds)
    engine_B1 = RedisFetchBaselineEngine(harness, rds)

    for K in K_LIST:
        symbols = [f'bybit_multi_{trial}_{K}_{i}' for i in range(K)]
        base_symbols = [SYMBOLS_POOL[i % len(SYMBOLS_POOL)] for i in range(K)]
        keys = [f'dash:state:{s}' for s in symbols]
        if keys: rds.delete(*keys)

        for t in range(N_WARMUP):
            for sym, base in zip(symbols, base_symbols):
                feat = data_by_symbol[base][t % len(data_by_symbol[base])]
                engine_A.process_tick(sym, feat, cfg.TAU_CONF)
                engine_B1.process_tick(sym, feat, cfg.TAU_CONF)

        lat_a, lat_b1 = [], []
        for t in range(N_EVAL):
            sym_idx = t % K
            sym = symbols[sym_idx]
            base = base_symbols[sym_idx]
            idx = (N_WARMUP + t // K) % len(data_by_symbol[base])
            feat = data_by_symbol[base][idx]

            t0 = time.perf_counter_ns()
            engine_A.process_tick(sym, feat, cfg.TAU_CONF)
            lat_a.append((time.perf_counter_ns() - t0) / 1e6)

            t0 = time.perf_counter_ns()
            engine_B1.process_tick(sym, feat, cfg.TAU_CONF)
            lat_b1.append((time.perf_counter_ns() - t0) / 1e6)

        dash_p99 = np.percentile(lat_a, 99)
        redis_p99 = np.percentile(lat_b1, 99)
        all_results.append({'trial': trial, 'K': K,
                             'dash_p99': dash_p99, 'redis_p99': redis_p99,
                             'dash_faster': dash_p99 < redis_p99})
        print(f'  K={K:3d}  DASH={dash_p99:.3f}  Redis={redis_p99:.3f}  {"DASH faster" if dash_p99 < redis_p99 else "Redis faster"}')

df = pd.DataFrame(all_results)
df.to_csv('results/bybit_multisymbol.csv', index=False)

print('\n=== Summary ===')
for K in K_LIST:
    sub = df[df['K']==K]
    n_faster = sub['dash_faster'].sum()
    print(f'K={K:3d}  DASH faster {n_faster}/{N_TRIALS}  '
          f'DASH={sub.dash_p99.mean():.3f}+-{sub.dash_p99.std():.3f}  '
          f'Redis={sub.redis_p99.mean():.3f}+-{sub.redis_p99.std():.3f}')

print('\n[OK] saved results/bybit_multisymbol.csv')
