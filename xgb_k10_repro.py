import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')

import numpy as np
import pandas as pd
import redis

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine
from tooling import MinioHandler
from benchmark import run_k_benchmark

# 5번 반복 실행해서 K=10 스파이크가 reproducible한지 확인
N_TRIALS = 5
K_LIST = [5, 10, 20]

print("=== XGBoost K=10 Spike Reproducibility Test ===")
print(f"Running {N_TRIALS} independent trials with different seeds\n")

results = []
for trial in range(N_TRIALS):
    seed = 100 + trial * 17  # 다른 seed
    print(f"[Trial {trial+1}/{N_TRIALS}] seed={seed}")

    df_lat, df_summary = run_k_benchmark(
        model_name="XGBoost",
        k_list=K_LIST,
        n_warmup=200,
        n_measure=2000,
        seed=seed,
        offset_stride=50,
    )

    for k in K_LIST:
        dash = df_lat[(df_lat['engine'] == 'A_Proposed') & (df_lat['k'] == k)]['total_ms']
        p99 = dash.quantile(0.99)
        p50 = dash.quantile(0.50)
        results.append({"trial": trial, "seed": seed, "k": k, "p50": p50, "p99": p99})
        print(f"  K={k:3d}  P50={p50:.3f}  P99={p99:.3f}")

df = pd.DataFrame(results)
df.to_csv("results/xgb_k10_repro.csv", index=False)

print("\n=== Summary (mean +/- std across trials) ===")
for k in K_LIST:
    sub = df[df['k'] == k]
    print(f"K={k:3d}  P99 mean={sub['p99'].mean():.3f}  std={sub['p99'].std():.3f}  "
          f"min={sub['p99'].min():.3f}  max={sub['p99'].max():.3f}")

print("\n[OK] saved results/xgb_k10_repro.csv")
