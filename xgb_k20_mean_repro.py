import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')

import numpy as np
import pandas as pd
import redis

import config as cfg
from benchmark import run_k_benchmark

# XGBoost K=20에서 DASH mean latency > Redis-Fetch mean latency 가 재현되는지 확인
N_TRIALS = 5
K_LIST = [10, 20, 50]   # 비교 기준점도 같이 봄

print("=== XGBoost K=20 Mean-Latency Inversion Reproducibility Test ===")
print(f"Running {N_TRIALS} independent trials with different seeds\n")

results = []
for trial in range(N_TRIALS):
    seed = 200 + trial * 23
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
        redis_b = df_lat[(df_lat['engine'] == 'B1_RedisFetch') & (df_lat['k'] == k)]['total_ms']
        dash_mean = dash.mean()
        redis_mean = redis_b.mean()
        diff = dash_mean - redis_mean
        results.append({
            "trial": trial, "seed": seed, "k": k,
            "dash_mean": dash_mean, "redis_mean": redis_mean,
            "diff": diff, "dash_slower": diff > 0,
        })
        flag = "  <-- DASH SLOWER" if diff > 0 else ""
        print(f"  K={k:3d}  DASH_mean={dash_mean:.3f}  Redis_mean={redis_mean:.3f}  diff={diff:+.3f}{flag}")

df = pd.DataFrame(results)
df.to_csv("results/xgb_k20_mean_repro.csv", index=False)

print("\n=== Summary: how often was DASH slower (mean) at each K? ===")
for k in K_LIST:
    sub = df[df['k'] == k]
    n_slower = sub['dash_slower'].sum()
    print(f"K={k:3d}  DASH slower in {n_slower}/{N_TRIALS} trials  "
          f"(mean diff={sub['diff'].mean():+.3f}, std={sub['diff'].std():.3f})")

print("\n[OK] saved results/xgb_k20_mean_repro.csv")
