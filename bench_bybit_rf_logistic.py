import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')
import config as cfg
import numpy as np
import pandas as pd
from benchmark import run_k_benchmark

cfg.SYMBOL = 'btcusdt_bybit'

N_TRIALS = 5
K_LIST = [1, 5, 10, 20, 50]
MODELS = ['RandomForest', 'Logistic']

all_results = []

for MODEL in MODELS:
    print(f'\n=== Bybit BTCUSDT - {MODEL} ===')
    for trial in range(N_TRIALS):
        seed = 500 + trial * 41
        print(f'  [Trial {trial+1}/5] seed={seed}')
        df_lat, _ = run_k_benchmark(
            model_name=MODEL, k_list=K_LIST,
            n_warmup=200, n_measure=2000,
            seed=seed, offset_stride=50,
        )
        for k in K_LIST:
            dash = df_lat[(df_lat['engine']=='A_Proposed') & (df_lat['k']==k)]['total_ms']
            redis = df_lat[(df_lat['engine']=='B1_RedisFetch') & (df_lat['k']==k)]['total_ms']
            all_results.append({
                'model': MODEL, 'trial': trial, 'k': k,
                'dash_p99': dash.quantile(0.99),
                'redis_p99': redis.quantile(0.99),
                'dash_faster': dash.quantile(0.99) < redis.quantile(0.99),
            })

df = pd.DataFrame(all_results)
df.to_csv('results/bybit_rf_logistic.csv', index=False)

print('\n=== Summary ===')
for MODEL in MODELS:
    print(f'\n[{MODEL}]')
    sub = df[df['model']==MODEL]
    for k in K_LIST:
        s = sub[sub['k']==k]
        n = s['dash_faster'].sum()
        print(f'  K={k:3d}  DASH faster {n}/5  '
              f'DASH={s.dash_p99.mean():.3f}+-{s.dash_p99.std():.3f}  '
              f'Redis={s.redis_p99.mean():.3f}+-{s.redis_p99.std():.3f}')

print('\n[OK] saved results/bybit_rf_logistic.csv')
