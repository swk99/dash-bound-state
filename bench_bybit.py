import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')
import config as cfg

cfg.SYMBOL = 'btcusdt_bybit'

from benchmark import run_k_benchmark
import pandas as pd
import numpy as np

all_lat = []

for model in ['XGBoost', 'RandomForest', 'Logistic']:
    df_lat, df_thr = run_k_benchmark(
        model_name=model,
        k_list=cfg.K_LIST,
        n_warmup=200,
        n_measure=2000,
        seed=42,
        offset_stride=50,
    )
    all_lat.append(df_lat)
    df_lat.to_csv(f'results/latency_k_{model}_bybit.csv', index=False)
    print(f'{model} done')

print('\n=== Bybit BTCUSDT Results ===')
for model in ['XGBoost', 'RandomForest', 'Logistic']:
    df = pd.read_csv(f'results/latency_k_{model}_bybit.csv')
    print(f'\n[{model}]')
    for k in [1, 5, 10, 20, 50]:
        dash = df[(df['engine']=='A_Proposed') & (df['k']==k)]['total_ms']
        redis = df[(df['engine']=='B1_RedisFetch') & (df['k']==k)]['total_ms']
        print(f'  K={k:2d}  DASH P50={dash.quantile(0.5):.3f} P99={dash.quantile(0.99):.3f}  '
              f'Redis P50={redis.quantile(0.5):.3f} P99={redis.quantile(0.99):.3f}')
