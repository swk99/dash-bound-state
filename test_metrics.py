import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
import sys
sys.path.insert(0, r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH')
import config as cfg
from tooling import MinioHandler

# 데이터 로드
tag = 'H30_a2_L94'
print('[*] 데이터 로딩...')
handler = MinioHandler()
df = handler.load_historical_features(
    symbol='btcusdt_1',
    n_rows=230000,
    cols=['sec'] + list(cfg.FEATURE_COLS),
    return_df=True,
    enforce_time_sort=True,
    time_col='sec',
)
print(f'    loaded {len(df):,} rows')

# forward return 계산
from labeling import forward_return
df['future_r'] = forward_return(df['r_t'], cfg.HORIZON_H)
thr = cfg.ALPHA * np.sqrt(cfg.HORIZON_H) * df['sigma_hat']
df['y'] = 0
df.loc[df['future_r'] >= thr, 'y'] = 2
df.loc[df['future_r'] <= -thr, 'y'] = 1
df = df[df['future_r'].notna()].copy()

# chronological split 70/15/15
H = cfg.HORIZON_H
n = len(df)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train + H : n_train + H + n_val]
test_df  = df.iloc[n_train + H + n_val + H:]
print(f'    test rows: {len(test_df):,}')

# scaler 로드
ART = Path(r'C:\Users\sogon\OneDrive - Goldsmiths College\research_1_DASH\artifacts')
sc = joblib.load(ART / f'scaler_{tag}.pkl')
mu = np.array(sc['mu'], dtype=np.float32)
sd = np.array(sc['sd'], dtype=np.float32)

X_test = test_df[list(cfg.FEATURE_COLS)].values.astype(np.float32)
X_test = (X_test - mu) / sd
y_test = test_df['y'].values
y_te_b = (y_test > 0).astype(int)

# thresholds
thr_json = json.load(open(ART / f'thresholds_{tag}.json'))

print('\n=== Test-Set Metrics ===')
for model_name, key in [('XGBoost','xgb'), ('RandomForest','rf'), ('Logistic','lr')]:
    thr_s1 = thr_json['models'][key]['s1']['thr']
    thr_s2 = thr_json['models'][key]['s2']['thr']

    if model_name == 'XGBoost':
        s1 = XGBClassifier(); s1.load_model(str(ART / f's1_xgb_{tag}.json'))
        s2 = XGBClassifier(); s2.load_model(str(ART / f's2_xgb_{tag}.json'))
    elif model_name == 'RandomForest':
        s1 = joblib.load(ART / f's1_rf_{tag}.pkl')
        s2 = joblib.load(ART / f's2_rf_{tag}.pkl')
    else:
        s1 = joblib.load(ART / f's1_lr_{tag}.pkl')
        s2 = joblib.load(ART / f's2_lr_{tag}.pkl')

    p1 = s1.predict_proba(X_test)[:, 1]
    y_pred_s1 = (p1 >= thr_s1).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_te_b, y_pred_s1, average='binary', zero_division=0)

    shock_mask = y_pred_s1 == 1
    if shock_mask.sum() > 0:
        y_te_sh = (y_test[shock_mask] == 2).astype(int)
        p2 = s2.predict_proba(X_test[shock_mask])[:, 1]
        y_pred_s2 = (p2 >= thr_s2).astype(int)
        p2_, r2_, f2_, _ = precision_recall_fscore_support(y_te_sh, y_pred_s2, average='binary', zero_division=0)
    else:
        p2_, r2_, f2_ = 0.0, 0.0, 0.0

    print(f'\n{model_name}:')
    print(f'  Stage-1: Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}')
    print(f'  Stage-2: Precision={p2_:.4f}  Recall={r2_:.4f}  F1={f2_:.4f}')
