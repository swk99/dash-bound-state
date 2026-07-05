import numpy as np
import pandas as pd
from tooling import MinioHandler
import config as cfg

def forward_return(r, H):
    r = r.to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(r)])
    out = np.full(len(r), np.nan)
    idx = np.arange(len(r) - H)
    out[idx] = cs[idx + H + 1] - cs[idx + 1]
    return out

def run_labeling():
    h = MinioHandler()
    df = h.load_historical_features(symbol=cfg.SYMBOL, n_rows=230000, return_df=True)

    # 미래 H초 수익률
    df['future_r'] = forward_return(df['r_t'], cfg.HORIZON_H)

    # adaptive threshold (논문 정의)
    thr = cfg.ALPHA * np.sqrt(cfg.HORIZON_H) * df['sigma_hat']

    df['y'] = 0
    df.loc[df['future_r'] >= thr, 'y'] = 2
    df.loc[df['future_r'] <= -thr, 'y'] = 1

    # 미래 정보 없는 tail만 제거
    df = df[df['future_r'].notna()].copy()

    df.to_parquet(cfg.LABELED_PATH)
    print("Labeling complete (aligned with the thesis definition).")

if __name__ == "__main__":
    run_labeling()