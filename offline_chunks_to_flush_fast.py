from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import config as cfg


EPSILON = cfg.EPSILON
LAMBDA = cfg.LAMBDA


def aggregate_trades_to_1s_bars(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    df["sec"] = df["ts"].dt.floor("s")
    df["buy_qty"] = np.where(df["is_buyer_maker"].astype(bool), 0.0, df["qty"].astype(float))
    df["sell_qty"] = np.where(df["is_buyer_maker"].astype(bool), df["qty"].astype(float), 0.0)
    df["pq"] = df["price"].astype(float) * df["qty"].astype(float)

    bars = df.groupby("sec", sort=True, observed=True).agg(
        sum_pq=("pq", "sum"),
        sum_q=("qty", "sum"),
        total_vol=("qty", "sum"),
        buy_vol=("buy_qty", "sum"),
        sell_vol=("sell_qty", "sum"),
        msg_count=("qty", "size"),
    ).reset_index()

    return bars


def compute_features_vectorized(
    bars: pd.DataFrame,
    last_vwap: Optional[float],
    ewma_sig2: Optional[float],
    vol_buf: List[float],
    vol_spike_win: int,
    lam: float,
) -> tuple[pd.DataFrame, float, float, List[float]]:
    """
    Vectorized feature computation over a bars DataFrame.
    Returns (features_df, last_vwap, ewma_sig2, vol_buf)
    """
    n = len(bars)
    vwap_arr = np.where(bars["sum_q"].values > 0,
                        bars["sum_pq"].values / bars["sum_q"].values,
                        np.nan)

    # forward fill NaN vwap (rare)
    if last_vwap is not None:
        vwap_arr = np.where(np.isnan(vwap_arr), last_vwap, vwap_arr)
    else:
        # fill forward within array
        for i in range(len(vwap_arr)):
            if np.isnan(vwap_arr[i]):
                vwap_arr[i] = vwap_arr[i-1] if i > 0 else 0.0

    # log returns
    r_t_arr = np.zeros(n, dtype=np.float64)
    prev = last_vwap if last_vwap is not None else None
    for i in range(n):
        if prev is not None and prev > 0:
            r_t_arr[i] = np.log(vwap_arr[i] / (prev + EPSILON))
        else:
            r_t_arr[i] = 0.0
        prev = vwap_arr[i]

    # EWMA sigma vectorized
    sig2_arr = np.zeros(n, dtype=np.float64)
    s2 = ewma_sig2 if ewma_sig2 is not None else None
    for i in range(n):
        r2 = r_t_arr[i] ** 2
        if s2 is None:
            s2 = r2
        else:
            s2 = lam * s2 + (1.0 - lam) * r2
        sig2_arr[i] = s2
    sigma_hat_arr = np.sqrt(sig2_arr + EPSILON)

    # volume baseline (rolling mean)
    total_vol = bars["total_vol"].values.astype(np.float64)
    vol_baseline_arr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        vol_buf.append(float(total_vol[i]))
        if len(vol_buf) > vol_spike_win:
            vol_buf.pop(0)
        vol_baseline_arr[i] = float(np.mean(vol_buf))

    buy_vol = bars["buy_vol"].values.astype(np.float64)
    sell_vol = bars["sell_vol"].values.astype(np.float64)

    ofi = buy_vol - sell_vol
    imbalance = ofi / (buy_vol + sell_vol + EPSILON)
    vol_spike = total_vol / (vol_baseline_arr + EPSILON)

    out = pd.DataFrame({
        "sec": bars["sec"].values,
        "total_vol": total_vol,
        "vwap": vwap_arr,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "msg_count": bars["msg_count"].values.astype(int),
        "r_t": r_t_arr,
        "sigma_hat": sigma_hat_arr,
        "OFI_t": ofi,
        "Imbalance_t": imbalance,
        "VolSpike_t": vol_spike,
    })

    new_last_vwap = float(vwap_arr[-1])
    new_sig2 = float(sig2_arr[-1])

    return out, new_last_vwap, new_sig2, vol_buf


def flush_parquet(df: pd.DataFrame, out_dir: Path, symbol: str, flush_idx: int) -> None:
    df["sec"] = pd.to_datetime(df["sec"], utc=True)
    p = out_dir / symbol / f"flush_{flush_idx}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"[OK] wrote {len(df)} rows -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=str, required=True)
    ap.add_argument("--symbol", type=str, default="btcusdt_2")
    ap.add_argument("--out-dir", type=str, default="offline_df1s_flush")
    ap.add_argument("--flush-rows", type=int, default=1000)
    ap.add_argument("--vol-spike-win", type=int, default=60)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_root = Path(args.out_dir)
    symbol = args.symbol.lower()

    files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not files:
        raise RuntimeError(f"No chunk_*.parquet found under: {chunks_dir}")

    print(f"[*] Found {len(files)} chunk files")
    print(f"[*] Output: {out_root / symbol}")

    last_vwap: Optional[float] = None
    ewma_sig2: Optional[float] = None
    vol_buf: List[float] = []

    flush_buf = []
    flush_idx = 0
    total_rows = 0

    for fi, fp in enumerate(files):
        df = pd.read_parquet(fp)
        df = df.sort_values("ts").reset_index(drop=True)

        bars = aggregate_trades_to_1s_bars(df)
        if bars.empty:
            continue

        feat_df, last_vwap, ewma_sig2, vol_buf = compute_features_vectorized(
            bars, last_vwap, ewma_sig2, vol_buf,
            vol_spike_win=args.vol_spike_win,
            lam=args.lam,
        )

        flush_buf.append(feat_df)
        total_rows += len(feat_df)

        # flush when enough rows accumulated
        while True:
            total_buf = sum(len(x) for x in flush_buf)
            if total_buf < args.flush_rows:
                break
            combined = pd.concat(flush_buf, ignore_index=True)
            flush_idx += 1
            flush_parquet(combined.iloc[:args.flush_rows], out_root, symbol, flush_idx)
            remaining = combined.iloc[args.flush_rows:].copy()
            flush_buf = [remaining] if len(remaining) > 0 else []

        if (fi + 1) % 10 == 0:
            print(f"[*] {fi+1}/{len(files)} chunks done | total 1s bars: {total_rows:,}")

    # flush remaining
    if flush_buf:
        combined = pd.concat(flush_buf, ignore_index=True)
        if len(combined) > 0:
            flush_idx += 1
            flush_parquet(combined, out_root, symbol, flush_idx)

    print(f"\n[DONE] total flush files: {flush_idx} | total 1s bars: {total_rows:,}")
    print(f"feature cols: {cfg.FEATURE_COLS}")


if __name__ == "__main__":
    main()