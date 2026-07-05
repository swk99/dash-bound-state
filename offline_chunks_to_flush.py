from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

import config as cfg


# -------------------------
# Same semantics as rt_ingest_trades.py
# -------------------------
@dataclass
class EWMAState:
    lam: float
    sig2: Optional[float] = None

    def update(self, r: float) -> float:
        r2 = float(r) ** 2
        if self.sig2 is None:
            self.sig2 = r2
        else:
            self.sig2 = self.lam * self.sig2 + (1.0 - self.lam) * r2
        return float(np.sqrt(self.sig2 + cfg.EPSILON))


@dataclass
class RollingMeanState:
    win: int
    buf: List[float]

    def __init__(self, win: int):
        self.win = int(win)
        self.buf = []

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        if len(self.buf) > self.win:
            self.buf.pop(0)
        return float(np.mean(self.buf)) if self.buf else float(x)


def build_feature_row(
    sec: pd.Timestamp,
    vwap: float,
    r_t: float,
    sigma_hat: float,
    total_vol: float,
    buy_vol: float,
    sell_vol: float,
    msg_count: int,
    vol_baseline: float,
) -> dict:
    ofi = buy_vol - sell_vol
    imbalance = ofi / (buy_vol + sell_vol + cfg.EPSILON)
    vol_spike = total_vol / (vol_baseline + cfg.EPSILON)

    return {
        "sec": sec,
        "total_vol": float(total_vol),
        "vwap": float(vwap),
        "buy_vol": float(buy_vol),
        "sell_vol": float(sell_vol),
        "msg_count": int(msg_count),
        "r_t": float(r_t),
        "sigma_hat": float(sigma_hat),
        "OFI_t": float(ofi),
        "Imbalance_t": float(imbalance),
        "VolSpike_t": float(vol_spike),
    }


def aggregate_trades_to_1s_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns expected:
      - ts (datetime64[ns, UTC] or parseable)
      - price (float)
      - qty (float)
      - is_buyer_maker (bool)
    Output per-second aggregates:
      sec, sum_pq, sum_q, total_vol, buy_vol, sell_vol, msg_count
    """
    if "ts" not in df.columns:
        raise ValueError("Missing column: ts")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).copy()

    # 1-second bin
    df["sec"] = df["ts"].dt.floor("s")

    # side: m=True -> buyer is maker -> aggressive sell
    # so buy = m==False, sell = m==True
    df["buy_qty"] = np.where(df["is_buyer_maker"].astype(bool), 0.0, df["qty"].astype(float))
    df["sell_qty"] = np.where(df["is_buyer_maker"].astype(bool), df["qty"].astype(float), 0.0)

    df["pq"] = df["price"].astype(float) * df["qty"].astype(float)

    g = df.groupby("sec", sort=True, observed=True)
    bars = g.agg(
        sum_pq=("pq", "sum"),
        sum_q=("qty", "sum"),
        total_vol=("qty", "sum"),
        buy_vol=("buy_qty", "sum"),
        sell_vol=("sell_qty", "sum"),
        msg_count=("qty", "size"),
    ).reset_index()

    return bars


def flush_parquet(rows: List[dict], out_dir: Path, symbol: str, flush_idx: int) -> None:
    df = pd.DataFrame(rows)
    df["sec"] = pd.to_datetime(df["sec"], utc=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / symbol / f"flush_{flush_idx}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    print(f"[OK] wrote {len(df)} rows -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=str, required=True,
                    help="Directory containing processed chunk_*.parquet")
    ap.add_argument("--symbol", type=str, default=cfg.SYMBOL)
    ap.add_argument("--out-dir", type=str, default="offline_df1s_flush",
                    help="Output directory root (will create <out>/<symbol>/flush_*.parquet)")
    ap.add_argument("--flush-rows", type=int, default=1000)
    ap.add_argument("--vol-spike-win", type=int, default=60)
    ap.add_argument("--lambda", dest="lam", type=float, default=cfg.LAMBDA)
    ap.add_argument("--reset-per-chunk", action="store_true",
                    help="If set, resets last_vwap/EWMA/baseline at each chunk boundary.")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        raise FileNotFoundError(f"chunks-dir not found: {chunks_dir}")

    symbol = args.symbol.lower()
    out_root = Path(args.out_dir)
    out_sym_dir = out_root  # flush_parquet will append /symbol

    # Stateful feature calc across chunks (default: continue across time)
    last_vwap: Optional[float] = None
    ewma = EWMAState(lam=args.lam)
    vol_base = RollingMeanState(win=args.vol_spike_win)

    flush_buf: List[dict] = []
    flush_idx = 0

    files = sorted(chunks_dir.glob("chunk_*.parquet"))
    if not files:
        raise RuntimeError(f"No chunk_*.parquet found under: {chunks_dir}")

    print(f"[*] Found {len(files)} chunk files under {chunks_dir}")
    print(f"[*] Output: {out_root / symbol}")
    print(f"[*] flush_rows={args.flush_rows} | vol_spike_win={args.vol_spike_win} | lambda={args.lam}")

    for fp in files:
        print(f"[*] Reading chunk: {fp.name}")
        df = pd.read_parquet(fp)

        needed = {"ts", "price", "qty", "is_buyer_maker"}
        miss = needed - set(df.columns)
        if miss:
            raise ValueError(f"{fp.name} missing columns: {sorted(miss)}")

        # Ensure time order inside chunk
        df = df.sort_values("ts").reset_index(drop=True)

        bars = aggregate_trades_to_1s_bars(df)
        if bars.empty:
            continue

        # Optionally reset states at chunk boundary (useful if big gaps)
        if args.reset_per_chunk:
            last_vwap = None
            ewma = EWMAState(lam=args.lam)
            vol_base = RollingMeanState(win=args.vol_spike_win)

        # Convert bars -> feature rows sequentially (needs last_vwap, EWMA, rolling baseline)
        for _, r in bars.iterrows():
            sec = r["sec"]
            sum_q = float(r["sum_q"])
            sum_pq = float(r["sum_pq"])
            total_vol = float(r["total_vol"])
            buy_vol = float(r["buy_vol"])
            sell_vol = float(r["sell_vol"])
            msg_count = int(r["msg_count"])

            if sum_q > 0:
                vwap = sum_pq / sum_q
            else:
                # should be rare; fallback
                vwap = last_vwap if last_vwap is not None else 0.0

            r_t = 0.0 if last_vwap is None else float(np.log(vwap / (last_vwap + cfg.EPSILON)))
            sigma_hat = ewma.update(r_t)
            baseline = vol_base.update(total_vol)

            row = build_feature_row(
                sec=sec,
                vwap=vwap,
                r_t=r_t,
                sigma_hat=sigma_hat,
                total_vol=total_vol,
                buy_vol=buy_vol,
                sell_vol=sell_vol,
                msg_count=msg_count,
                vol_baseline=baseline,
            )
            flush_buf.append(row)
            last_vwap = vwap

            if len(flush_buf) >= int(args.flush_rows):
                flush_idx += 1
                flush_parquet(flush_buf, out_sym_dir, symbol, flush_idx)
                flush_buf = []

    # final tail
    if flush_buf:
        flush_idx += 1
        flush_parquet(flush_buf, out_sym_dir, symbol, flush_idx)

    print("[DONE]")
    print(f"  - total flush files: {flush_idx}")
    print(f"  - feature cols downstream: {cfg.FEATURE_COLS}")


if __name__ == "__main__":
    main()