# rt_ingest_trades.py
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import redis
import boto3
import websockets
import argparse

import config as cfg


@dataclass
class BarState:
    sec_ts: Optional[pd.Timestamp] = None  # current second (UTC, floored)
    sum_pq: float = 0.0
    sum_q: float = 0.0
    total_vol: float = 0.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    msg_count: int = 0


def ensure_bucket(s3):
    buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if cfg.BUCKET_NAME not in buckets:
        s3.create_bucket(Bucket=cfg.BUCKET_NAME)


def make_s3_client():
    s3 = boto3.client(
        "s3",
        endpoint_url=cfg.MINIO_ENDPOINT,
        aws_access_key_id=cfg.MINIO_ACCESS_KEY,
        aws_secret_access_key=cfg.MINIO_SECRET_KEY,
    )
    ensure_bucket(s3)
    return s3


def side_from_is_buyer_maker(is_buyer_maker: bool) -> int:
    # Binance: m=True -> buyer is maker -> usually aggressive sell
    return -1 if is_buyer_maker else +1


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
        "total_vol": total_vol,
        "vwap": vwap,
        "buy_vol": buy_vol,
        "sell_vol": sell_vol,
        "msg_count": msg_count,
        "r_t": r_t,
        "sigma_hat": sigma_hat,
        "OFI_t": ofi,
        "Imbalance_t": imbalance,
        "VolSpike_t": vol_spike,
    }


class RollingVolBaseline:
    """simple rolling mean baseline for volume spike"""
    def __init__(self, win: int):
        self.win = int(win)
        self.buf: list[float] = []

    def update(self, vol: float) -> float:
        self.buf.append(float(vol))
        if len(self.buf) > self.win:
            self.buf.pop(0)
        return float(np.mean(self.buf)) if self.buf else float(vol)


class EWMA:
    def __init__(self, lam: float = 0.94):
        self.lam = float(lam)
        self.sig2: Optional[float] = None

    def update(self, r: float) -> float:
        r2 = float(r) ** 2
        if self.sig2 is None:
            self.sig2 = r2
        else:
            self.sig2 = self.lam * self.sig2 + (1 - self.lam) * r2
        return float(np.sqrt(self.sig2 + cfg.EPSILON))


def redis_push_feature(rds: redis.Redis, key: str, x: np.ndarray, lookback_w: int):
    rds.lpush(key, x.astype(np.float32).tobytes())
    rds.ltrim(key, 0, int(lookback_w) - 1)


def flush_to_s3_parquet(s3, rows: list[dict], flush_idx: int, symbol: str):
    df = pd.DataFrame(rows)
    df["sec"] = pd.to_datetime(df["sec"], utc=True)

    tmp = Path(f"flush_{flush_idx}.parquet")
    df.to_parquet(tmp, index=False)

    key = f"{cfg.S3_DATA_PREFIX}{symbol}/flush_{flush_idx}.parquet"
    s3.upload_file(str(tmp), cfg.BUCKET_NAME, key)
    tmp.unlink(missing_ok=True)
    print(f"[COLD] Flushed {len(df)} rows -> s3://{cfg.BUCKET_NAME}/{key}")


async def run(symbol: str, flush_every_rows: int, vol_spike_win: int):
    symbol = symbol.lower()
    stream = f"{symbol}@trade"
    ws_url = f"wss://stream.binance.com:443/ws/{stream}"

    rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)
    s3 = make_s3_client()

    state = BarState()
    ewma = EWMA(cfg.LAMBDA)
    vol_base = RollingVolBaseline(vol_spike_win)

    last_vwap: Optional[float] = None
    flush_rows: list[dict] = []
    flush_idx = 0

    # --- heartbeat / liveness ---
    hb_every = 5.0  # seconds (no spam)
    last_hb = time.time()
    last_rx = time.time()
    total_trades = 0

    # Hot feature key (per symbol)
    feat_key = f"feat_buffer:{symbol}"
    rds.delete(feat_key)

    reconnect_backoff = [1, 2, 5, 10, 20, 30, 60]
    attempt = 0

    while True:
        try:
            print("[WS] connecting:", ws_url)
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                attempt = 0
                print("[WS] connected.")
                async for msg in ws:
                    data = json.loads(msg)

                    # mark liveness on every received message
                    last_rx = time.time()
                    total_trades += 1

                    # heartbeat: print once every hb_every seconds
                    now = time.time()
                    if now - last_hb >= hb_every:
                        try:
                            llen = rds.llen(feat_key)
                        except Exception:
                            llen = -1

                        print(
                            f"[HB] alive | sym={symbol} | trades={total_trades:,} | "
                            f"cur_sec={state.sec_ts} | bar_msgs={state.msg_count} | "
                            f"flush_buf={len(flush_rows)} | redis_llen={llen}"
                        )

                        # detect "connected but no data" stalls (should almost never happen)
                        if now - last_rx > 20:
                            print(f"[WARN] no trade messages for {now - last_rx:.1f}s (network stall?)")

                        last_hb = now

                    ts_ms = int(data["T"])
                    price = float(data["p"])
                    qty = float(data["q"])
                    is_buyer_maker = bool(data["m"])
                    side = side_from_is_buyer_maker(is_buyer_maker)

                    ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
                    sec = ts.floor("S")

                    if state.sec_ts is None:
                        state.sec_ts = sec

                    # new second -> finalize previous bar
                    if sec != state.sec_ts:
                        vwap = (state.sum_pq / state.sum_q) if state.sum_q > 0 else (
                            last_vwap if last_vwap is not None else price
                        )

                        r_t = 0.0 if last_vwap is None else float(np.log(vwap / (last_vwap + cfg.EPSILON)))
                        sigma_hat = ewma.update(r_t)
                        baseline = vol_base.update(state.total_vol)

                        row = build_feature_row(
                            sec=state.sec_ts,
                            vwap=vwap,
                            r_t=r_t,
                            sigma_hat=sigma_hat,
                            total_vol=state.total_vol,
                            buy_vol=state.buy_vol,
                            sell_vol=state.sell_vol,
                            msg_count=state.msg_count,
                            vol_baseline=baseline,
                        )

                        # ✅ Redis hot buffer uses cfg.FEATURE_COLS order (must match training)
                        x = np.array([row[c] for c in cfg.FEATURE_COLS], dtype=np.float32)
                        redis_push_feature(rds, feat_key, x, lookback_w=cfg.LOOKBACK_W)

                        # collect for cold flush (full row with 11 cols)
                        flush_rows.append(row)

                        if len(flush_rows) >= int(flush_every_rows):
                            flush_idx += 1
                            flush_to_s3_parquet(s3, flush_rows, flush_idx, symbol=symbol)
                            flush_rows = []

                        last_vwap = vwap
                        state = BarState(sec_ts=sec)

                    # accumulate current second
                    state.sum_pq += price * qty
                    state.sum_q += qty
                    state.total_vol += qty
                    state.msg_count += 1
                    if side == +1:
                        state.buy_vol += qty
                    else:
                        state.sell_vol += qty

        except Exception as e:
            wait = reconnect_backoff[min(attempt, len(reconnect_backoff) - 1)]
            attempt += 1
            print(f"[WS] error: {type(e).__name__}: {e}  -> reconnect in {wait}s")
            await asyncio.sleep(wait)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", type=str, default=cfg.SYMBOL, help="e.g., btcusdt")
    ap.add_argument("--flush-rows", type=int, default=1000, help="how many 1s bars per parquet flush")
    ap.add_argument("--vol-spike-win", type=int, default=60, help="rolling window (seconds) for volume baseline")
    args = ap.parse_args()

    asyncio.run(run(symbol=args.symbol, flush_every_rows=args.flush_rows, vol_spike_win=args.vol_spike_win))