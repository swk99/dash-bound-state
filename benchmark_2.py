# benchmark_2.py
# Main-track system benchmarks:
# 1) Burst stress test
# 2) Memory vs stream length N
# 3) Gate ablation (force_stage2)

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import psutil
import redis

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine
from tooling import MinioHandler

# -------------------------
# Settings (you can tune)
# -------------------------
WARMUP = 200
N_EVAL = 2000

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Redis connection
rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, decode_responses=False)


def _load_data(needed_rows: int) -> np.ndarray:
    """Load historical features from MinIO/S3 flush logs (same as your pipeline)."""
    handler = MinioHandler()
    data = handler.load_historical_features(
        symbol=cfg.SYMBOL,
        n_rows=needed_rows,
        allow_dummy=False,
    )
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=np.float32)

    d = len(cfg.FEATURE_COLS)
    if data.ndim != 2 or data.shape[1] < d:
        raise ValueError(f"Loaded data shape {data.shape}, expected (N, >= {d})")
    if data.shape[1] != d:
        # keep only canonical dims
        data = data[:, :d]
    if data.shape[0] < needed_rows:
        raise RuntimeError(f"Need {needed_rows} rows but got {data.shape[0]}")
    return data


def _mk_engine(model_name: str) -> ProposedStatefulEngine:
    harness = load_dash_harness(
        model_name=model_name,
        lambda_val=cfg.LAMBDA,
        alpha=cfg.ALPHA,
        h=cfg.HORIZON_H,
        tau_conf=cfg.TAU_CONF,
        lookback_w=cfg.LOOKBACK_W,
    )
    return ProposedStatefulEngine(harness, rds)


def _redis_mem_usage_bytes(key: str) -> Optional[int]:
    """Best-effort Redis MEMORY USAGE for a key."""
    try:
        v = rds.execute_command("MEMORY", "USAGE", key)
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


# ============================================================
# 1) Burst Stress Test
# ============================================================
def run_burst_stress(
    model_name: str,
    data: np.ndarray,
    *,
    burst_factor: int = 5,
    deadline_ms: float = 10.0,
    symbol: str = "sym_burst",
) -> Dict[str, Any]:
    print("\n==============================", flush=True)
    print(f"[BURST TEST] {model_name}  x{burst_factor}", flush=True)
    print("==============================", flush=True)

    rds.flushall()
    engine = _mk_engine(model_name)

    # warmup
    for t in range(WARMUP):
        engine.process_tick(symbol, data[t], cfg.TAU_CONF)

    lat = []
    miss = 0

    # NOTE:
    # This benchmark is "max-burst" (no sleep) by default.
    # burst_factor here is used to report a stricter notion of "deadline".
    # (If you want real-time pacing, add sleep based on 1s/burst_factor.)
    for t in range(N_EVAL):
        idx = WARMUP + t

        t0 = time.perf_counter_ns()
        engine.process_tick(symbol, data[idx], cfg.TAU_CONF)
        t1 = time.perf_counter_ns()

        ms = (t1 - t0) / 1e6
        lat.append(ms)
        if ms > deadline_ms:
            miss += 1

    lat = np.asarray(lat, dtype=np.float64)
    out = {
        "model": model_name,
        "burst_factor": int(burst_factor),
        "deadline_ms": float(deadline_ms),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "miss_rate": float(miss / len(lat)),
        "n": int(len(lat)),
    }

    print(f"P99 latency: {out['p99_ms']:.3f} ms", flush=True)
    print(f"Deadline miss rate: {out['miss_rate']:.4f}  (deadline={deadline_ms}ms)", flush=True)

    pd.DataFrame([out]).to_csv(RESULTS_DIR / f"burst_{model_name}.csv", index=False)
    return out


# ============================================================
# 2) Memory vs N Scaling
# ============================================================
def run_memory_vs_n(
    model_name: str,
    data: np.ndarray,
    *,
    n_list: List[int] = [1000, 5000, 10000, 20000],
    symbol: str = "sym_mem",
) -> pd.DataFrame:
    print("\n==============================", flush=True)
    print(f"[MEMORY VS N] {model_name}", flush=True)
    print("==============================", flush=True)

    process = psutil.Process()
    rows = []

    for N in n_list:
        rds.flushall()
        engine = _mk_engine(model_name)

        # warmup not strictly needed; keep it simple
        for t in range(N):
            engine.process_tick(symbol, data[t], cfg.TAU_CONF)

        rss_mb = process.memory_info().rss / (1024 * 1024)
        redis_key = f"dash:state:{symbol}"
        redis_len = int(rds.llen(redis_key))
        redis_bytes = _redis_mem_usage_bytes(redis_key)

        row = {
            "model": model_name,
            "N": int(N),
            "rss_mb": float(rss_mb),
            "redis_list_len": int(redis_len),   # should be ~W
            "redis_mem_bytes": None if redis_bytes is None else int(redis_bytes),
            "W": int(cfg.LOOKBACK_W),
        }
        rows.append(row)

        if redis_bytes is None:
            print(f"N={N} | RSS={rss_mb:.2f} MB | Redis window={redis_len}", flush=True)
        else:
            print(f"N={N} | RSS={rss_mb:.2f} MB | Redis window={redis_len} | RedisMem={redis_bytes} bytes", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"memory_vs_n_{model_name}.csv", index=False)
    return df


# ============================================================
# 3) Gate Ablation Test
# ============================================================
def run_gate_ablation(
    model_name: str,
    data: np.ndarray,
    *,
    symbol: str = "sym_gate",
) -> Dict[str, Any]:
    print("\n==============================", flush=True)
    print(f"[GATE ABLATION] {model_name}", flush=True)
    print("==============================", flush=True)

    rds.flushall()
    engine = _mk_engine(model_name)

    lat_gate = []
    lat_force = []

    # warmup
    for t in range(WARMUP):
        engine.process_tick(symbol, data[t], cfg.TAU_CONF)

    for t in range(N_EVAL):
        idx = WARMUP + t

        # normal gating
        s1 = time.perf_counter_ns()
        engine.process_tick(symbol, data[idx], cfg.TAU_CONF)
        e1 = time.perf_counter_ns()
        lat_gate.append((e1 - s1) / 1e6)

        # force stage2
        s2 = time.perf_counter_ns()
        engine.process_tick(symbol, data[idx], cfg.TAU_CONF, force_stage2=True)
        e2 = time.perf_counter_ns()
        lat_force.append((e2 - s2) / 1e6)

    g = np.asarray(lat_gate, dtype=np.float64)
    f = np.asarray(lat_force, dtype=np.float64)

    out = {
        "model": model_name,
        "p99_gate_ms": float(np.percentile(g, 99)),
        "p99_force_ms": float(np.percentile(f, 99)),
        "ratio_p99": float(np.percentile(f, 99) / max(1e-12, np.percentile(g, 99))),
        "p95_gate_ms": float(np.percentile(g, 95)),
        "p95_force_ms": float(np.percentile(f, 95)),
        "ratio_p95": float(np.percentile(f, 95) / max(1e-12, np.percentile(g, 95))),
        "n": int(len(g)),
    }

    print(f"P99 (normal gate): {out['p99_gate_ms']:.3f} ms", flush=True)
    print(f"P99 (force stage2): {out['p99_force_ms']:.3f} ms", flush=True)
    print(f"P99 ratio (force/gate): {out['ratio_p99']:.2f}x", flush=True)

    pd.DataFrame([out]).to_csv(RESULTS_DIR / f"gate_ablation_{model_name}.csv", index=False)
    return out


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # ensure enough rows for ALL tests
    maxN = 20000
    needed_rows = max(maxN, WARMUP + N_EVAL + 10)
    data = _load_data(needed_rows)

    model = "Logistic"

    run_burst_stress(model, data, burst_factor=5, deadline_ms=10.0)
    run_memory_vs_n(model, data, n_list=[1000, 5000, 10000, 20000])
    run_gate_ablation(model, data)

    print("\n[DONE] Results saved under ./results", flush=True)