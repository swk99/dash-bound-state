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
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine
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

    rds.delete(f"dash:state:{symbol}")
    engine = _mk_engine(model_name)

    # warmup
    for t in range(WARMUP):
        engine.process_tick(symbol, data[t], None)

    lat = []
    miss = 0

    # NOTE:
    # This benchmark is "max-burst" (no sleep) by default.
    # burst_factor here is used to report a stricter notion of "deadline".
    # (If you want real-time pacing, add sleep based on 1s/burst_factor.)
    for t in range(N_EVAL):
        idx = WARMUP + t

        t0 = time.perf_counter_ns()
        engine.process_tick(symbol, data[idx], None)
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
    return run_memory_vs_nw(model_name, data, n_list=n_list, symbol=symbol)


def run_memory_vs_nw(model_name, data, *, n_list=(1000, 5000, 10000, 20000),
                     w_list=(10, 30, 60, 120), repeats=5, symbol="sym_mem"):
    """E2: paired A/B1 footprint, N x W x repeat. Never clears unrelated keys."""
    import uuid
    if repeats < 1 or not n_list or not w_list or min(n_list) < 1 or min(w_list) < 1:
        raise ValueError("positive N, W and repeats required")
    if max(n_list) > len(data):
        raise ValueError("not enough input rows")
    rows = []
    run = uuid.uuid4().hex
    for W in w_list:
        for N in n_list:
            for repeat in range(repeats):
                for cls in (ProposedStatefulEngine, RedisFetchBaselineEngine):
                    engine = _mk_engine(model_name)
                    engine.wrapper.lookback_w = int(W)
                    engine = cls(engine.wrapper, rds)
                    sym = f"{symbol}:{run}:{cls.__name__}:{repeat}"
                    key = f"dash:state:{sym}"
                    try:
                        for t in range(N):
                            engine.process_tick(sym, data[t])
                        size = rds.memory_usage(key, samples=0)
                        if size is None:
                            raise RuntimeError("MEMORY USAGE returned no state")
                        rows.append(dict(model=model_name, engine=cls.__name__, N=N, W=W,
                            repeat=repeat, d=len(cfg.FEATURE_COLS), redis_key=key,
                            redis_list_len=rds.llen(key), redis_mem_bytes=int(size),
                            rss_mb=psutil.Process().memory_info().rss / 2**20))
                    finally:
                        rds.delete(key)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"memory_vs_nw_{model_name}.csv", index=False)
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

    import uuid
    token = uuid.uuid4().hex
    symbols = [f"{symbol}:{token}:gate", f"{symbol}:{token}:force"]
    engines = [_mk_engine(model_name), _mk_engine(model_name)]
    lat_gate, lat_force, raw_rows = [], [], []
    fired = 0
    rng = np.random.default_rng(20260928)
    try:
        for t in range(WARMUP):
            for eng, sym in zip(engines, symbols):
                eng.process_tick(sym, data[t])
        for t in range(N_EVAL):
            idx = WARMUP + t
            for which in rng.permutation(2):
                result = engines[which].process_tick(symbols[which], data[idx],
                    force_stage2=bool(which))
                (lat_force if which else lat_gate).append(result['total_ms'])
                if not which:
                    fired += int(result['stage2_fired'])
                raw_rows.append(dict(result, idx=idx, policy='force' if which else 'gate',
                    model=model_name, tau=engines[which].wrapper.tau_conf, seed=20260928))
    finally:
        rds.delete(*[f"dash:state:{sym}" for sym in symbols])
    pd.DataFrame(raw_rows).to_csv(RESULTS_DIR / f"gate_ticks_{model_name}.csv", index=False)

    g = np.asarray(lat_gate, dtype=np.float64)
    f = np.asarray(lat_force, dtype=np.float64)

    out = {
        "model": model_name,
        "gate_firing_rate": fired / N_EVAL,
        "tau": engines[0].wrapper.tau_conf,
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