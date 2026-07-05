# benchmark_3.py
# Benchmark 3: Impact of Network Jitter and Remote State Access
# - Injects artificial delay/jitter with `tc netem` on a selected interface
# - Measures tail latency (P50/P95/P99) for A (Proposed) vs B1 (RedisFetch)
# - Saves raw per-tick latencies + summary CSV under ./results

from __future__ import annotations

import argparse
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import redis

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine
from tooling import MinioHandler


# -------------------------
# Paths
# -------------------------
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


# -------------------------
# Redis
# -------------------------
rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT, decode_responses=False)


# -------------------------
# Helpers
# -------------------------
def _normalize_result(res: Dict[str, Any], *, engine_name: str) -> Dict[str, Any]:
    out = dict(res)

    if "A_ms" in out:
        A_ms = float(out.get("A_ms", 0.0))
    else:
        A_ms = float(out.get("A_update_ms", 0.0)) + float(out.get("A_fetch_ms", 0.0))
    out["A_ms"] = A_ms

    out["B1_ms"] = float(out.get("B1_ms", 0.0))
    out["B2_ms"] = float(out.get("B2_ms", 0.0))
    out["total_ms"] = float(out.get("total_ms", out["A_ms"] + out["B1_ms"] + out["B2_ms"]))

    out["engine"] = engine_name
    return out


def _load_data(needed_rows: int) -> np.ndarray:
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
        data = data[:, :d]
    if data.shape[0] < needed_rows:
        raise RuntimeError(f"Need {needed_rows} rows but got {data.shape[0]}")
    return data


def _run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)


def tc_clear(iface: str) -> None:
    # best-effort: ignore errors if qdisc doesn't exist
    try:
        _run(["tc", "qdisc", "del", "dev", iface, "root"], check=False)
    except Exception:
        pass


def tc_set_netem(iface: str, delay_ms: float, jitter_ms: float) -> None:
    """
    Apply netem delay/jitter:
      tc qdisc add dev <iface> root netem delay <delay>ms <jitter>ms distribution normal
    """
    tc_clear(iface)
    # Use "distribution normal" to approximate jitter
    cmd = [
        "tc", "qdisc", "add", "dev", iface, "root", "netem",
        "delay", f"{delay_ms}ms", f"{jitter_ms}ms",
        "distribution", "normal"
    ]
    p = _run(cmd, check=True)
    if p.stderr.strip():
        # tc sometimes prints warnings to stderr; not necessarily fatal
        print(f"[tc] {p.stderr.strip()}", flush=True)


def require_root_for_tc() -> bool:
    # On Linux, tc qdisc modifications usually require root
    return (os.geteuid() == 0)


def summarize_latency(x_ms: np.ndarray) -> Dict[str, float]:
    return {
        "p50_ms": float(np.percentile(x_ms, 50)),
        "p95_ms": float(np.percentile(x_ms, 95)),
        "p99_ms": float(np.percentile(x_ms, 99)),
        "mean_ms": float(np.mean(x_ms)),
    }


# -------------------------
# Benchmark core
# -------------------------
def run_network_jitter_benchmark(
    *,
    model_name: str,
    k: int,
    iface: str,
    n_warmup: int,
    n_eval: int,
    offset_stride: int,
    seed: int,
    # scenarios: list of (name, delay_ms, jitter_ms)
    scenarios: List[Tuple[str, float, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs scenarios; for each scenario measures A and B1 total_ms distributions.
    Saves raw rows + summary rows.
    """
    rng = np.random.default_rng(seed)

    # compute needed rows (similar safety logic as benchmark.py)
    max_offset = (k - 1) * offset_stride
    needed_rows = max_offset + n_warmup + (n_eval // max(1, k)) + 64
    needed_rows = max(needed_rows, n_warmup + n_eval + 256)
    needed_rows = max(needed_rows, 3000)

    data = _load_data(needed_rows)

    alpha = float(cfg.ALPHA)
    W = int(cfg.LOOKBACK_W)
    tau = float(cfg.TAU_CONF)

    tag = cfg.make_tag(h=int(cfg.HORIZON_H), alpha=alpha, lambda_val=float(cfg.LAMBDA))
    run_id = f"NET_{model_name}_{tag}_W{W}_tau{tau:g}_K{k}"

    print("\n==============================================", flush=True)
    print(f"[BENCHMARK 3] Network jitter / remote Redis access", flush=True)
    print("==============================================", flush=True)
    print(f"run_id: {run_id}", flush=True)
    print(f"redis : {cfg.REDIS_HOST}:{cfg.REDIS_PORT}", flush=True)
    print(f"iface : {iface}", flush=True)
    print(f"K     : {k}", flush=True)
    print(f"warmup: {n_warmup} | eval: {n_eval}", flush=True)
    print("----------------------------------------------", flush=True)

    harness = load_dash_harness(
        model_name=model_name,
        lambda_val=float(cfg.LAMBDA),
        alpha=alpha,
        h=int(cfg.HORIZON_H),
        tau_conf=tau,
        lookback_w=W,
    )

    engine_A = ProposedStatefulEngine(harness, rds)
    engine_B1 = RedisFetchBaselineEngine(harness, rds)

    symbols = [f"sym_{run_id}_{i}" for i in range(k)]
    offsets = {s: i * offset_stride for i, s in enumerate(symbols)}

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    try:
        for scen_name, delay_ms, jitter_ms in scenarios:
            print(f"\n--- Scenario: {scen_name} (delay={delay_ms}ms, jitter={jitter_ms}ms) ---", flush=True)

            # Apply tc for non-baseline scenarios
            if scen_name.lower() != "baseline":
                if not require_root_for_tc():
                    print("[WARN] Not running as root. Cannot apply tc netem. "
                          "Run with: sudo python benchmark_3.py ...", flush=True)
                    print("[WARN] Proceeding WITHOUT tc (results will resemble baseline).", flush=True)
                else:
                    tc_set_netem(iface, delay_ms=delay_ms, jitter_ms=jitter_ms)
                    print("[INFO] tc netem applied.", flush=True)
            else:
                # baseline: ensure clean qdisc
                if require_root_for_tc():
                    tc_clear(iface)

            # Reset redis state
            rds.flushall()

            # -------------------------
            # Warmup
            # -------------------------
            for t in range(n_warmup):
                sym = symbols[t % k]
                idx = offsets[sym] + t
                feat = data[idx]
                engine_A.process_tick(sym, feat, tau)
                engine_B1.process_tick(sym, feat, tau)

            # -------------------------
            # Measure A
            # -------------------------
            lat_A = []
            for t in range(n_eval):
                sym = symbols[t % k]
                idx = offsets[sym] + n_warmup + (t // k)
                feat = data[idx]
                res = engine_A.process_tick(sym, feat, tau)
                res = _normalize_result(res, engine_name="A_Proposed")
                lat_A.append(res["total_ms"])

                res.update({
                    "run_id": run_id,
                    "scenario": scen_name,
                    "delay_ms": float(delay_ms),
                    "jitter_ms": float(jitter_ms),
                    "model": model_name,
                    "tag": tag,
                    "alpha": alpha,
                    "lookback_w": W,
                    "tau": tau,
                    "k": int(k),
                    "trial": int(t),
                    "idx": int(idx),
                })
                raw_rows.append(res)

            lat_A = np.asarray(lat_A, dtype=np.float64)
            sA = summarize_latency(lat_A)
            print(f"[A] P99={sA['p99_ms']:.3f} ms | P95={sA['p95_ms']:.3f} | P50={sA['p50_ms']:.3f}", flush=True)

            # -------------------------
            # Measure B1
            # -------------------------
            lat_B1 = []
            for t in range(n_eval):
                sym = symbols[t % k]
                idx = offsets[sym] + n_warmup + (t // k)
                feat = data[idx]
                res = engine_B1.process_tick(sym, feat, tau)
                res = _normalize_result(res, engine_name="B1_RedisFetch")
                lat_B1.append(res["total_ms"])

                res.update({
                    "run_id": run_id,
                    "scenario": scen_name,
                    "delay_ms": float(delay_ms),
                    "jitter_ms": float(jitter_ms),
                    "model": model_name,
                    "tag": tag,
                    "alpha": alpha,
                    "lookback_w": W,
                    "tau": tau,
                    "k": int(k),
                    "trial": int(t),
                    "idx": int(idx),
                })
                raw_rows.append(res)

            lat_B1 = np.asarray(lat_B1, dtype=np.float64)
            sB1 = summarize_latency(lat_B1)
            print(f"[B1] P99={sB1['p99_ms']:.3f} ms | P95={sB1['p95_ms']:.3f} | P50={sB1['p50_ms']:.3f}", flush=True)

            # Ratio (tail)
            ratio_p99 = float(sB1["p99_ms"] / max(1e-12, sA["p99_ms"]))
            ratio_p95 = float(sB1["p95_ms"] / max(1e-12, sA["p95_ms"]))
            ratio_p50 = float(sB1["p50_ms"] / max(1e-12, sA["p50_ms"]))

            summary_rows.append({
                "run_id": run_id,
                "scenario": scen_name,
                "delay_ms": float(delay_ms),
                "jitter_ms": float(jitter_ms),
                "iface": iface,
                "redis_host": cfg.REDIS_HOST,
                "redis_port": int(cfg.REDIS_PORT),

                "model": model_name,
                "tag": tag,
                "alpha": alpha,
                "lookback_w": W,
                "tau": tau,
                "k": int(k),
                "n_warmup": int(n_warmup),
                "n_eval": int(n_eval),

                "A_p50_ms": sA["p50_ms"],
                "A_p95_ms": sA["p95_ms"],
                "A_p99_ms": sA["p99_ms"],
                "A_mean_ms": sA["mean_ms"],

                "B1_p50_ms": sB1["p50_ms"],
                "B1_p95_ms": sB1["p95_ms"],
                "B1_p99_ms": sB1["p99_ms"],
                "B1_mean_ms": sB1["mean_ms"],

                "ratio_p50": ratio_p50,
                "ratio_p95": ratio_p95,
                "ratio_p99": ratio_p99,
            })

            print(f"[RATIO] B1/A  P99={ratio_p99:.2f}x | P95={ratio_p95:.2f}x | P50={ratio_p50:.2f}x", flush=True)

    finally:
        # Always attempt to clean qdisc so you don't “brick” your loopback
        if require_root_for_tc():
            tc_clear(iface)

    df_raw = pd.DataFrame(raw_rows)
    df_sum = pd.DataFrame(summary_rows)
    return df_raw, df_sum


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="XGBoost", choices=["XGBoost", "RandomForest", "Logistic"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--iface", type=str, default="lo", help="Interface to apply tc netem (lo for localhost Redis).")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--offset-stride", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # scenario knobs
    ap.add_argument("--intra-delay", type=float, default=1.0)
    ap.add_argument("--intra-jitter", type=float, default=0.1)
    ap.add_argument("--cross-delay", type=float, default=10.0)
    ap.add_argument("--cross-jitter", type=float, default=1.0)

    args = ap.parse_args()

    scenarios = [
        ("Baseline", 0.0, 0.0),
        ("IntraCluster", float(args.intra_delay), float(args.intra_jitter)),
        ("CrossRegion", float(args.cross_delay), float(args.cross_jitter)),
    ]

    df_raw, df_sum = run_network_jitter_benchmark(
        model_name=args.model,
        k=int(args.k),
        iface=str(args.iface),
        n_warmup=int(args.warmup),
        n_eval=int(args.eval),
        offset_stride=int(args.offset_stride),
        seed=int(args.seed),
        scenarios=scenarios,
    )

    raw_path = RESULTS_DIR / f"network_jitter_raw_{args.model}_K{int(args.k)}.csv"
    sum_path = RESULTS_DIR / f"network_jitter_summary_{args.model}_K{int(args.k)}.csv"

    df_raw.to_csv(raw_path, index=False)
    df_sum.to_csv(sum_path, index=False)

    print("\n[V] Benchmark 3 complete.", flush=True)
    print(f"    - raw    : {raw_path}", flush=True)
    print(f"    - summary: {sum_path}", flush=True)