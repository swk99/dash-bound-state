# benchmark.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import redis

import config as cfg
from models import load_dash_harness
from engine import ProposedStatefulEngine, RedisFetchBaselineEngine, InMemoryRecomputeEngine
from tooling import MinioHandler

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

rds = redis.Redis(host=cfg.REDIS_HOST, port=cfg.REDIS_PORT)


def _normalize_result(
    res: Dict[str, Any],
    *,
    engine_name: str,
) -> Dict[str, Any]:
    """
    Normalize engine outputs into a consistent schema for downstream analysis.

    Standard columns:
      - A_ms:    total state-management / fetch overhead for the engine
      - B1_ms:   stage-1 inference time
      - B2_ms:   stage-2 inference time (0 if not invoked)
      - total_ms
    """
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


def run_k_benchmark(
    model_name: str,
    k_list: Optional[list[int]] = None,

    # measurement params
    n_warmup: int = 200,
    n_measure: int = 2000,
    seed: int = 42,
    offset_stride: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paper-consistent K-scaling benchmark across engines:
      - A: ProposedStatefulEngine
      - B1: RedisFetchBaselineEngine
      - B2: InMemoryRecomputeEngine

    Uses SINGLE source of truth from config.py (no sweeps):
      - ALPHA, LOOKBACK_W, TAU_CONF, HORIZON_H, LAMBDA, K_LIST
    """
    k_list = cfg.K_LIST if k_list is None else k_list

    rng = np.random.default_rng(seed)  # (kept in case you later want randomized symbol order etc.)

    # ---------- compute needed_rows safely ----------
    k_max = max(k_list)
    max_offset = (k_max - 1) * offset_stride
    max_idx_warmup = max_offset + (n_warmup - 1)

    k_min = min(k_list)
    max_div = (n_measure - 1) // max(1, k_min)
    max_idx_measure = max_offset + n_warmup + max_div

    max_idx = max(max_idx_warmup, max_idx_measure)
    needed_rows = (max_idx + 1) + 16
    needed_rows = max(needed_rows, 3000)

    # ---------- load historical features ----------
    handler = MinioHandler()
    real_data = handler.load_historical_features(
        symbol=cfg.SYMBOL,
        n_rows=needed_rows,
        allow_dummy=False,
    )

    if isinstance(real_data, pd.DataFrame):
        real_data = real_data.values
    real_data = np.asarray(real_data, dtype=np.float32)

    if real_data.ndim != 2 or real_data.shape[1] != len(cfg.FEATURE_COLS):
        raise ValueError(
            f"Loaded data has shape {real_data.shape}; expected (N, {len(cfg.FEATURE_COLS)})"
        )
    if real_data.shape[0] < needed_rows:
        raise RuntimeError(
            f"Loaded only {real_data.shape[0]} rows, but benchmark needs {needed_rows} rows"
        )

    # ---------- fixed (no sweep) hyperparams ----------
    alpha = float(cfg.ALPHA)
    W = int(cfg.LOOKBACK_W)
    tau = float(cfg.TAU_CONF)

    tag = cfg.make_tag(h=int(cfg.HORIZON_H), alpha=alpha, lambda_val=float(cfg.LAMBDA))
    run_id = f"{model_name}_{tag}_W{W}_tau{tau:g}"

    print(f"\n[*] Benchmark: {run_id}")

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
    engine_B2 = InMemoryRecomputeEngine(harness, real_data)

    all_records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for k in k_list:
        k = int(k)
        print(f"    - K={k}")

        symbols = [f"sym_{run_id}_{i}" for i in range(k)]
        offsets = {s: i * offset_stride for i, s in enumerate(symbols)}

        # reset redis state for those symbols
        keys = [f"dash:state:{s}" for s in symbols]
        if keys:
            rds.delete(*keys)

        # -------------------------
        # Warm-up
        # -------------------------
        for t in range(n_warmup):
            sym = symbols[t % k]
            idx = offsets[sym] + t
            feat = real_data[idx]

            engine_A.process_tick(sym, feat, tau)
            engine_B1.process_tick(sym, feat, tau)
            engine_B2.process_tick(current_idx=int(idx), tau=tau)

        # -------------------------
        # Measure A
        # -------------------------
        t0 = time.perf_counter()
        A_hits = 0
        for t in range(n_measure):
            sym = symbols[t % k]
            idx = offsets[sym] + n_warmup + (t // k)
            feat = real_data[idx]

            res = engine_A.process_tick(sym, feat, tau)
            res = _normalize_result(res, engine_name="A_Proposed")

            if res["B2_ms"] > 0.0:
                A_hits += 1

            res.update({
                "model": model_name,
                "run_id": run_id,
                "tag": tag,
                "alpha": alpha,
                "lookback_w": W,
                "tau": tau,
                "horizon_h": int(cfg.HORIZON_H),
                "lambda": float(cfg.LAMBDA),
                "k": k,
                "trial": int(t),
                "idx": int(idx),
            })
            all_records.append(res)
        t1 = time.perf_counter()
        A_ops = n_measure / max(1e-12, (t1 - t0))

        # -------------------------
        # Measure B1
        # -------------------------
        t0 = time.perf_counter()
        B1_hits = 0
        for t in range(n_measure):
            sym = symbols[t % k]
            idx = offsets[sym] + n_warmup + (t // k)
            feat = real_data[idx]

            res = engine_B1.process_tick(sym, feat, tau)
            res = _normalize_result(res, engine_name="B1_RedisFetch")

            if res["B2_ms"] > 0.0:
                B1_hits += 1

            res.update({
                "model": model_name,
                "run_id": run_id,
                "tag": tag,
                "alpha": alpha,
                "lookback_w": W,
                "tau": tau,
                "horizon_h": int(cfg.HORIZON_H),
                "lambda": float(cfg.LAMBDA),
                "k": k,
                "trial": int(t),
                "idx": int(idx),
            })
            all_records.append(res)
        t1 = time.perf_counter()
        B1_ops = n_measure / max(1e-12, (t1 - t0))

        # -------------------------
        # Measure B2
        # -------------------------
        t0 = time.perf_counter()
        B2_hits = 0
        for t in range(n_measure):
            idx = offsets[symbols[t % k]] + n_warmup + (t // k)

            res = engine_B2.process_tick(current_idx=int(idx), tau=tau)
            res = _normalize_result(res, engine_name="B2_InMemory")

            if res["B2_ms"] > 0.0:
                B2_hits += 1

            res.update({
                "model": model_name,
                "run_id": run_id,
                "tag": tag,
                "alpha": alpha,
                "lookback_w": W,
                "tau": tau,
                "horizon_h": int(cfg.HORIZON_H),
                "lambda": float(cfg.LAMBDA),
                "k": k,
                "trial": int(t),
                "idx": int(idx),
            })
            all_records.append(res)
        t1 = time.perf_counter()
        B2_ops = n_measure / max(1e-12, (t1 - t0))

        summary_rows.append({
            "model": model_name,
            "run_id": run_id,
            "tag": tag,
            "alpha": alpha,
            "lookback_w": W,
            "tau": tau,
            "horizon_h": int(cfg.HORIZON_H),
            "lambda": float(cfg.LAMBDA),
            "k": k,
            "n_warmup": int(n_warmup),
            "n_measure": int(n_measure),
            "offset_stride": int(offset_stride),

            "A_ops_per_sec": float(A_ops),
            "B1_ops_per_sec": float(B1_ops),
            "B2_ops_per_sec": float(B2_ops),

            "A_gate_hit_rate": float(A_hits / n_measure),
            "B1_gate_hit_rate": float(B1_hits / n_measure),
            "B2_gate_hit_rate": float(B2_hits / n_measure),
        })

    df = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(summary_rows)
    return df, df_summary


if __name__ == "__main__":
    model_list = ["XGBoost", "RandomForest", "Logistic"]

    all_lat = []
    all_thr = []

    for model in model_list:
        df_results, df_summary = run_k_benchmark(
            model_name=model,
            k_list=cfg.K_LIST,
            n_warmup=200,
            n_measure=2000,
            seed=42,
            offset_stride=50,
        )

        all_lat.append(df_results)
        all_thr.append(df_summary)

        df_results.to_csv(RESULTS_DIR / f"latency_k_{model}.csv", index=False)
        df_summary.to_csv(RESULTS_DIR / f"throughput_k_{model}.csv", index=False)

    df_results_all = pd.concat(all_lat, ignore_index=True)
    df_summary_all = pd.concat(all_thr, ignore_index=True)

    df_results_all.to_csv(RESULTS_DIR / "latency_k_sweep_FULL.csv", index=False)
    df_summary_all.to_csv(RESULTS_DIR / "throughput_k_sweep_FULL.csv", index=False)

    print("[V] All models benchmark complete.")
    print(f"    - latency:   {RESULTS_DIR / 'latency_k_sweep_FULL.csv'}")
    print(f"    - throughput:{RESULTS_DIR / 'throughput_k_sweep_FULL.csv'}")