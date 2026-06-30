# engine.py (paper-consistent: A/B1/B2 pipelines, shared wrapper, gate ablation)
from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np
import redis

import config as cfg
from models import DASHModelWrapper


def _feat_dim() -> int:
    return int(len(cfg.FEATURE_COLS))


def _to_feat_vec(feat_vec: np.ndarray) -> np.ndarray:
    """
    Ensure 1D float32 feature vector with canonical dimension (=len(cfg.FEATURE_COLS)).
    Truncates if longer, errors if shorter.
    """
    d = _feat_dim()
    x = np.asarray(feat_vec, dtype=np.float32).reshape(-1)
    if x.size < d:
        raise ValueError(f"feat_vec too small: got {x.size}, need {d}")
    if x.size != d:
        x = x[:d]
    return x


def _from_redis_bytes(b: bytes) -> np.ndarray:
    """
    Read exactly feat_dim float32 values from bytes.
    This prevents old/corrupted entries (e.g., 12-d) from propagating.
    """
    d = _feat_dim()
    x = np.frombuffer(b, dtype=np.float32, count=d)
    if x.size != d:
        # If bytes shorter than expected, that's a real corruption.
        raise ValueError(f"Redis entry decode failed: got {x.size} floats, expected {d}")
    return x


class ProposedStatefulEngine:
    """
    A: Proposed Stateful Engine (Redis-based, O(1) state maintenance w.r.t. stream length N).
    - Each tick: LPUSH + LTRIM to maintain bounded window size W.
    - Inference input:
        - non-LSTM: current feature vector (O(1))
        - LSTM: fetch full window (LRANGE) -> O(W) (unavoidable for seq models)
    """
    def __init__(self, wrapper: DASHModelWrapper, rds: redis.Redis):
        self.wrapper = wrapper
        self.rds = rds

    def process_tick(
        self,
        symbol: str,
        feat_vec: np.ndarray,
        tau: float,
        force_stage2: bool = False,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter_ns()

        W = int(self.wrapper.lookback_w)
        key = f"dash:state:{symbol}"

        feat = _to_feat_vec(feat_vec)

        # ---- A: Hot-tier update (LPUSH/LTRIM) ----
        t_a0 = time.perf_counter_ns()
        pipe = self.rds.pipeline(transaction=False)
        pipe.lpush(key, feat.tobytes())
        pipe.ltrim(key, 0, W - 1)
        pipe.execute()
        t_a1 = time.perf_counter_ns()
        A_ms = (t_a1 - t_a0) / 1e6

        # ---- Prepare inference input ----
        if self.wrapper.model_type == "LSTM":
            raw_list = self.rds.lrange(key, 0, W - 1)[::-1]  # oldest -> newest
            if len(raw_list) == 0:
                # fallback (should be rare)
                x_infer = feat.reshape(1, -1)
            else:
                x_infer = np.stack([_from_redis_bytes(b) for b in raw_list])
        else:
            x_infer = feat  # (6,)

        # ---- B1/B2: Model inference (timed inside wrapper) ----
        yhat, p1, p2, B1_ms, B2_ms = self.wrapper.predict_hierarchical_timed(
            x_infer, tau=tau, force_stage2=force_stage2
        )

        t_end = time.perf_counter_ns()
        return {
            "engine": "A_ProposedStateful",
            "yhat": int(yhat),
            "A_ms": float(A_ms),
            "B1_ms": float(B1_ms),
            "B2_ms": float(B2_ms),
            "total_ms": float((t_end - t_start) / 1e6),
        }


class RedisFetchBaselineEngine:
    """
    B1: Redis-Fetch baseline (I/O-bound, O(W)).
    - Redis is only a passive store.
    - Each tick:
        1) write current tick (LPUSH/LTRIM) to keep same state as A
        2) fetch full window via LRANGE every tick (network + serialization)
    - This isolates Redis I/O overhead (vs A's incremental maintenance idea).

    ✅ Root fix:
      - Always store and decode exactly feat_dim floats (count=feat_dim).
      - Prevents 12-d vectors from ever reaching XGBoost.
    """
    def __init__(self, wrapper: DASHModelWrapper, rds: redis.Redis):
        self.wrapper = wrapper
        self.rds = rds

    def process_tick(
        self,
        symbol: str,
        feat_vec: np.ndarray,
        tau: float,
        force_stage2: bool = False,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter_ns()

        W = int(self.wrapper.lookback_w)
        key = f"dash:state:{symbol}"

        feat = _to_feat_vec(feat_vec)

        # ---- A: Update + Fetch (I/O-heavy) ----
        t_a0 = time.perf_counter_ns()
        pipe = self.rds.pipeline(transaction=False)
        pipe.lpush(key, feat.tobytes())      # ✅ always 6-d bytes
        pipe.ltrim(key, 0, W - 1)
        pipe.execute()

        raw_list = self.rds.lrange(key, 0, W - 1)[::-1]  # oldest -> newest
        if len(raw_list) == 0:
            window = feat.reshape(1, -1)
        else:
            # ✅ decode exactly 6 floats even if old entries had extra bytes
            window = np.stack([_from_redis_bytes(b) for b in raw_list])

        t_a1 = time.perf_counter_ns()
        A_ms = (t_a1 - t_a0) / 1e6

        # Inference input semantics
        if self.wrapper.model_type == "LSTM":
            x_infer = window
        else:
            x_infer = window[-1]  # (6,)

        yhat, p1, p2, B1_ms, B2_ms = self.wrapper.predict_hierarchical_timed(
            x_infer, tau=tau, force_stage2=force_stage2
        )

        t_end = time.perf_counter_ns()
        return {
            "engine": "B1_RedisFetch",
            "yhat": int(yhat),
            "A_ms": float(A_ms),
            "B1_ms": float(B1_ms),
            "B2_ms": float(B2_ms),
            "total_ms": float((t_end - t_start) / 1e6),
        }


class InMemoryRecomputeEngine:
    """
    B2: In-memory recomputation baseline (CPU-bound, O(W)).
    - Maintains sliding window in local memory (slicing).
    - Each tick recomputes rolling-like transforms (z-score + diffs) to emulate heavier compute.
    - This isolates CPU recomputation cost.
    """
    def __init__(self, wrapper: DASHModelWrapper, raw_buffer: np.ndarray, heavy_recompute: bool = True):
        self.wrapper = wrapper
        self.buffer = np.asarray(raw_buffer, dtype=np.float32)
        self.heavy_recompute = bool(heavy_recompute)

        # Optional safety: if raw_buffer has more than feat_dim cols, keep only feat_dim
        d = _feat_dim()
        if self.buffer.ndim == 2 and self.buffer.shape[1] != d:
            if self.buffer.shape[1] < d:
                raise ValueError(f"raw_buffer has too few cols: {self.buffer.shape[1]} < {d}")
            self.buffer = self.buffer[:, :d]

    def process_tick(
        self,
        current_idx: int,
        tau: float,
        force_stage2: bool = False,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter_ns()

        W = int(self.wrapper.lookback_w)

        # ---- A: Slicing & recompute (CPU-heavy) ----
        t_a0 = time.perf_counter_ns()
        start = max(0, current_idx - W + 1)
        window = self.buffer[start: current_idx + 1]

        if self.heavy_recompute:
            _ = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-9)
            _ = np.diff(window, axis=0)

        t_a1 = time.perf_counter_ns()
        A_ms = (t_a1 - t_a0) / 1e6

        x_infer = window if self.wrapper.model_type == "LSTM" else window[-1]
        yhat, p1, p2, B1_ms, B2_ms = self.wrapper.predict_hierarchical_timed(
            x_infer, tau=tau, force_stage2=force_stage2
        )

        t_end = time.perf_counter_ns()
        return {
            "engine": "B2_InMemoryRecompute",
            "yhat": int(yhat),
            "A_ms": float(A_ms),
            "B1_ms": float(B1_ms),
            "B2_ms": float(B2_ms),
            "total_ms": float((t_end - t_start) / 1e6),
        } 