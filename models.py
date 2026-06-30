# models.py (patched: scaler inference + feat_dim from cfg + gate ablation + tuned thresholds)
# [LSTM REMOVED]

from __future__ import annotations

import time
import json
from typing import Optional, Tuple, Dict, Any

import numpy as np
import joblib
import torch
from xgboost import XGBClassifier

import config as cfg  # single source of truth


# =============================================================================
# 0) Constants: Label Semantics (Consistency)
# =============================================================================
Y_NORMAL = 0
Y_NEG = 1
Y_POS = 2

FEAT_DIM = len(cfg.FEATURE_COLS)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_predict_proba_binary(model, x_row: np.ndarray) -> float:
    """
    Assumes binary classifier and returns P(class=1).
    (Stage-1: shock(1) vs normal(0), Stage-2: pos(1) vs neg(0))
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_row)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"predict_proba returned shape {proba.shape}, expected (N,2)+")
        return float(proba[0, 1])
    y = model.predict(x_row)
    return float(y[0])


# =============================================================================
# Threshold loading (calibrated operating point)
# =============================================================================
def _load_thresholds_if_exists(tag: str) -> Optional[Dict[str, Any]]:
    """
    Expected file: cfg.ART_DIR / f"thresholds_{tag}.json"
    Expected structure (recommended):
      {
        "models": {
          "xgb": {"s1": {"thr": ...}, "s2": {"thr": ...}},
          "rf":  {"s1": {"thr": ...}, "s2": {"thr": ...}},
          "lr":  {"s1": {"thr": ...}, "s2": {"thr": ...}}
        }
      }
    """
    p = cfg.ART_DIR / f"thresholds_{tag}.json"
    if not p.exists():
        print(f"[!] thresholds not found at {p}. Using cfg.TAU_CONF and tau_s2=0.5.")
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to read thresholds file {p}: {type(e).__name__}: {e}")
        return None


def _model_key(model_name: str) -> str:
    """
    Map human model_name to thresholds JSON key.
    """
    mapping = {
        "XGBoost": "xgb",
        "RandomForest": "rf",
        "Logistic": "lr",
    }
    if model_name not in mapping:
        raise ValueError(f"Unknown model type for thresholds: {model_name}")
    return mapping[model_name]


class DASHModelWrapper:
    """
    Stage-1: Shock detection -> pi_1 = P(shock)
    Gate: if pi_1 < tau_conf -> Normal (unless force_stage2=True)
    Stage-2: Direction inference -> pi_2 = P(pos | shock)
    Final:
      if gate pass: yhat = POS if pi_2 >= tau_s2 else NEG
    """

    def __init__(
        self,
        model_type: str,
        s1_model,
        s2_model,
        tau_conf: float = cfg.TAU_CONF,
        tau_s2: float = 0.5,  # ✅ Stage-2 tuned threshold (default naive=0.5)
        device: Optional[torch.device] = None,
        scaler: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type
        self.s1 = s1_model
        self.s2 = s2_model

        self.tau_conf = float(tau_conf)
        self.tau_s2 = float(tau_s2)

        self.lookback_w = int(cfg.LOOKBACK_W)
        self.device = device or _get_device()

        # scaler dict: {"mu":..., "sd":..., "feat_cols":[...]}
        self.scaler = scaler
        if self.scaler is not None:
            self._mu = np.asarray(self.scaler["mu"], dtype=np.float32)
            self._sd = np.asarray(self.scaler["sd"], dtype=np.float32)
            self._feat_cols = list(self.scaler.get("feat_cols", []))
            if self._mu.shape[0] != FEAT_DIM or self._sd.shape[0] != FEAT_DIM:
                raise ValueError(
                    f"Scaler dim mismatch. scaler has {self._mu.shape[0]} dims, "
                    f"but cfg.FEATURE_COLS has {FEAT_DIM} dims."
                )
        else:
            self._mu = None
            self._sd = None
            self._feat_cols = []

    def _apply_scaler(self, x: np.ndarray) -> np.ndarray:
        """
        Apply (x - mu) / sd for both 1D and 2D inputs.
        IMPORTANT: input feature ordering must match training feat_cols order.
        """
        if self._mu is None or self._sd is None:
            return np.asarray(x, dtype=np.float32)

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            if x.shape[0] != FEAT_DIM:
                raise ValueError(f"Expected x dim {FEAT_DIM}, got {x.shape}")
            return (x - self._mu) / self._sd

        if x.ndim == 2:
            # For tabular models, take the last row (most recent)
            if x.shape[1] != FEAT_DIM:
                raise ValueError(f"Expected x shape (T,{FEAT_DIM}), got {x.shape}")
            x_last = x[-1, :]
            return (x_last - self._mu) / self._sd

        raise ValueError(f"Expected x ndim 1 or 2, got {x.ndim}")

    def predict_hierarchical_timed(
        self,
        x_input: np.ndarray,
        tau: Optional[float] = None,          # optional override for gate (Stage-1)
        force_stage2: bool = False,           # gate ablation flag
        tau_s2: Optional[float] = None,       # optional override for Stage-2 threshold
    ) -> Tuple[int, float, Optional[float], float, float]:
        """
        Returns: (yhat, pi_1, pi_2_or_None, stage1_ms, stage2_ms)
        """
        tau_gate = self.tau_conf if tau is None else float(tau)
        tau_dir = self.tau_s2 if tau_s2 is None else float(tau_s2)

        # -------------------------
        # Stage 1 (Timed)
        # -------------------------
        t1_0 = time.perf_counter_ns()

        x_scaled = self._apply_scaler(np.asarray(x_input, dtype=np.float32))
        x_row = x_scaled.reshape(1, -1)
        pi_1 = _safe_predict_proba_binary(self.s1, x_row)

        t1_1 = time.perf_counter_ns()
        b1_ms = (t1_1 - t1_0) / 1e6

        # Gate (Early Exit) unless ablation forces stage2
        if (not force_stage2) and (pi_1 < tau_gate):
            return Y_NORMAL, float(pi_1), None, float(b1_ms), 0.0

        # -------------------------
        # Stage 2 (Timed)
        # -------------------------
        t2_0 = time.perf_counter_ns()

        pi_2 = _safe_predict_proba_binary(self.s2, x_row)

        t2_1 = time.perf_counter_ns()
        b2_ms = (t2_1 - t2_0) / 1e6

        yhat = Y_POS if pi_2 >= tau_dir else Y_NEG
        return int(yhat), float(pi_1), float(pi_2), float(b1_ms), float(b2_ms)


def _load_scaler_if_exists(tag: str) -> Optional[Dict[str, Any]]:
    p = cfg.ART_DIR / f"scaler_{tag}.pkl"
    if p.exists():
        obj = joblib.load(p)
        if "mu" in obj and "sd" in obj:
            return obj
    print(f"[!] scaler not found at {p}. Inference will run WITHOUT scaling (may hurt performance).")
    return None


def load_dash_harness(
    model_name: str,
    lambda_val: float = cfg.LAMBDA,
    alpha: float = cfg.ALPHA,
    h: int = cfg.HORIZON_H,
    tau_conf: float = cfg.TAU_CONF,          # fallback default (overridden by thresholds if present)
    lookback_w: int = cfg.LOOKBACK_W,        # kept for API compatibility (unused)
) -> DASHModelWrapper:
    """
    Loads artifacts based on canonical cfg.make_tag().

    Also loads calibrated thresholds from thresholds_{tag}.json when available:
      - Stage-1 gate threshold (tau_conf) from tuned S1 thr
      - Stage-2 direction threshold (tau_s2) from tuned S2 thr
    """
    tag = cfg.make_tag(h=h, alpha=alpha, lambda_val=lambda_val)
    device = _get_device()

    print(f"[*] Loading DASH Harness: {model_name} (Tag: {tag})")

    scaler = _load_scaler_if_exists(tag)

    # -------------------------
    # Load models
    # -------------------------
    if model_name == "XGBoost":
        s1 = XGBClassifier()
        s2 = XGBClassifier()
        s1.load_model(str(cfg.ART_DIR / f"s1_xgb_{tag}.json"))
        s2.load_model(str(cfg.ART_DIR / f"s2_xgb_{tag}.json"))

    elif model_name == "RandomForest":
        s1 = joblib.load(cfg.ART_DIR / f"s1_rf_{tag}.pkl")
        s2 = joblib.load(cfg.ART_DIR / f"s2_rf_{tag}.pkl")

    elif model_name == "Logistic":
        s1 = joblib.load(cfg.ART_DIR / f"s1_lr_{tag}.pkl")
        s2 = joblib.load(cfg.ART_DIR / f"s2_lr_{tag}.pkl")

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # -------------------------
    # Load tuned thresholds (optional)
    # -------------------------
    tau_gate = float(tau_conf)   # fallback
    tau_s2 = 0.5                 # fallback

    thr_json = _load_thresholds_if_exists(tag)
    if thr_json is not None:
        try:
            key = _model_key(model_name)
            tau_gate = float(thr_json["models"][key]["s1"]["thr"])
            tau_s2 = float(thr_json["models"][key]["s2"]["thr"])
            print(f"[*] Loaded tuned thresholds: tau_conf={tau_gate:.6f}, tau_s2={tau_s2:.6f}")
        except Exception as e:
            print(f"[!] thresholds_{tag}.json found but parse failed: {type(e).__name__}: {e}")
            print("[!] Falling back to cfg.TAU_CONF and tau_s2=0.5.")

    return DASHModelWrapper(
        model_type=model_name,
        s1_model=s1,
        s2_model=s2,
        tau_conf=tau_gate,
        tau_s2=tau_s2,
        device=device,
        scaler=scaler,
    )