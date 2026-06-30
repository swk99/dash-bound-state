# train_2.py (paper-consistent + safe artifacts + model-specific thresholds + optional e2e)  [LSTM REMOVED]
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score

import config as cfg


# =============================================================================
# Constants
# =============================================================================
Y_NORMAL = 0
Y_NEG = 1
Y_POS = 2

FEAT_DIM = len(cfg.FEATURE_COLS)


# =============================================================================
# Basic helpers
# =============================================================================
def infer_label_col(df: pd.DataFrame) -> str:
    candidates = ["y", "label", "labels", "y_true", "target", "class"]
    for c in candidates:
        if c in df.columns:
            vals = df[c].dropna().unique()
            if set(map(int, vals)).issubset({0, 1, 2}):
                return c
    raise ValueError(
        "Could not infer label column. Need 3-class label in {0,1,2}. "
        f"Columns present: {list(df.columns)[:40]} ..."
    )


def infer_feat_cols(df: pd.DataFrame, explicit: Optional[List[str]] = None) -> List[str]:
    """
    Prefer config-driven feature ordering to guarantee training/inference consistency.
    """
    if explicit:
        missing = [c for c in explicit if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return explicit

    if hasattr(cfg, "FEATURE_COLS"):
        cols = list(getattr(cfg, "FEATURE_COLS"))
        if all(c in df.columns for c in cols):
            return cols

    if hasattr(cfg, "FEAT_COLS"):
        cols = list(getattr(cfg, "FEAT_COLS"))
        if all(c in df.columns for c in cols):
            return cols

    default6 = ["r_t", "sigma_hat", "OFI_t", "Imbalance_t", "VolSpike_t", "msg_count"]
    if all(c in df.columns for c in default6):
        return default6

    raise ValueError(
        "Could not infer feature columns. Provide --feat-cols or define cfg.FEATURE_COLS/FEAT_COLS. "
        f"Default tried: {default6}. Columns: {list(df.columns)[:40]} ..."
    )


def load_df(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    suf = data_path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(data_path)
    if suf == ".csv":
        return pd.read_csv(data_path)
    if suf == ".feather":
        return pd.read_feather(data_path)
    raise ValueError(f"Unsupported file type: {suf}. Use parquet/csv/feather.")


def sort_time_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["sec", "timestamp", "ts", "time", "t"]:
        if c in df.columns:
            return df.sort_values(c).reset_index(drop=True)
    return df.reset_index(drop=True)


def train_val_split_time_with_gap(
    df: pd.DataFrame, val_ratio: float, gap: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time split with a GAP to prevent horizon-label leakage:
      - train uses [0 : split-gap)
      - val   uses [split : end)
    """
    n = len(df)
    split = int(n * (1 - val_ratio))
    split = max(1, min(split, n - 1))

    tr_end = max(1, split - gap)
    if tr_end <= 1:
        raise ValueError(
            f"Not enough rows for split with gap={gap}. "
            f"n={n}, split={split}, tr_end={tr_end}"
        )
    df_tr = df.iloc[:tr_end].copy()
    df_va = df.iloc[split:].copy()
    if len(df_va) < 2:
        raise ValueError("Validation split too small. Reduce val_ratio or use more data.")
    return df_tr, df_va


def fit_scaler(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-8
    return mu, sd


def apply_scaler(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def pos_weight_binary(y01: np.ndarray) -> float:
    y01 = np.asarray(y01).astype(int)
    pos = int(y01.sum())
    neg = int(len(y01) - pos)
    if pos == 0:
        raise ValueError("No positive samples in y. Cannot compute scale_pos_weight.")
    return neg / pos


# =============================================================================
# Threshold selection helpers
# =============================================================================
def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)


def _f_beta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * _safe_div(p * r, b2 * p + r)


def choose_threshold(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    *,
    policy: str = "f1",
    beta: float = 1.0,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - thr: chosen threshold
      - precision, recall, f1, fbeta
      - ap: PR-AUC (average_precision_score)
      - policy info
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    if len(y_true) != len(p_pos):
        raise ValueError("y_true and p_pos must have same length.")
    if len(y_true) < 5:
        raise ValueError("Too few samples to choose threshold reliably.")

    prec, rec, thr = precision_recall_curve(y_true, p_pos)
    prec2 = prec[1:]
    rec2 = rec[1:]
    thr2 = thr

    ap = float(average_precision_score(y_true, p_pos))

    if len(thr2) == 0:
        return {
            "thr": 0.5,
            "precision": float(prec[-1]),
            "recall": float(rec[-1]),
            "f1": float(_f_beta(float(prec[-1]), float(rec[-1]), 1.0)),
            "fbeta": float(_f_beta(float(prec[-1]), float(rec[-1]), beta)),
            "ap": ap,
            "policy": policy,
            "beta": beta,
            "min_precision": min_precision,
            "min_recall": min_recall,
        }

    f1 = np.array([_f_beta(float(p), float(r), 1.0) for p, r in zip(prec2, rec2)], dtype=float)
    fbeta = np.array([_f_beta(float(p), float(r), beta) for p, r in zip(prec2, rec2)], dtype=float)

    mask = np.ones_like(thr2, dtype=bool)
    if min_precision is not None:
        mask &= (prec2 >= float(min_precision))
    if min_recall is not None:
        mask &= (rec2 >= float(min_recall))

    def pick_best(idx: np.ndarray, score: np.ndarray) -> int:
        best = idx[np.argmax(score[idx])]
        best_score = score[best]
        ties = idx[np.where(np.isclose(score[idx], best_score, atol=1e-12))[0]]
        if len(ties) > 1:
            best = ties[np.lexsort((thr2[ties], rec2[ties], prec2[ties]))][-1]
        return int(best)

    valid = np.where(mask)[0]
    if len(valid) == 0:
        valid = np.arange(len(thr2))

    policy = policy.lower().strip()
    if policy == "recall_at_precision":
        score = rec2
        best_i = pick_best(valid, score)
    elif policy == "precision_at_recall":
        score = prec2
        best_i = pick_best(valid, score)
    elif policy == "fbeta":
        score = fbeta
        best_i = pick_best(valid, score)
    else:
        score = f1
        best_i = pick_best(valid, score)

    return {
        "thr": float(thr2[best_i]),
        "precision": float(prec2[best_i]),
        "recall": float(rec2[best_i]),
        "f1": float(f1[best_i]),
        "fbeta": float(fbeta[best_i]),
        "ap": ap,
        "policy": policy,
        "beta": float(beta),
        "min_precision": None if min_precision is None else float(min_precision),
        "min_recall": None if min_recall is None else float(min_recall),
    }


# =============================================================================
# Optional imbalance calibration helpers (kept from your code)
# =============================================================================
def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


def adjust_proba_binary(
    p_pos: np.ndarray,
    pi_neg: float,
    pi_pos: float,
    tau: float,
) -> np.ndarray:
    """
    For models that output p_pos directly (XGB/RF/LR):
      logit(p) + tau*log(pi_pos/pi_neg) -> sigmoid
    """
    if tau == 0.0:
        return p_pos
    p = np.clip(p_pos.astype(np.float64), 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    logit = logit + tau * np.log(max(pi_pos, 1e-12) / max(pi_neg, 1e-12))
    return 1.0 / (1.0 + np.exp(-logit))


# =============================================================================
# E2E helper (3-class)
# =============================================================================
def e2e_predict_3class(
    *,
    p_s1: np.ndarray,
    thr_s1: float,
    p_s2: np.ndarray,
    thr_s2: float,
) -> np.ndarray:
    """
    Given per-row probabilities for S1 (shock) and S2 (pos within shock),
    return 3-class prediction:
      - if shock < thr_s1 => NORMAL(0)
      - else if pos < thr_s2 => NEG(1)
      - else POS(2)
    """
    y_hat = np.full(len(p_s1), Y_NORMAL, dtype=int)
    shock_hat = (p_s1 >= thr_s1)
    y_hat[shock_hat] = np.where(p_s2[shock_hat] >= thr_s2, Y_POS, Y_NEG)
    return y_hat


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data", type=str, required=False,
        help="Path to labeled dataset (parquet/csv/feather). If omitted, tries cfg.TRAIN_PATH/cfg.DATA_PATH/cfg.LABELED_PATH."
    )
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument(
        "--gap", type=int, default=None,
        help="Gap to prevent horizon leakage. Default=cfg.HORIZON_H."
    )

    ap.add_argument(
        "--feat-cols", type=str, default="",
        help="Comma-separated feature columns. If empty uses cfg.FEATURE_COLS/FEAT_COLS or default 6."
    )
    ap.add_argument(
        "--label-col", type=str, default="",
        help="3-class label column name. If empty auto-detect."
    )

    # Threshold tuning policy
    ap.add_argument(
        "--thr-policy",
        type=str,
        default="f1",
        choices=["f1", "fbeta", "recall_at_precision", "precision_at_recall"],
        help="How to choose thresholds from PR curve on validation."
    )
    ap.add_argument("--beta", type=float, default=1.0, help="beta for F-beta (used if thr-policy=fbeta).")
    ap.add_argument("--min-precision", type=float, default=None, help="Constraint: precision >= this (for recall_at_precision).")
    ap.add_argument("--min-recall", type=float, default=None, help="Constraint: recall >= this (for precision_at_recall).")

    # Optional end-to-end 3-class report
    ap.add_argument("--report-e2e", action="store_true", help="Also print 3-class end-to-end report on validation (row-level).")

    args = ap.parse_args()

    # Resolve data path
    data_path: Optional[Path] = None
    if args.data:
        data_path = Path(args.data)
    else:
        for cand_name in ["TRAIN_PATH", "DATA_PATH", "LABELED_PATH"]:
            if hasattr(cfg, cand_name):
                cand = Path(getattr(cfg, cand_name))
                if cand.exists():
                    data_path = cand
                    break
    if data_path is None:
        raise ValueError(
            "No --data provided and could not find cfg.TRAIN_PATH/cfg.DATA_PATH/cfg.LABELED_PATH.\n"
            r'Run: python train_2.py --data "C:\path\to\labeled_train.parquet"'
        )

    df = load_df(data_path)
    df = sort_time_if_possible(df)
    print(f"[*] Loaded data: {data_path} | rows={len(df)} | cols={len(df.columns)}")

    # Columns
    feat_cols = [c.strip() for c in args.feat_cols.split(",") if c.strip()] if args.feat_cols else None
    feat_cols = infer_feat_cols(df, feat_cols)
    label_col = args.label_col.strip() if args.label_col else infer_label_col(df)

    if len(feat_cols) != FEAT_DIM:
        print(
            f"[!] WARNING: feat_cols dim={len(feat_cols)} but cfg.FEATURE_COLS dim={FEAT_DIM}. "
            f"Ensure inference uses SAME ordering/dim as training. feat_cols={feat_cols}"
        )

    print(f"[*] Using label_col='{label_col}' | feat_cols={feat_cols}")

    # Split with gap
    gap = args.gap if args.gap is not None else int(getattr(cfg, "HORIZON_H", 30))
    df_tr, df_va = train_val_split_time_with_gap(df, val_ratio=args.val_ratio, gap=gap)
    print(f"[*] Split: train={len(df_tr)} | val={len(df_va)} | gap={gap}")

    # Extract arrays
    X_tr_raw = df_tr[feat_cols].astype(np.float32).to_numpy()
    X_va_raw = df_va[feat_cols].astype(np.float32).to_numpy()

    y3_tr = df_tr[label_col].astype(int).to_numpy()
    y3_va = df_va[label_col].astype(int).to_numpy()

    # Stage-1 (binary): shock vs normal
    y1_tr = (y3_tr != Y_NORMAL).astype(int)
    y1_va = (y3_va != Y_NORMAL).astype(int)

    # Stage-2 (binary): pos vs neg, meaningful only when shock
    shock_tr = (y3_tr != Y_NORMAL)
    shock_va = (y3_va != Y_NORMAL)
    y2_tr = (y3_tr == Y_POS).astype(int)
    y2_va = (y3_va == Y_POS).astype(int)

    if shock_tr.sum() == 0 or shock_va.sum() == 0:
        raise ValueError(
            "Stage-2 requires shock samples in BOTH train and val.\n"
            "Increase dataset size, reduce alpha, or adjust split/val-ratio."
        )

    # scaler (train only)
    mu, sd = fit_scaler(X_tr_raw)
    X_tr = apply_scaler(X_tr_raw, mu, sd)
    X_va = apply_scaler(X_va_raw, mu, sd)

    # artifacts
    tag = cfg.make_tag(h=cfg.HORIZON_H, alpha=cfg.ALPHA, lambda_val=cfg.LAMBDA)
    art_dir = Path(cfg.ART_DIR)
    art_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = art_dir / f"scaler_{tag}.pkl"
    joblib.dump({"mu": mu, "sd": sd, "feat_cols": feat_cols}, scaler_path)
    print(f"[*] Tag={tag}")
    print(f"[*] Saving artifacts to: {art_dir}")
    print(f"[OK] saved {scaler_path.name} (IMPORTANT: apply at inference too)")

    # logit-adjust settings for non-neural proba
    use_la = bool(getattr(cfg, "USE_LOGIT_ADJ", False))
    tau = float(getattr(cfg, "LA_TAU", 0.0))

    # priors for S1 and S2 (train-based)
    pi_pos_s1 = float(y1_tr.mean())
    pi_neg_s1 = 1.0 - pi_pos_s1

    # Stage-2 shock-only arrays (train/val)
    X2_tr = X_tr[shock_tr]
    X2_va = X_va[shock_va]
    y2_tr_shock = y2_tr[shock_tr]
    y2_va_shock = y2_va[shock_va]

    pi_pos_s2 = float(y2_tr_shock.mean())
    pi_neg_s2 = 1.0 - pi_pos_s2

    # thresholds summary
    thr_summary: Dict[str, Any] = {
        "tag": tag,
        "thr_policy": args.thr_policy,
        "beta": float(args.beta),
        "min_precision": None if args.min_precision is None else float(args.min_precision),
        "min_recall": None if args.min_recall is None else float(args.min_recall),
        "use_logit_adjust": bool(use_la),
        "la_tau": float(tau),
        "models": {},
    }

    def tune_and_report_binary(
        *,
        model_name: str,
        stage: str,
        y_true: np.ndarray,
        p_pos: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        info = choose_threshold(
            y_true, p_pos,
            policy=args.thr_policy,
            beta=args.beta,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        thr = float(info["thr"])
        print(
            f"\n[{model_name} {stage}] PR-AUC(AP)={info['ap']:.4f} | chosen_thr={thr:.6f} "
            f"(P={info['precision']:.4f}, R={info['recall']:.4f}, F1={info['f1']:.4f}, Fβ={info['fbeta']:.4f})"
        )
        print(classification_report(y_true, (p_pos >= thr).astype(int), digits=4))
        return thr, info

    # -------------------------------------------------------------------------
    # 1) XGBoost (S1/S2) -> booster JSON
    # -------------------------------------------------------------------------
    spw_s1 = pos_weight_binary(y1_tr)
    xgb_s1 = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,          # early stopping으로 알아서 줄임
        learning_rate=0.03,         # 0.01~0.05 권장
        max_depth=3,
        min_child_weight=8,         # ↑ 과적합/노이즈 억제 (S1에 도움 큼)
        gamma=0.5,                  # split 보수적으로 → FP 줄이기 방향
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,              # L1: 불필요 feature 억제
        reg_lambda=2.0,             # L2: 안정화
        max_delta_step=1,           # imbalance에서 확률 튐 완화
        eval_metric="logloss",
        tree_method="hist",
        random_state=100,
        scale_pos_weight=spw_s1,
    )
    xgb_s1.fit(X_tr, y1_tr)
    xgb_s1_path = art_dir / f"s1_xgb_{tag}.json"
    xgb_s1.get_booster().save_model(str(xgb_s1_path))

    spw_s2 = pos_weight_binary(y2_tr_shock)
    xgb_s2 = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=10,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        max_delta_step=1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=spw_s2,
        early_stopping_rounds=80,
    )
    #xgb_s2.fit(X2_tr, y2_tr_shock)
    # S2: pos vs neg (shock rows only)
    xgb_s2.fit(
        X2_tr, y2_tr_shock,
        eval_set=[(X2_va, y2_va_shock)],   # <-- 너 코드의 validation 변수명에 맞게!
        verbose=False
    )
    xgb_s2_path = art_dir / f"s2_xgb_{tag}.json"
    xgb_s2.get_booster().save_model(str(xgb_s2_path))

    print("[OK] saved", xgb_s1_path.name, "and", xgb_s2_path.name)

    p_xgb_s1 = xgb_s1.predict_proba(X_va)[:, 1]
    if use_la:
        p_xgb_s1 = adjust_proba_binary(p_xgb_s1, pi_neg=pi_neg_s1, pi_pos=pi_pos_s1, tau=tau)
    thr_xgb_s1, info_xgb_s1 = tune_and_report_binary(model_name="XGB", stage="S1 (shock vs normal)", y_true=y1_va, p_pos=p_xgb_s1)

    p_xgb_s2_va_shock = xgb_s2.predict_proba(X2_va)[:, 1]
    if use_la:
        p_xgb_s2_va_shock = adjust_proba_binary(p_xgb_s2_va_shock, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
    thr_xgb_s2, info_xgb_s2 = tune_and_report_binary(model_name="XGB", stage="S2 (pos vs neg | shock rows)", y_true=y2_va_shock, p_pos=p_xgb_s2_va_shock)

    thr_summary["models"]["xgb"] = {"s1": info_xgb_s1, "s2": info_xgb_s2}

    # -------------------------------------------------------------------------
    # 2) RandomForest (S1/S2) -> joblib
    # -------------------------------------------------------------------------
    rf_s1 = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=20,
        n_jobs=1,
        random_state=52,
        class_weight="balanced",
    )
    rf_s1.fit(X_tr, y1_tr)
    rf_s1_path = art_dir / f"s1_rf_{tag}.pkl"
    joblib.dump(rf_s1, rf_s1_path)

    rf_s2 = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=20,
        n_jobs=1,
        random_state=53,
        class_weight="balanced",
    )
    rf_s2.fit(X2_tr, y2_tr_shock)
    rf_s2_path = art_dir / f"s2_rf_{tag}.pkl"
    joblib.dump(rf_s2, rf_s2_path)

    print("[OK] saved", rf_s1_path.name, "and", rf_s2_path.name)

    p_rf_s1 = rf_s1.predict_proba(X_va)[:, 1]
    if use_la:
        p_rf_s1 = adjust_proba_binary(p_rf_s1, pi_neg=pi_neg_s1, pi_pos=pi_pos_s1, tau=tau)
    thr_rf_s1, info_rf_s1 = tune_and_report_binary(model_name="RF", stage="S1 (shock vs normal)", y_true=y1_va, p_pos=p_rf_s1)

    p_rf_s2_va_shock = rf_s2.predict_proba(X2_va)[:, 1]
    if use_la:
        p_rf_s2_va_shock = adjust_proba_binary(p_rf_s2_va_shock, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
    thr_rf_s2, info_rf_s2 = tune_and_report_binary(model_name="RF", stage="S2 (pos vs neg | shock rows)", y_true=y2_va_shock, p_pos=p_rf_s2_va_shock)

    thr_summary["models"]["rf"] = {"s1": info_rf_s1, "s2": info_rf_s2}

    # -------------------------------------------------------------------------
    # 3) Logistic Regression (S1/S2) -> joblib
    # -------------------------------------------------------------------------
    lr_s1 = LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
    )
    lr_s1.fit(X_tr, y1_tr)
    lr_s1_path = art_dir / f"s1_lr_{tag}.pkl"
    joblib.dump(lr_s1, lr_s1_path)

    lr_s2 = LogisticRegression(
        max_iter=5000,
        n_jobs=-1,
        class_weight="balanced",
        solver="lbfgs",
    )
    lr_s2.fit(X2_tr, y2_tr_shock)
    lr_s2_path = art_dir / f"s2_lr_{tag}.pkl"
    joblib.dump(lr_s2, lr_s2_path)

    print("[OK] saved", lr_s1_path.name, "and", lr_s2_path.name)

    p_lr_s1 = lr_s1.predict_proba(X_va)[:, 1]
    if use_la:
        p_lr_s1 = adjust_proba_binary(p_lr_s1, pi_neg=pi_neg_s1, pi_pos=pi_pos_s1, tau=tau)
    thr_lr_s1, info_lr_s1 = tune_and_report_binary(model_name="LR", stage="S1 (shock vs normal)", y_true=y1_va, p_pos=p_lr_s1)

    p_lr_s2_va_shock = lr_s2.predict_proba(X2_va)[:, 1]
    if use_la:
        p_lr_s2_va_shock = adjust_proba_binary(p_lr_s2_va_shock, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
    thr_lr_s2, info_lr_s2 = tune_and_report_binary(model_name="LR", stage="S2 (pos vs neg | shock rows)", y_true=y2_va_shock, p_pos=p_lr_s2_va_shock)

    thr_summary["models"]["lr"] = {"s1": info_lr_s1, "s2": info_lr_s2}

    # -------------------------------------------------------------------------
    # Save thresholds JSON (single source of truth for inference)
    # -------------------------------------------------------------------------
    thr_path = art_dir / f"thresholds_{tag}.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_summary, f, indent=2)
    print(f"\n[OK] saved {thr_path.name} (model-specific thresholds + tuning metadata)")

    # -------------------------------------------------------------------------
    # Optional end-to-end report (row-level) for each classical model
    # -------------------------------------------------------------------------
    if args.report_e2e:
        print("\n====================")
        print("[E2E] 3-class row-level report on validation")
        print(" - Uses each model's own tuned thresholds")
        print(" - S2 proba is trained on shock-only, but applied to ALL rows for e2e")
        print("====================")

        # XGB e2e
        p1 = p_xgb_s1
        p2_all = xgb_s2.predict_proba(X_va)[:, 1]
        if use_la:
            p2_all = adjust_proba_binary(p2_all, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
        yhat3 = e2e_predict_3class(p_s1=p1, thr_s1=thr_xgb_s1, p_s2=p2_all, thr_s2=thr_xgb_s2)
        print("\n[XGB E2E] 3-class report:")
        print(classification_report(y3_va, yhat3, digits=4))

        # RF e2e
        p1 = p_rf_s1
        p2_all = rf_s2.predict_proba(X_va)[:, 1]
        if use_la:
            p2_all = adjust_proba_binary(p2_all, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
        yhat3 = e2e_predict_3class(p_s1=p1, thr_s1=thr_rf_s1, p_s2=p2_all, thr_s2=thr_rf_s2)
        print("\n[RF E2E] 3-class report:")
        print(classification_report(y3_va, yhat3, digits=4))

        # LR e2e
        p1 = p_lr_s1
        p2_all = lr_s2.predict_proba(X_va)[:, 1]
        if use_la:
            p2_all = adjust_proba_binary(p2_all, pi_neg=pi_neg_s2, pi_pos=pi_pos_s2, tau=tau)
        yhat3 = e2e_predict_3class(p_s1=p1, thr_s1=thr_lr_s1, p_s2=p2_all, thr_s2=thr_lr_s2)
        print("\n[LR E2E] 3-class report:")
        print(classification_report(y3_va, yhat3, digits=4))

    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("\n[DONE] Artifacts created:")
    for pth in [
        scaler_path,
        xgb_s1_path, xgb_s2_path,
        rf_s1_path, rf_s2_path,
        lr_s1_path, lr_s2_path,
        thr_path,
    ]:
        print("  -", pth)


if __name__ == "__main__":
    main() 