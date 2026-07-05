"""
alert_rate.py
-------------
Stage-1 invocation rate에서 시간당 alert 수 역산.

공식:
  alerts_per_hour = invocation_rate × ticks_per_hour
  ticks_per_hour  = 3600 (1초 bar 기준)

실행:
  python alert_rate.py
"""

import json
from pathlib import Path

# ── 설정 ──────────────────────────────────────────────
TICKS_PER_HOUR = 3600  # 1초 bar 기준

# Stage-2 invocation rate (Table tab:stage2 에서)
# 각 모델별 K값에 따른 invocation rate
INVOCATION_RATES = {
    "Logistic": {1: 0.0005, 5: 0.0025, 10: 0.0020, 20: 0.0010, 50: 0.0005},
    "XGBoost":  {1: 0.0035, 5: 0.0025, 10: 0.0020, 20: 0.0020, 50: 0.0045},
    "RandomForest": {1: 0.0000, 5: 0.0000, 10: 0.0000, 20: 0.0000, 50: 0.0000},
}

# Stage-1 precision (validation set 기준)
STAGE1_PRECISION = {
    "XGBoost":     0.1667,
    "RandomForest": 0.1577,
    "Logistic":    0.1748,
}

# Stage-1 invocation rate (= Stage-2 invocation의 상위 단계)
# Stage-1은 모든 tick에서 실행되므로 invocation rate = 1.0
# Stage-2 invocation rate가 곧 "alert 발생률"

# ── 계산 ──────────────────────────────────────────────
print("=" * 65)
print("DASH Alert Rate Analysis")
print("  Base: 1-second bars, 3600 ticks/hour per symbol")
print("=" * 65)

for model, rates in INVOCATION_RATES.items():
    print(f"\n[{model}]")
    print(f"  {'K':>4}  {'InvRate':>9}  {'Alerts/hr/sym':>14}  {'True Alerts/hr':>15}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*14}  {'-'*15}")
    for K, rate in rates.items():
        alerts_per_hr_per_sym = rate * TICKS_PER_HOUR
        # True alerts = precision × total alerts (Stage-1 precision 기준)
        true_alerts = alerts_per_hr_per_sym * STAGE1_PRECISION[model]
        print(f"  {K:>4}  {rate:>9.4f}  {alerts_per_hr_per_sym:>14.1f}  {true_alerts:>15.1f}")

print("\n" + "=" * 65)
print("Summary (K=10, canonical configuration):")
print("=" * 65)
for model in ["XGBoost", "RandomForest", "Logistic"]:
    rate = INVOCATION_RATES[model][10]
    alerts = rate * TICKS_PER_HOUR
    true_alerts = alerts * STAGE1_PRECISION[model]
    print(f"  {model:<15}: {alerts:.1f} alerts/hr/symbol "
          f"(~{true_alerts:.1f} actionable, "
          f"precision={STAGE1_PRECISION[model]:.4f})")

print("\nNote: 'True Alerts' = alerts × Stage-1 precision")
print("      Stage-2 is invoked only on these candidates.")
print("      For K=50 symbols: multiply per-symbol rate by K.")

