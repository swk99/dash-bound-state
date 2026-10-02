# DASH evidence runs — 2 October 2026

Experiments behind the JSS/ICPE submission. Code under test is the repository root at this commit
(combined SHA256 of root `*.py` = `7a47b365f9e6f648…`; artifact hashes in `replication/expected_sha256.json`).
Full specification: `reports/DASH_Final_Experiment_Specification_v1.2.docx`. All summaries: `reports/DASH_all_results_20261002.xlsx`.

| Folder | What | Run ID | Environment |
|---|---|---|---|
| `replication/` | P0-direct replication, XGBoost, W=30, K=1/100, A/B1/B2, 5 repeats × 5,000 requests, inference threads = 1; Stage-2 E0 on rule-selected windows | `rep_cloud_20261002T141735Z`, `rep_laptop_20261002T201141Z` | cloud container (2 vCPU, Redis 7.0.15) and author laptop (WSL2, Redis 8.0.5) |
| `robustness/` | Single-host controlled conditions: W ∈ {10,30,60,120}; client CPU 1 vs 2 cores | `robust_cloud_20261002` | cloud container |
| `network_profiles/` | Delay/jitter profiles P0–P3 through an experiment-only Python TCP proxy (netem unavailable), K=1/10/100; Lua update cost; gating ablation | `dash3_20261002T1122Z` | cloud container, default XGBoost threading |
| `state_checks/` | Lua correctness (3 methods), state-only E2 memory, first E0, historical E6 re-analysis | `dash_20261002T110902Z_3a641a` | cloud container |

## Headline results
- Bounded state: for fixed W, d the Redis footprint is constant in N (std 0 over 5 repeats); logical payload 4·d·min(N,W) bytes. Not W-independent.
- Path equivalence: A/B1/B2 labels identical (0 mismatches) for Logistic, RF and XGBoost in both environments, including Stage-2 ticks.
- P99 B1/A (median of 5 repeats, pre-registered rule): K=1 cloud 1.28 vs laptop 0.89 → **direction not maintained**; K=100 cloud 1.17 vs laptop 1.43 → maintained.
- Mechanism (W sweep): B1 state cost grows with W (0.39→0.61 ms), A is flat (max/min 1.09).

## Reproducing
- Raw per-request data are stored as parquet checkpoints (`ckpt*/`). The large long-format CSVs (200–560 MB) are regenerated with the `assemble*.py` scripts; they are not committed.
- Run a new environment: see `replication/README_labpc.md` (`bash setup_and_run.sh <label>`). The run aborts if any SHA256 differs from `expected_sha256.json`.
- Scripts were run from `/home/claude/...` paths; set `DASH_REPO`, `DASH_DATA_GLOB`, `DASH_OUT` (replication) or edit the path constants at the top of the assemble/robustness scripts before rerunning.
- Input data (btcusdt flush parquet, 74,855 rows) are not in the repository; manifest hash `dff1a605…` is checked by the run script.

## Limitations (summary)
Two heterogeneous environments only; closed-loop single client; 5 repeats, descriptive statistics only; proxy-based network delays are synthetic and not comparable with P0-direct runs; robustness conditions vary one factor on one host. Full list: spec §14 and the `Limitations` sheet.
