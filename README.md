# DASH bounded-state execution

Two-stage streaming classifier with Redis bounded windows. State is bounded in
stream length N for fixed W and d; it is **Theta(W d)**, not independent of W.
See [theory and evidence](docs/theory-and-evidence.md) for assumptions and limits.

Use Python 3.11 and `pip install -r requirements.txt`. Existing sklearn artifacts
require 1.6.1. Redis must support Lua and MEMORY USAGE (Redis >=4).
Configure REDIS_HOST/REDIS_PORT and MinIO environment settings in config.py.
Legacy root model/scaler/threshold files remain supported.

- Atomic LPUSH/LTRIM: `redis_state.RedisWindowWriter` (ingestion and A/B1).
- E2: call `benchmark_2.run_memory_vs_nw(model_name, canonical_float32_data)`.
  Requires Redis, artifacts and sufficient real rows; defaults to five repeats.
- E0: `python validate_paths.py features.npy --model Logistic`.
- E6/conditional IID BCa: `python analyze_latency.py results/latency_k_Logistic.csv
  --latency-column total_ms --group-by engine k` (check CSV column names first).
- Tests: install `fakeredis[lua]`, then `python -m unittest discover -s tests`.

No E1 knee or W-independent footprint is claimed. Historical CSVs have not been
regenerated. Full model validation needs the target services and data.
