# DASH P0 direct replication: experiment PC run

Same script, same protocol as the cloud run (`rep_cloud_*`). Do not edit `run_replication.py`, since its SHA256 is checked.

## Preconditions
- Same repo files, artifacts, and btcusdt flush parquet set as the cloud run. The script checks these against `expected_sha256.json` and stops on mismatch.
- Redis reachable **directly** (no proxy, no netem). Use an experiment-only Redis if possible. The script never runs FLUSH* or SCRIPT FLUSH and deletes only its own `jss4:` keys.
- Python with sklearn 1.6.1, xgboost, redis, psutil, pandas, pyarrow, torch. Record any version differences; they appear in the CSV.
- Close other heavy workloads. Do not run anything else during the ~4 minutes.

## Run (Linux/macOS)
```bash
export DASH_REPO=/path/to/dash-bound-state-main
export DASH_DATA_GLOB='/path/to/btcusdt/flush_*.parquet'
export DASH_ENV_LABEL=labpc REDIS_HOST=localhost REDIS_PORT=6379
python run_replication.py          # ~4 min; writes ckpt_labpc/ + progress_labpc.log
python assemble_replication.py     # merges ckpt_cloud/ + ckpt_labpc/ -> dash_replication_results.csv
```
## Run (Windows PowerShell)
```powershell
$env:DASH_REPO="C:\path\dash-bound-state-main"; $env:DASH_DATA_GLOB="C:\path\btcusdt\flush_*.parquet"
$env:DASH_ENV_LABEL="labpc"; $env:REDIS_HOST="localhost"; $env:REDIS_PORT="6379"
python run_replication.py; python assemble_replication.py
```
Send back `dash_replication_results.csv`, the `ckpt_labpc/` folder and `progress_labpc.log`.
The cross-environment decision rows (`preregistered_decision`, `cross_environment_direction_maintained`) are filled in automatically once both environments are present.
