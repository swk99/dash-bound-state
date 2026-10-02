#!/usr/bin/env bash
# Usage (on a fresh Ubuntu 22.04/24.04 VM, files in ~):  bash setup_and_run.sh inst1
# Needs in $HOME: dash_labpc_replication_package.zip, dash-bound-state-updated.zip, btcusdt_1.zip
set -euo pipefail
LABEL="${1:?usage: bash setup_and_run.sh <label e.g. inst1>}"
cd "$HOME"
sudo apt-get update -qq && sudo apt-get install -y -qq redis-server python3-venv unzip >/dev/null
redis-cli ping
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -q scikit-learn==1.6.1 xgboost pandas pyarrow psutil redis scipy joblib
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
[ -d pkg ] || unzip -q dash_labpc_replication_package.zip -d pkg
[ -d dash-bound-state-main ] || unzip -q dash-bound-state-updated.zip
[ -d data ] || unzip -q btcusdt_1.zip -d data
cd ~/pkg
export DASH_REPO="$HOME/dash-bound-state-main" DASH_DATA_GLOB="$HOME/data/btcusdt/flush_*.parquet"
export DASH_ENV_LABEL="$LABEL" REDIS_HOST=localhost REDIS_PORT=6379
{ echo "label=$LABEL"; nproc; lscpu | grep -E 'Model name|^CPU\(s\)'; free -b | sed -n 2p; redis-server --version;
  curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null && echo; } > "env_$LABEL.txt" || true
python run_replication.py
zip -qr "$HOME/result_$LABEL.zip" "ckpt_$LABEL" "progress_$LABEL.log" "preregistered_rules_$LABEL.json" "env_$LABEL.txt"
echo "DONE -> $HOME/result_$LABEL.zip"
