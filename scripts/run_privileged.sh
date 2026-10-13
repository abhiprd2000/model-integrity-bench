#!/usr/bin/env bash
# Runs the three root-only experiments sequentially (never concurrently: they
# time disk I/O). Log: .cache/privileged_run.log. Stops at the first failure.
#   cd ~/Desktop/model-integrity-bench && bash scripts/run_privileged.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=.cache/privileged_run.log
sudo -v                                  # prompts for the password once
( while true; do sudo -n true; sleep 50; done ) 2>/dev/null &   # keep ticket alive
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
{
  echo "== $(date -Is) install fsverity"
  command -v fsverity >/dev/null || sudo apt install -y fsverity
  fsverity --version
  echo "== $(date -Is) r_load_cold"
  sudo taskset -c 2 ./target/release/r_load_cold
  echo "== $(date -Is) loader_bench_cold"
  sudo taskset -c 2 .venv/bin/python scripts/loader_bench_cold.py
  echo "== $(date -Is) fsverity_bench"
  sudo .venv/bin/python scripts/fsverity_bench.py
  echo "== $(date -Is) ALL DONE"
} 2>&1 | tee "$LOG"
