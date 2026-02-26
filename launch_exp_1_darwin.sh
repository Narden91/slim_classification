#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiments.py \
  --experiments "1" \
  --n_runs "${N_RUNS:-30}" \
  --start_run "${START_RUN:-0}" \
  --pop_size "${POP_SIZE:-100}" \
  --n_iter "${N_ITER:-100}"
