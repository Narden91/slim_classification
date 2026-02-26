#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiments.py \
  --experiments "2" \
  --n_runs "${N_RUNS:-20}" \
  --start_run "${START_RUN:-0}" \
  --pop_size "${POP_SIZE:-500}" \
  --n_iter "${N_ITER:-2000}" \
  --sigmoid_scale "${SIGMOID_SCALE:-1}" \
  --fitness_function "${FITNESS_FUNCTION:-binary_cross_entropy}"
