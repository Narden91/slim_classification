#!/usr/bin/env bash
set -euo pipefail

bash launch_custom_experiments.sh \
  --script experiments/exp_1_darwin.py \
  --from "${FROM_TASK:-0}" \
  --count "${COUNT:-20}" \
  --pop-size "${POP_SIZE:-500}" \
  --n-iter "${N_ITER:-2000}" \
  --sigmoid-scale "${SIGMOID_SCALE:-1}" \
  --fitness-function "${FITNESS_FUNCTION:-binary_cross_entropy}"
