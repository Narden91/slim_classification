#!/usr/bin/env bash
set -euo pipefail

bash launch_custom_experiments.sh \
  --script experiments/exp_3_ablation.py \
  --from "${FROM_TASK:-0}" \
  --count "${COUNT:-30}"
