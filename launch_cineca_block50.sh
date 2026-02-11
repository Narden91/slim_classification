#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT="run_experiments_cineca_block50.slurm"
TASK_LIST="config/task_list.csv"
BENCH_DIR="slim_gsgp/datasets/benchmark"
PYTHON_SCRIPT="slim_gsgp/example_binary_classification.py"
VENV_ACTIVATE="venv_slim/bin/activate"

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

[ -f "$SLURM_SCRIPT" ] || fail "Missing SLURM script: $SLURM_SCRIPT"
[ -f "$TASK_LIST" ] || fail "Missing task list: $TASK_LIST"
[ -d "$BENCH_DIR" ] || fail "Missing benchmark dataset folder: $BENCH_DIR"
[ -f "$PYTHON_SCRIPT" ] || fail "Missing python entrypoint: $PYTHON_SCRIPT"
[ -f "$VENV_ACTIVATE" ] || fail "Missing venv activate script: $VENV_ACTIVATE"

# Basic SLURM script sanity check
bash -n "$SLURM_SCRIPT" || fail "SLURM script has syntax errors"

# Ensure at least 250 tasks (header + 250 rows)
LINE_COUNT=$(wc -l < "$TASK_LIST")
[ "$LINE_COUNT" -ge 251 ] || fail "Task list must have at least 250 rows (found $((LINE_COUNT - 1)))"

# Validate first 250 datasets exist
mapfile -t DATASETS < <(tail -n +2 "$TASK_LIST" | head -n 250 | awk -F',' '{gsub(/\r/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')

[ "${#DATASETS[@]}" -eq 250 ] || fail "Expected 250 datasets, found ${#DATASETS[@]}"

missing=0
for ds in "${DATASETS[@]}"; do
  if [ -z "$ds" ]; then
    echo "Empty dataset name in task list" >&2
    missing=$((missing + 1))
    continue
  fi

  if [ ! -f "$BENCH_DIR/${ds}.csv" ]; then
    echo "Missing dataset file: $BENCH_DIR/${ds}.csv" >&2
    missing=$((missing + 1))
  fi

done

[ "$missing" -eq 0 ] || fail "One or more dataset files are missing"

if ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found in PATH; run on a login node"
fi

echo "Preflight OK. Launching 5 blocks of 50 tasks (0-249)"

sbatch --array=0-49 "$SLURM_SCRIPT"
sbatch --array=50-99 "$SLURM_SCRIPT"
sbatch --array=100-149 "$SLURM_SCRIPT"
sbatch --array=150-199 "$SLURM_SCRIPT"
sbatch --array=200-249 "$SLURM_SCRIPT"
