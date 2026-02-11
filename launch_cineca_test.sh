#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT="run_experiments_cineca_test.slurm"
TASK_LIST="config/task_list_test.csv"
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

echo "Checking task list and dataset files..."
mapfile -t DATASETS < <(tail -n +2 "$TASK_LIST" | awk -F',' '{gsub(/\r/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')

[ "${#DATASETS[@]}" -gt 0 ] || fail "Task list is empty"

missing=0
for ds in "${DATASETS[@]}"; do
  if [ ! -f "$BENCH_DIR/${ds}.csv" ]; then
    echo "Missing dataset file: $BENCH_DIR/${ds}.csv" >&2
    missing=$((missing + 1))
  fi
  
  # Check duplicates
  if [ "$(printf '%s\n' "${DATASETS[@]}" | grep -Fx "$ds" | wc -l)" -gt 1 ]; then
    echo "Duplicate dataset in task list: $ds" >&2
  fi
  
  # Check dataset token in task list line
  if ! grep -q ",$ds," "$TASK_LIST"; then
    echo "Dataset not found in task list rows: $ds" >&2
  fi
  
  # Avoid empty dataset names
  if [ -z "$ds" ]; then
    echo "Empty dataset name in task list" >&2
  fi
  
  # Keep output dirs ready
  mkdir -p logs/slurm logs_test
  
done

[ "$missing" -eq 0 ] || fail "One or more dataset files are missing"

TASKS=$(( ${#DATASETS[@]} - 1 ))

if ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found in PATH; run on a login node"
fi

echo "Preflight OK. Launching array job 0-$TASKS"
sbatch --array=0-$TASKS "$SLURM_SCRIPT"
