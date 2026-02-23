#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CUSTOM EXPERIMENTS LAUNCHER
# =============================================================================

SLURM_SCRIPT="run_custom_experiments.slurm"
TASK_LIST="config/experiments_task_list.csv"

usage() {
  cat <<EOF
Usage:
  $0 --script <path_to_experiment_script.py> [--count N]

Example: 
  $0 --script experiments/exp_1_darwin.py --count 30

Description:
  Launches a SLURM array job for the custom publication experiments located 
  in the experiments/ folder. By default, reads 30 seeds from the task list.
EOF
  exit 1
}

SCRIPT_PATH=""
COUNT=30

while [ $# -gt 0 ]; do
  case "$1" in
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    --count)
      COUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

[ -z "$SCRIPT_PATH" ] && usage
[ -f "$SCRIPT_PATH" ] || { echo "ERROR: Script not found: $SCRIPT_PATH"; exit 1; }
[ -f "$SLURM_SCRIPT" ] || { echo "ERROR: SLURM script not found: $SLURM_SCRIPT"; exit 1; }
[ -f "$TASK_LIST" ] || { echo "ERROR: Task list not found: $TASK_LIST"; exit 1; }

MAX_TASK=$((COUNT - 1))

echo "Submitting array for $COUNT tasks (0 to $MAX_TASK)..."
echo "Target Python Script: $SCRIPT_PATH"

# Dispatch to SLURM seamlessly
sbatch \
  --export=ALL,PYTHON_SCRIPT="$SCRIPT_PATH",TASK_LIST="$TASK_LIST" \
  --array="0-${MAX_TASK}" \
  "$SLURM_SCRIPT"

echo "Done."
