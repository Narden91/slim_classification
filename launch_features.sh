#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# FEATURE IMPORTANCE EXPERIMENT LAUNCHER
# =============================================================================

SLURM_SCRIPT="run_experiments_features.slurm"
TASK_LIST="config/task_list_features.csv"
PYTHON_SCRIPT="scripts/run_feature_importance_exp.py"
# We don't strictly check for benchmark CSVs here because the python script 
# handles loading via data_loader fallback for missing files.

FROM_TASK=""
TO_TASK=""
COUNT=""
ALL_MODE=0

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  $0 --all
  $0 --from N --to M
  $0 --from N --count K
  $0 --count K

Selection modes:
  --all            Launch all task rows in task_list_features.csv
  --from/--to      Launch inclusive task-id interval [N, M]
  --count          Launch K tasks starting from --from (default: 0)

Batching:
  Tasks are submitted in batches of 50 via multiple sbatch array calls.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --all)
      ALL_MODE=1
      shift
      ;;
    --from)
      [ $# -ge 2 ] || fail "Missing value for --from"
      FROM_TASK="$2"
      shift 2
      ;;
    --to)
      [ $# -ge 2 ] || fail "Missing value for --to"
      TO_TASK="$2"
      shift 2
      ;;
    --count)
      [ $# -ge 2 ] || fail "Missing value for --count"
      COUNT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      ;;
  esac
done

[ -f "$SLURM_SCRIPT" ] || fail "Missing SLURM script: $SLURM_SCRIPT"
[ -f "$TASK_LIST" ] || fail "Missing task list: $TASK_LIST"
[ -f "$PYTHON_SCRIPT" ] || fail "Missing python entrypoint: $PYTHON_SCRIPT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "WARNING: sbatch not found. This script is intended to run on a SLURM cluster login node."
  echo "Continuing check..."
fi

LINE_COUNT=$(wc -l < "$TASK_LIST")
[ "$LINE_COUNT" -ge 2 ] || fail "Task list contains no task rows"
MAX_TASK_INDEX=$((LINE_COUNT - 2))

if [ "$ALL_MODE" -eq 1 ]; then
  [ -z "$FROM_TASK" ] || fail "--all cannot be combined with --from"
  [ -z "$TO_TASK" ] || fail "--all cannot be combined with --to"
  [ -z "$COUNT" ] || fail "--all cannot be combined with --count"
  FROM_TASK=0
  TO_TASK="$MAX_TASK_INDEX"
else
  if [ -n "$COUNT" ]; then
    [[ "$COUNT" =~ ^[0-9]+$ ]] || fail "--count must be a non-negative integer"
    [ "$COUNT" -gt 0 ] || fail "--count must be greater than 0"

    if [ -z "$FROM_TASK" ]; then
      FROM_TASK=0
    fi
    [[ "$FROM_TASK" =~ ^[0-9]+$ ]] || fail "--from must be a non-negative integer"
    [ -z "$TO_TASK" ] || fail "Use either --to or --count, not both"

    TO_TASK=$((FROM_TASK + COUNT - 1))
  else
    # Logic for --from/--to or just --from
    if [ -n "$FROM_TASK" ] && [ -z "$TO_TASK" ]; then
         fail "When using --from without --count, --to is required"
    fi
    # If nothing specified
    if [ -z "$FROM_TASK" ] && [ -z "$TO_TASK" ]; then
         fail "Specify one mode: --all OR --count [--from N] OR --from N --to M"
    fi
    
    [[ "$FROM_TASK" =~ ^[0-9]+$ ]] || fail "--from must be a non-negative integer"
    [[ "$TO_TASK" =~ ^[0-9]+$ ]] || fail "--to must be a non-negative integer"
  fi
fi

[ "$FROM_TASK" -le "$TO_TASK" ] || fail "Invalid interval: --from must be <= --to"
[ "$TO_TASK" -le "$MAX_TASK_INDEX" ] || fail "Requested to-task ($TO_TASK) exceeds available max task index ($MAX_TASK_INDEX)"

SELECTED_COUNT=$((TO_TASK - FROM_TASK + 1))

echo "Preflight OK"
echo "Task List:   $TASK_LIST"
echo "Task interval: $FROM_TASK-$TO_TASK"
echo "Total tasks:   $SELECTED_COUNT"

# Submitting in batches
BATCH_SIZE=50

for ((start=FROM_TASK; start<=TO_TASK; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1))
  if [ "$end" -gt "$TO_TASK" ]; then
    end="$TO_TASK"
  fi

  echo "Submitting array block: ${start}-${end}"
  
  # We export variables needed by the SLURM script
  sbatch \
    --export=ALL,TASK_FROM="$FROM_TASK",TASK_TO="$TO_TASK",TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT" \
    --array="${start}-${end}" \
    "$SLURM_SCRIPT"
done
