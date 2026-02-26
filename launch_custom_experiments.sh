#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CUSTOM EXPERIMENTS LAUNCHER
# =============================================================================

SLURM_SCRIPT="run_custom_experiments.slurm"
TASK_LIST="config/experiments_task_list.csv"
PYTHON_SCRIPT=""

FROM_TASK=""
TO_TASK=""
COUNT=""
ALL_MODE=0
POP_SIZE="${POP_SIZE:-500}"
N_ITER="${N_ITER:-2000}"
SIGMOID_SCALE="${SIGMOID_SCALE:-1}"
FITNESS_FUNCTION="${FITNESS_FUNCTION:-binary_cross_entropy}"

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  $0 --script experiments/exp_1_darwin.py --count 20
  $0 --script experiments/exp_2_hand_stat.py --all
  $0 --script experiments/exp_3_ablation.py --from 0 --to 29

Selection modes:
  --all            Launch all task rows in experiments_task_list.csv
  --from/--to      Launch inclusive task-id interval [N, M]
  --count          Launch K tasks starting from --from (default: 0)

Defaults:
  If no selection mode is provided, this launcher runs --count 20 from --from 0.

Execution parameters:
  --pop-size         Population size (default: 500)
  --n-iter           Number of generations (default: 2000)
  --sigmoid-scale    Sigmoid scaling factor (default: 1)
  --fitness-function Fitness function (default: binary_cross_entropy)

Batching:
  Tasks are submitted in batches of 50 via multiple sbatch array calls.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --script)
      [ $# -ge 2 ] || fail "Missing value for --script"
      PYTHON_SCRIPT="$2"
      shift 2
      ;;
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
    --pop-size)
      [ $# -ge 2 ] || fail "Missing value for --pop-size"
      POP_SIZE="$2"
      shift 2
      ;;
    --n-iter)
      [ $# -ge 2 ] || fail "Missing value for --n-iter"
      N_ITER="$2"
      shift 2
      ;;
    --sigmoid-scale)
      [ $# -ge 2 ] || fail "Missing value for --sigmoid-scale"
      SIGMOID_SCALE="$2"
      shift 2
      ;;
    --fitness-function)
      [ $# -ge 2 ] || fail "Missing value for --fitness-function"
      FITNESS_FUNCTION="$2"
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

[ -n "$PYTHON_SCRIPT" ] || fail "--script is required"
[ -f "$PYTHON_SCRIPT" ] || fail "Script not found: $PYTHON_SCRIPT"
[ -f "$SLURM_SCRIPT" ] || fail "SLURM script not found: $SLURM_SCRIPT"
[ -f "$TASK_LIST" ] || fail "Task list not found: $TASK_LIST"

if ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found in PATH; run on a login node"
fi

bash -n "$SLURM_SCRIPT" || fail "SLURM script has syntax errors"

LINE_COUNT=$(wc -l < "$TASK_LIST")
[ "$LINE_COUNT" -ge 2 ] || fail "Task list contains no task rows"
MAX_TASK_INDEX=$((LINE_COUNT - 2))

# Default behavior: 20 runs
if [ "$ALL_MODE" -eq 0 ] && [ -z "$FROM_TASK" ] && [ -z "$TO_TASK" ] && [ -z "$COUNT" ]; then
  FROM_TASK=0
  COUNT=20
fi

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
    [ -n "$FROM_TASK" ] || fail "Specify one mode: --all OR --count [--from N] OR --from N --to M"
    [ -n "$TO_TASK" ] || fail "When using --from without --count, --to is required"
    [[ "$FROM_TASK" =~ ^[0-9]+$ ]] || fail "--from must be a non-negative integer"
    [[ "$TO_TASK" =~ ^[0-9]+$ ]] || fail "--to must be a non-negative integer"
  fi
fi

[ "$FROM_TASK" -le "$TO_TASK" ] || fail "Invalid interval: --from must be <= --to"
[ "$TO_TASK" -le "$MAX_TASK_INDEX" ] || fail "Requested to-task ($TO_TASK) exceeds available max task index ($MAX_TASK_INDEX)"

SELECTED_COUNT=$((TO_TASK - FROM_TASK + 1))
FULL_BATCHES=$((SELECTED_COUNT / 50))
REMAINDER=$((SELECTED_COUNT % 50))
TOTAL_BATCHES=$((FULL_BATCHES + (REMAINDER > 0 ? 1 : 0)))

echo "Preflight OK"
echo "Script:        $PYTHON_SCRIPT"
echo "Task interval: $FROM_TASK-$TO_TASK"
echo "Total tasks:   $SELECTED_COUNT"
echo "Pop size:      $POP_SIZE"
echo "N iter:        $N_ITER"
echo "Sigmoid scale: $SIGMOID_SCALE"
echo "Fitness fn:    $FITNESS_FUNCTION"
echo "Batch size:    50"
echo "Batches:       $TOTAL_BATCHES (full=$FULL_BATCHES, remainder=$REMAINDER)"

for ((start=FROM_TASK; start<=TO_TASK; start+=50)); do
  end=$((start + 49))
  if [ "$end" -gt "$TO_TASK" ]; then
    end="$TO_TASK"
  fi

  echo "Submitting array block: ${start}-${end}"
  sbatch \
    --export=ALL,TASK_FROM="$FROM_TASK",TASK_TO="$TO_TASK",TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT",POP_SIZE="$POP_SIZE",N_ITER="$N_ITER",SIGMOID_SCALE="$SIGMOID_SCALE",FITNESS_FUNCTION="$FITNESS_FUNCTION" \
    --array="${start}-${end}" \
    "$SLURM_SCRIPT"
done

echo "Submission complete."
