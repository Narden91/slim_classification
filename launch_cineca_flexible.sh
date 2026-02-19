#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT="${SLURM_SCRIPT:-run_experiments_cineca_flexible.slurm}"
TASK_LIST="${TASK_LIST:-config/task_list.csv}"
BENCH_DIR="${BENCH_DIR:-slim_gsgp/datasets/benchmark}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-slim_gsgp/example_binary_classification.py}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv_slim/bin/activate}"

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
  --all            Launch all task rows in task_list.csv
  --from/--to      Launch inclusive task-id interval [N, M]
  --count          Launch K tasks starting from --from (default: 0)

Batching:
  Tasks are submitted in batches of 50 via multiple sbatch array calls.
  If total tasks is not a multiple of 50, the final batch is partial.

Examples:
  $0 --all
  $0 --from 0 --to 249
  $0 --from 250 --count 1023
  $0 --count 1023
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
[ -d "$BENCH_DIR" ] || fail "Missing benchmark dataset folder: $BENCH_DIR"
[ -f "$PYTHON_SCRIPT" ] || fail "Missing python entrypoint: $PYTHON_SCRIPT"
[ -f "$VENV_ACTIVATE" ] || fail "Missing venv activate script: $VENV_ACTIVATE"

if ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found in PATH; run on a login node"
fi

bash -n "$SLURM_SCRIPT" || fail "SLURM script has syntax errors"

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

# Validate datasets used by selected rows (unique dataset names only)
mapfile -t DATASETS < <(
  awk -F',' -v from="$FROM_TASK" -v to="$TO_TASK" '
    NR >= from + 2 && NR <= to + 2 {
      gsub(/\r/, "", $2)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
      if ($2 != "") print $2
    }
  ' "$TASK_LIST" | sort -u
)

missing=0
for ds in "${DATASETS[@]}"; do
  if [ ! -f "$BENCH_DIR/${ds}.csv" ]; then
    echo "Missing dataset file: $BENCH_DIR/${ds}.csv" >&2
    missing=$((missing + 1))
  fi
done

[ "$missing" -eq 0 ] || fail "One or more dataset files are missing"

echo "Preflight OK"
echo "Task interval: $FROM_TASK-$TO_TASK"
echo "Total tasks:   $SELECTED_COUNT"
echo "Batch size:    50"
echo "Batches:       $TOTAL_BATCHES (full=$FULL_BATCHES, remainder=$REMAINDER)"

for ((start=FROM_TASK; start<=TO_TASK; start+=50)); do
  end=$((start + 49))
  if [ "$end" -gt "$TO_TASK" ]; then
    end="$TO_TASK"
  fi

  echo "Submitting array block: ${start}-${end}"
  sbatch \
    --export=ALL,TASK_FROM="$FROM_TASK",TASK_TO="$TO_TASK",TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT" \
    --array="${start}-${end}" \
    "$SLURM_SCRIPT"
done
