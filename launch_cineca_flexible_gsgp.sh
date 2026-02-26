#!/usr/bin/env bash
set -euo pipefail

SLURM_SCRIPT="${SLURM_SCRIPT:-run_experiments_cineca_flexible_gsgp.slurm}"
TASK_LIST="${TASK_LIST:-config/task_list_gsgp.csv}"
BENCH_DIR="${BENCH_DIR:-slim_gsgp/datasets/benchmark}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-slim_gsgp/example_binary_classification.py}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv_slim/bin/activate}"

POP_SIZE="${POP_SIZE:-500}"
N_ITER="${N_ITER:-2000}"
SIGMOID_SCALE="${SIGMOID_SCALE:-1}"
FITNESS_FUNCTION="${FITNESS_FUNCTION:-binary_cross_entropy}"

FROM_TASK=""
TO_TASK=""
COUNT=""
ALL_MODE=0
DRY_RUN=0
SKIP_SANITY=0

SANITY_POP_SIZE="${SANITY_POP_SIZE:-20}"
SANITY_N_ITER="${SANITY_N_ITER:-5}"
SANITY_SIGMOID_SCALE="${SANITY_SIGMOID_SCALE:-0.01}"

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage:
  $0 --all [--dry-run] [--pop-size 500 --n-iter 2000 --sigmoid-scale 1 --fitness-function binary_cross_entropy]
  $0 --from N --to M [--dry-run] [--pop-size 500 --n-iter 2000 --sigmoid-scale 1 --fitness-function binary_cross_entropy]
  $0 --from N --count K [--dry-run] [--pop-size 500 --n-iter 2000 --sigmoid-scale 1 --fitness-function binary_cross_entropy]
  $0 --count K [--dry-run] [--pop-size 500 --n-iter 2000 --sigmoid-scale 1 --fitness-function binary_cross_entropy]

Selection modes:
  --all            Launch all task rows in task_list_gsgp.csv
  --from/--to      Launch inclusive task-id interval [N, M]
  --count          Launch K tasks starting from --from (default: 0)

Batching:
  Tasks are submitted in batches of 50 via multiple sbatch array calls.
  If total tasks is not a multiple of 50, the final batch is partial.

Dry-run:
  --dry-run        Validate everything and print sbatch commands without submitting.

Optional runtime params:
  --pop-size         Population size (default: 500)
  --n-iter           Number of generations (default: 2000)
  --sigmoid-scale    Sigmoid scale (default: 1)
  --fitness-function Fitness function (default: binary_cross_entropy)

Sanity check:
  A short local GSGP run is executed automatically before sbatch submission.
  Use --skip-sanity to disable it.

Examples:
  $0 --all
  $0 --from 0 --to 249
  $0 --count 120 --dry-run
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-sanity)
      SKIP_SANITY=1
      shift
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

if [ "$DRY_RUN" -eq 0 ] && ! command -v sbatch >/dev/null 2>&1; then
  fail "sbatch not found in PATH; run on a login node or use --dry-run"
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

echo "Preflight OK (GSGP)"
echo "Task interval: $FROM_TASK-$TO_TASK"
echo "Total tasks:   $SELECTED_COUNT"
echo "Pop size:      $POP_SIZE"
echo "N iter:        $N_ITER"
echo "Sigmoid scale: $SIGMOID_SCALE"
echo "Fitness fn:    $FITNESS_FUNCTION"
echo "Batch size:    50"
echo "Batches:       $TOTAL_BATCHES (full=$FULL_BATCHES, remainder=$REMAINDER)"
echo "Dry-run:       $DRY_RUN"

if [ "$DRY_RUN" -eq 0 ] && [ "$SKIP_SANITY" -eq 0 ]; then
  [[ "$SANITY_POP_SIZE" =~ ^[0-9]+$ ]] || fail "SANITY_POP_SIZE must be a non-negative integer"
  [[ "$SANITY_N_ITER" =~ ^[0-9]+$ ]] || fail "SANITY_N_ITER must be a non-negative integer"
  [ "$SANITY_POP_SIZE" -gt 0 ] || fail "SANITY_POP_SIZE must be greater than 0"
  [ "$SANITY_N_ITER" -gt 0 ] || fail "SANITY_N_ITER must be greater than 0"

  FIRST_LINE_NUM=$((FROM_TASK + 2))
  FIRST_TASK_CONFIG=$(sed -n "${FIRST_LINE_NUM}p" "$TASK_LIST")
  [ -n "$FIRST_TASK_CONFIG" ] || fail "Could not read first selected task row from $TASK_LIST"

  IFS=',' read -r _TASK_ID SANITY_DATASET SANITY_SEED _RUN_NUMBER <<< "$FIRST_TASK_CONFIG"
  SANITY_DATASET=$(echo "$SANITY_DATASET" | tr -d '\r' | xargs)
  SANITY_SEED=$(echo "$SANITY_SEED" | tr -d '\r' | xargs)

  [ -n "$SANITY_DATASET" ] || fail "Sanity-check dataset is empty in selected task row"
  [[ "$SANITY_SEED" =~ ^-?[0-9]+$ ]] || fail "Sanity-check seed is not a valid integer: $SANITY_SEED"

  echo "Running local sanity check before submission..."
  echo "Sanity config: dataset=$SANITY_DATASET seed=$SANITY_SEED pop_size=$SANITY_POP_SIZE n_iter=$SANITY_N_ITER"

  source "$VENV_ACTIVATE" || fail "Failed to activate virtual environment: $VENV_ACTIVATE"

  python "$PYTHON_SCRIPT" \
    --dataset="$SANITY_DATASET" \
    --algorithm="gsgp" \
    --pop-size="$SANITY_POP_SIZE" \
    --n-iter="$SANITY_N_ITER" \
    --sigmoid-scale="$SANITY_SIGMOID_SCALE" \
    --fitness-function="$FITNESS_FUNCTION" \
    --seed="$SANITY_SEED" \
    --device="cpu" \
    --verbose="0" \
    >/dev/null

  echo "Sanity check passed."
elif [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry-run mode: skipping local sanity check and sbatch submission."
elif [ "$SKIP_SANITY" -eq 1 ]; then
  echo "Sanity check skipped by user (--skip-sanity)."
fi

for ((start=FROM_TASK; start<=TO_TASK; start+=50)); do
  end=$((start + 49))
  if [ "$end" -gt "$TO_TASK" ]; then
    end="$TO_TASK"
  fi

  cmd=(
    sbatch
    --export=ALL,TASK_FROM="$FROM_TASK",TASK_TO="$TO_TASK",TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT",POP_SIZE="$POP_SIZE",N_ITER="$N_ITER",SIGMOID_SCALE="$SIGMOID_SCALE",FITNESS_FUNCTION="$FITNESS_FUNCTION"
    --array="${start}-${end}"
    "$SLURM_SCRIPT"
  )

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] ${cmd[*]}"
  else
    echo "Submitting array block: ${start}-${end}"
    "${cmd[@]}"
  fi
done
