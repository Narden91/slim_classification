#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=0-01:30:00
#SBATCH --account=FUPA1_OMDEAFD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --error=results/slurm_logs/custom_exp_job_%A_task_%a.err
#SBATCH --output=results/slurm_logs/custom_exp_job_%A_task_%a.out
#SBATCH --job-name=slim_cust
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=emanuele.nardone@unicas.it

# =============================================================================
# CUSTOM EXPERIMENTS SLURM Script
# =============================================================================

echo "================================================"
echo "CUSTOM EXPERIMENT JOB - SLURM Array"
echo "================================================"

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load python/3.11.7

WORK_DIR="/pitagora_scratch/userexternal/enardone/slim_classification"
cd "$WORK_DIR"

VENV_PYTHON="${VENV_PYTHON:-$WORK_DIR/venv_slim/bin/python}"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: venv python not found or not executable: $VENV_PYTHON"
    exit 1
fi

source "$(dirname "$VENV_PYTHON")/activate"

PYTHON_BIN="$VENV_PYTHON"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Variables exported from bash launcher
PYTHON_SCRIPT="${PYTHON_SCRIPT:-}"
TASK_LIST="${TASK_LIST:-config/experiments_task_list.csv}"
TASK_FROM="${TASK_FROM:-0}"
TASK_TO="${TASK_TO:-0}"
POP_SIZE="${POP_SIZE:-500}"
N_ITER="${N_ITER:-2000}"
SIGMOID_SCALE="${SIGMOID_SCALE:-1}"
FITNESS_FUNCTION="${FITNESS_FUNCTION:-binary_cross_entropy}"

if [ -z "$PYTHON_SCRIPT" ]; then
    echo "ERROR: PYTHON_SCRIPT not set"
    exit 1
fi

if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list file not found: $TASK_LIST"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if ! [[ "$TASK_FROM" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TASK_FROM must be a non-negative integer (got: $TASK_FROM)"
    exit 1
fi

if ! [[ "$TASK_TO" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TASK_TO must be a non-negative integer (got: $TASK_TO)"
    exit 1
fi

if [ "$TASK_FROM" -gt "$TASK_TO" ]; then
    echo "ERROR: TASK_FROM must be <= TASK_TO (got TASK_FROM=$TASK_FROM TASK_TO=$TASK_TO)"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt "$TASK_FROM" ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$TASK_TO" ]; then
    echo "ERROR: Task ID out of range (${TASK_FROM}-${TASK_TO}): $SLURM_ARRAY_TASK_ID"
    exit 1
fi

LINE_COUNT=$(wc -l < "$TASK_LIST")
MAX_TASK_INDEX=$((LINE_COUNT - 2))
if [ "$SLURM_ARRAY_TASK_ID" -gt "$MAX_TASK_INDEX" ]; then
    echo "ERROR: Task ID exceeds task list max index ($MAX_TASK_INDEX): $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ── Parse CSV line (header ignored) ──
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
TASK_CONFIG=$(sed -n "${LINE_NUM}p" "$TASK_LIST")
if [ -z "$TASK_CONFIG" ]; then
    echo "ERROR: Could not read task configuration for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=',' read -r TASK_ID DATASET SLIM_VERSION SEED RUN_NUMBER <<< "$TASK_CONFIG"

# Strip formatting chars
TASK_ID=$(echo "$TASK_ID" | tr -d '\r' | xargs)
DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
SLIM_VERSION=$(echo "$SLIM_VERSION" | tr -d '\r' | xargs)
SEED=$(echo "$SEED" | tr -d '\r' | xargs)
RUN_NUMBER=$(echo "$RUN_NUMBER" | tr -d '\r' | xargs)
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)

# ── Output directories ───────────────────────────────────────────────────────
LOG_DIR="results/experiments/${SCRIPT_NAME}/logs"
mkdir -p "$LOG_DIR"
mkdir -p results/slurm_logs

LOG_FILE="${LOG_DIR}/run_seed_${SEED}.log"
OUT_DIR="results/experiments/${SCRIPT_NAME}"

echo "Script:           $PYTHON_SCRIPT"
echo "Task ID:          $TASK_ID"
echo "Dataset:          $DATASET"
echo "SLIM Version:     $SLIM_VERSION"
echo "Run Number:       $RUN_NUMBER"
echo "Seed / Run ID:    $SEED"
echo "Output Directory: $OUT_DIR"
echo "Python Binary:    $PYTHON_BIN"
echo "Population Size:  $POP_SIZE"
echo "Generations:      $N_ITER"
echo "Sigmoid Scale:    $SIGMOID_SCALE"
echo "Fitness Function: $FITNESS_FUNCTION"
echo "================================================"

echo "Running custom python experiment..."

START_TIME=$(date +%s)

# Using standard custom experiment arguments
"$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --run_id="$SEED" \
    --output_dir="$OUT_DIR" \
    --pop_size="$POP_SIZE" \
    --n_iter="$N_ITER" \
    --sigmoid-scale="$SIGMOID_SCALE" \
    --fitness-function="$FITNESS_FUNCTION" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "OK: Custom Experiment completed successfully"
else
    echo "FAIL: Custom Experiment failed with code $EXIT_CODE"
fi
echo "Duration: ${DURATION} seconds"
echo "================================================"

exit $EXIT_CODE
