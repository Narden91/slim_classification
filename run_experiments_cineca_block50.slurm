#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=0-00:30:00
#SBATCH --account=FUPA1_OMDEAFD
#SBATCH --partition=dcgp_fua_dbg
#SBATCH --qos=normal
#SBATCH --error=logs/slurm/block50_job_%A_task_%a.err
#SBATCH --output=logs/slurm/block50_job_%A_task_%a.out
#SBATCH --job-name=slim_block50
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=emanuele.nardone@unicas.it

# =============================================================================
# BLOCK-50 SLURM Script - First 250 configurations
# =============================================================================
# Purpose : Run first 250 configurations in blocks of 50 using SLURM arrays
# Usage   : sbatch --array=0-49  run_experiments_cineca_block50.slurm
#           sbatch --array=50-99 run_experiments_cineca_block50.slurm
#           sbatch --array=100-149 run_experiments_cineca_block50.slurm
#           sbatch --array=150-199 run_experiments_cineca_block50.slurm
#           sbatch --array=200-249 run_experiments_cineca_block50.slurm
# =============================================================================

echo "================================================"
echo "BLOCK-50 JOB - SLURM Array Experiment"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

# ── Environment setup ────────────────────────────────────────────────────────
module purge
module load python/3.11.7

WORK_DIR="/pitagora_scratch/userexternal/enardone/slim_classification"
cd "$WORK_DIR"

source venv_slim/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ── Configuration ────────────────────────────────────────────────────────────
TASK_LIST="config/task_list.csv"
PYTHON_SCRIPT="slim_gsgp/example_binary_classification.py"

# Fixed configuration
POP_SIZE=100
N_ITER=1000
MAX_DEPTH="None"
P_INFLATE=0.7
SIGMOID_SCALE=0.01
ALGORITHM="slim"

# ── Task list validation ─────────────────────────────────────────────────────
if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list file not found: $TASK_LIST"
    exit 1
fi

# Guard: only first 250 configurations
if [ "$SLURM_ARRAY_TASK_ID" -lt 0 ] || [ "$SLURM_ARRAY_TASK_ID" -gt 249 ]; then
    echo "ERROR: Task ID out of range (0-249): $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Read configuration for this task (skip header, get line = task_id + 2)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
TASK_CONFIG=$(sed -n "${LINE_NUM}p" "$TASK_LIST")

if [ -z "$TASK_CONFIG" ]; then
    echo "ERROR: Could not read task configuration for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ── Parse CSV line (task_id,dataset,slim_version,seed,run_number) ────────────
IFS=',' read -r TASK_ID DATASET SLIM_VERSION SEED RUN_NUMBER <<< "$TASK_CONFIG"

# Strip carriage returns and whitespace
TASK_ID=$(echo "$TASK_ID" | tr -d '\r' | xargs)
DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
SLIM_VERSION=$(echo "$SLIM_VERSION" | tr -d '\r' | xargs)
SEED=$(echo "$SEED" | tr -d '\r' | xargs)
RUN_NUMBER=$(echo "$RUN_NUMBER" | tr -d '\r' | xargs)

# Sanitize SLIM_VERSION for filesystem (replace * with MUL)
SLIM_VERSION_SAFE="${SLIM_VERSION//\*/MUL}"

echo "================================================"
echo "BLOCK-50 Experiment Configuration"
echo "================================================"
echo "Task ID:         $TASK_ID"
echo "Dataset:         $DATASET"
echo "SLIM Version:    $SLIM_VERSION"
echo "Seed:            $SEED"
echo "Run Number:      $RUN_NUMBER"
echo "Population Size: $POP_SIZE"
echo "Iterations:      $N_ITER"
echo "================================================"

# ── Output directories ───────────────────────────────────────────────────────
LOG_DIR="logs_block50/${DATASET}/${ALGORITHM}/${SLIM_VERSION_SAFE}"
mkdir -p "$LOG_DIR"
mkdir -p logs/slurm

LOG_FILE="${LOG_DIR}/run_${RUN_NUMBER}_seed_${SEED}.log"

echo "Starting BLOCK-50 experiment at $(date)"
echo "Output will be saved to: $LOG_FILE"

# ── Run the experiment ───────────────────────────────────────────────────────
START_TIME=$(date +%s)

python "$PYTHON_SCRIPT" \
    --dataset="$DATASET" \
    --algorithm="$ALGORITHM" \
    --slim-version="$SLIM_VERSION" \
    --pop-size="$POP_SIZE" \
    --n-iter="$N_ITER" \
    --max-depth="$MAX_DEPTH" \
    --p-inflate="$P_INFLATE" \
    --sigmoid-scale="$SIGMOID_SCALE" \
    --seed="$SEED" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

# ── Report results ───────────────────────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "OK: Experiment completed successfully"
    echo "Exit code: $EXIT_CODE"
else
    echo "FAIL: Experiment failed"
    echo "Exit code: $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "Duration: ${DURATION_MIN}m ${DURATION_SEC}s (${DURATION} seconds total)"
echo "End time: $(date)"
echo "================================================"

# Append timing to summary CSV (thread-safe with flock)
TIMING_LOG="logs_block50/timing_summary.csv"
{
    flock -x 200
    if [ ! -f "$TIMING_LOG" ]; then
        echo "task_id,dataset,slim_version,seed,duration_seconds,exit_code,node" > "$TIMING_LOG"
    fi
    echo "$TASK_ID,$DATASET,$SLIM_VERSION,$SEED,$DURATION,$EXIT_CODE,$SLURM_NODELIST" >> "$TIMING_LOG"
} 200>"${TIMING_LOG}.lock"

exit $EXIT_CODE
