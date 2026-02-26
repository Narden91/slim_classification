#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=0-00:30:00
#SBATCH --account=FUPA1_OMDEAFD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --error=logs/slurm/gsgp_flexible_job_%A_task_%a.err
#SBATCH --output=logs/slurm/gsgp_flexible_job_%A_task_%a.out
#SBATCH --job-name=gsgp_flex50
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=emanuele.nardone@unicas.it

echo "================================================"
echo "GSGP FLEXIBLE JOB - SLURM Array Experiment"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "================================================"

module purge
module load python/3.11.7

WORK_DIR="/pitagora_scratch/userexternal/enardone/slim_classification"
cd "$WORK_DIR"

source venv_slim/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

TASK_LIST="${TASK_LIST:-config/task_list_gsgp.csv}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-slim_gsgp/example_binary_classification.py}"
TASK_FROM="${TASK_FROM:-0}"
TASK_TO="${TASK_TO:-0}"

POP_SIZE="${POP_SIZE:-500}"
N_ITER="${N_ITER:-2000}"
SIGMOID_SCALE="${SIGMOID_SCALE:-1}"
FITNESS_FUNCTION="${FITNESS_FUNCTION:-binary_cross_entropy}"
ALGORITHM="${ALGORITHM:-gsgp}"

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

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
TASK_CONFIG=$(sed -n "${LINE_NUM}p" "$TASK_LIST")

if [ -z "$TASK_CONFIG" ]; then
    echo "ERROR: Could not read task configuration for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

IFS=',' read -r TASK_ID DATASET SEED RUN_NUMBER <<< "$TASK_CONFIG"

TASK_ID=$(echo "$TASK_ID" | tr -d '\r' | xargs)
DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
SEED=$(echo "$SEED" | tr -d '\r' | xargs)
RUN_NUMBER=$(echo "$RUN_NUMBER" | tr -d '\r' | xargs)

echo "================================================"
echo "GSGP Experiment Configuration"
echo "================================================"
echo "Task ID:         $TASK_ID"
echo "Selected Range:  ${TASK_FROM}-${TASK_TO}"
echo "Dataset:         $DATASET"
echo "Algorithm:       $ALGORITHM"
echo "Seed:            $SEED"
echo "Run Number:      $RUN_NUMBER"
echo "Population Size: $POP_SIZE"
echo "Iterations:      $N_ITER"
echo "Sigmoid Scale:   $SIGMOID_SCALE"
echo "Fitness Function:$FITNESS_FUNCTION"
echo "================================================"

LOG_DIR="logs_gsgp_flexible/${DATASET}/${ALGORITHM}"
mkdir -p "$LOG_DIR"
mkdir -p logs/slurm

LOG_FILE="${LOG_DIR}/run_${RUN_NUMBER}_seed_${SEED}.log"

echo "Starting GSGP experiment at $(date)"
echo "Output will be saved to: $LOG_FILE"

START_TIME=$(date +%s)

python "$PYTHON_SCRIPT" \
    --dataset="$DATASET" \
    --algorithm="$ALGORITHM" \
    --pop-size="$POP_SIZE" \
    --n-iter="$N_ITER" \
    --sigmoid-scale="$SIGMOID_SCALE" \
    --fitness-function="$FITNESS_FUNCTION" \
    --seed="$SEED" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

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

TIMING_LOG="logs_gsgp_flexible/timing_summary.csv"
{
    flock -x 200
    if [ ! -f "$TIMING_LOG" ]; then
        echo "task_id,dataset,algorithm,seed,duration_seconds,exit_code,node" > "$TIMING_LOG"
    fi
    echo "$TASK_ID,$DATASET,$ALGORITHM,$SEED,$DURATION,$EXIT_CODE,$SLURM_NODELIST" >> "$TIMING_LOG"
} 200>"${TIMING_LOG}.lock"

exit $EXIT_CODE
