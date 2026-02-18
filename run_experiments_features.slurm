#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=0-00:30:00
#SBATCH --account=FUPA1_OMDEAFD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --error=logs/slurm/feature_imp_job_%A_task_%a.err
#SBATCH --output=logs/slurm/feature_imp_job_%A_task_%a.out
#SBATCH --job-name=slim_feat
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=emanuele.nardone@unicas.it

# =============================================================================
# FEATURE IMPORTANCE EXPERIMENT SLURM Script
# =============================================================================

echo "================================================"
echo "FEATURE IMPORTANCE JOB - SLURM Array Experiment"
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

if [ -d "venv_slim" ]; then
    source venv_slim/bin/activate
else
    echo "WARNING: venv_slim not found, assuming python is in path"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# ── Configuration ────────────────────────────────────────────────────────────
TASK_LIST="${TASK_LIST:-config/task_list_features.csv}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-scripts/run_feature_importance_exp.py}"
TASK_FROM="${TASK_FROM:-0}"
TASK_TO="${TASK_TO:-0}"

# Fixed parameters
POP_SIZE=100
N_ITER=2000
P_INFLATE=0.7
ALGORITHM="slim_feature_imp"

# ── Task list validation ─────────────────────────────────────────────────────
if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list file not found: $TASK_LIST"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -lt "$TASK_FROM" ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$TASK_TO" ]; then
    echo "ERROR: Task ID out of range (${TASK_FROM}-${TASK_TO}): $SLURM_ARRAY_TASK_ID"
    # exit 1  # Standard array jobs manipulate TASK_ID, so this check depends on usage
fi

# Read configuration line (skip header, line = task_id + 2)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
TASK_CONFIG=$(sed -n "${LINE_NUM}p" "$TASK_LIST")

if [ -z "$TASK_CONFIG" ]; then
    echo "ERROR: Could not read task configuration for task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ── Parse CSV line (task_id,dataset,slim_version,seed,run_number,max_depth) ──
IFS=',' read -r TASK_ID DATASET SLIM_VERSION SEED RUN_NUMBER MAX_DEPTH <<< "$TASK_CONFIG"

# Strip formatting chars
TASK_ID=$(echo "$TASK_ID" | tr -d '\r' | xargs)
DATASET=$(echo "$DATASET" | tr -d '\r' | xargs)
SLIM_VERSION=$(echo "$SLIM_VERSION" | tr -d '\r' | xargs)
SEED=$(echo "$SEED" | tr -d '\r' | xargs)
RUN_NUMBER=$(echo "$RUN_NUMBER" | tr -d '\r' | xargs)
MAX_DEPTH=$(echo "$MAX_DEPTH" | tr -d '\r' | xargs)

# Sanitize
SLIM_VERSION_SAFE="${SLIM_VERSION//\*/MUL}"

echo "================================================"
echo "Task Configuration"
echo "================================================"
echo "Task ID:         $TASK_ID"
echo "Dataset:         $DATASET"
echo "SLIM Version:    $SLIM_VERSION"
echo "Seed:            $SEED"
echo "Run Number:      $RUN_NUMBER"
echo "Max Depth:       $MAX_DEPTH"
echo "================================================"

# ── Output directories ───────────────────────────────────────────────────────
LOG_DIR="logs_feature_imp/${DATASET}/${SLIM_VERSION_SAFE}/depth_${MAX_DEPTH}"
mkdir -p "$LOG_DIR"
mkdir -p logs/slurm

LOG_FILE="${LOG_DIR}/run_${RUN_NUMBER}_seed_${SEED}.log"

echo "Running python script..."

START_TIME=$(date +%s)

python "$PYTHON_SCRIPT" \
    --dataset="$DATASET" \
    --max_depth="$MAX_DEPTH" \
    --seed="$SEED" \
    --slim_version="$SLIM_VERSION" \
    --pop_size="$POP_SIZE" \
    --n_iter="$N_ITER" \
    --p_inflate="$P_INFLATE" \
    --log_dir="logs_feature_imp" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "OK: Experiment completed successfully"
else
    echo "FAIL: Experiment failed with code $EXIT_CODE"
fi
echo "Duration: ${DURATION} seconds"
echo "================================================"

exit $EXIT_CODE
