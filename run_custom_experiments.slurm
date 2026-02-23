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

if [ -d "venv_slim" ]; then
    source venv_slim/bin/activate
else
    echo "WARNING: venv_slim not found, assuming python is in path"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Variables exported from bash launcher
PYTHON_SCRIPT="${PYTHON_SCRIPT}"
TASK_LIST="${TASK_LIST:-config/experiments_task_list.csv}"

if [ ! -f "$TASK_LIST" ] || [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Missing task list or python script."
    exit 1
fi

# ── Parse CSV line (header ignored) ──
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 2))
TASK_CONFIG=$(sed -n "${LINE_NUM}p" "$TASK_LIST")
IFS=',' read -r TASK_ID DATASET SLIM_VERSION SEED RUN_NUMBER <<< "$TASK_CONFIG"

# Strip formatting chars
SEED=$(echo "$SEED" | tr -d '\r' | xargs)
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)

# ── Output directories ───────────────────────────────────────────────────────
LOG_DIR="results/experiments/${SCRIPT_NAME}/logs"
mkdir -p "$LOG_DIR"
mkdir -p results/slurm_logs

LOG_FILE="${LOG_DIR}/run_seed_${SEED}.log"
OUT_DIR="results/experiments/${SCRIPT_NAME}"

echo "Script:           $PYTHON_SCRIPT"
echo "Seed / Run ID:    $SEED"
echo "Output Directory: $OUT_DIR"
echo "================================================"

echo "Running custom python experiment..."

START_TIME=$(date +%s)

# Using standard custom experiment arguments
python "$PYTHON_SCRIPT" \
    --run_id="$SEED" \
    --output_dir="$OUT_DIR" \
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
