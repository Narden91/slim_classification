#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=0-01:30:00
#SBATCH --account=FUPA1_OMDEAFD
#SBATCH --partition=dcgp_fua_prod
#SBATCH --qos=normal
#SBATCH --error=results/slurm_logs/explainability_job_%A_task_%a.err
#SBATCH --output=results/slurm_logs/explainability_job_%A_task_%a.out
#SBATCH --job-name=slim_expl
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=emanuele.nardone@unicas.it

# =============================================================================
# EXPLAINABILITY EXPERIMENT SLURM Script
# =============================================================================

echo "================================================"
echo "EXPLAINABILITY JOB - SLURM Array Experiment"
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
TASK_LIST_DEFAULT="config/task_list_explainability.csv"
PYTHON_SCRIPT_DEFAULT="scripts/run_explainability_exp.py"

# By default, ignore inherited TASK_LIST/PYTHON_SCRIPT from shell/session to prevent
# accidental cross-experiment execution. Set ALLOW_ENV_OVERRIDE=1 to opt in.
ALLOW_ENV_OVERRIDE="${ALLOW_ENV_OVERRIDE:-0}"
if [ "$ALLOW_ENV_OVERRIDE" = "1" ]; then
    TASK_LIST="${TASK_LIST:-$TASK_LIST_DEFAULT}"
    PYTHON_SCRIPT="${PYTHON_SCRIPT:-$PYTHON_SCRIPT_DEFAULT}"
else
    TASK_LIST="$TASK_LIST_DEFAULT"
    PYTHON_SCRIPT="$PYTHON_SCRIPT_DEFAULT"
fi

EXPORT_FORMATS="${EXPORT_FORMATS:-html,svg,text}"
TASK_FROM="${TASK_FROM:-0}"
TASK_TO="${TASK_TO:-0}"

# Fixed parameters
POP_SIZE=100
N_ITER=2000
P_INFLATE=0.7

# ── Task list validation ─────────────────────────────────────────────────────
if [ ! -f "$TASK_LIST" ]; then
    echo "ERROR: Task list file not found: $TASK_LIST"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
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
echo "Resolved Runtime Configuration"
echo "================================================"
echo "ALLOW_ENV_OVERRIDE: $ALLOW_ENV_OVERRIDE"
echo "TASK_LIST:          $TASK_LIST"
echo "PYTHON_SCRIPT:      $PYTHON_SCRIPT"
echo "EXPORT_FORMATS:     $EXPORT_FORMATS"
echo "================================================"

# Best-effort dependency preflight (non-blocking)
if [[ ",${EXPORT_FORMATS}," == *",html,"* ]] || [[ ",${EXPORT_FORMATS}," == *",svg,"* ]] || [[ ",${EXPORT_FORMATS}," == *",pdf,"* ]]; then
    python - <<'PY'
import importlib.util
import shutil

has_plotly = importlib.util.find_spec("plotly") is not None
has_kaleido = importlib.util.find_spec("kaleido") is not None
has_chrome = any(shutil.which(cmd) for cmd in ("google-chrome", "chrome", "chromium", "chromium-browser"))

print("Dependency preflight:")
print(f" - plotly:  {'OK' if has_plotly else 'MISSING'}")
print(f" - kaleido: {'OK' if has_kaleido else 'MISSING'}")
print(f" - chrome:  {'OK' if has_chrome else 'MISSING'}")
PY
    echo "================================================"
fi

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
# Logs are automatically managed by SLURM and saved to results/slurm_logs/
mkdir -p results/slurm_logs

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
    --export_formats="$EXPORT_FORMATS"

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
