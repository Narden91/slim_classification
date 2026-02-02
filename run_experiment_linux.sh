#!/bin/bash
# Script to run multiple binary classification experiments with different seeds
# Designed for Linux environment using the local venv

# Use the Python from the virtual environment
PYTHON_PATH="./venv/bin/python"

# Check if venv exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Virtual environment not found at $PYTHON_PATH"
    echo "Please ensure you're in the project directory and venv is activated"
    exit 1
fi

# Set the full path to the script
SCRIPT_PATH="slim_gsgp/example_binary_classification.py"

# Set the number of runs per dataset
NUM_RUNS=1

# Set the base seed (each run will use base_seed + run_index)
BASE_SEED=42

#Blood 
#Clima 
#Eeg 
#Fertility 
#Gina 
#Hill 
#Ilpd 
#Kc 
#Liver 
#Musk 
#Ozone 
#Pc1 
#Pc3 
#Qsar 
#Retinopathy 
#Scene 
#Spam 
#Spect 

# Define the datasets to process
DATASETS=("gina") # ("gina" "eeg" "scene" "fertility" "liver" "ozone")

ALGORITHM="slim"
SLIM_VERSION=("SLIM+SIG2") # ("SLIM+SIG2" "SLIM*SIG2" "SLIM+ABS" "SLIM*ABS" "SLIM+SIG1" "SLIM*SIG1")

# Set other parameters
POP_SIZE=500
N_ITER=2000
MAX_DEPTH="None"
P_INFLATE=0.7
SIGMOID_SCALE=0.01

# Create logs directory if it doesn't exist
mkdir -p logs

# Print Python version and path to help with debugging
echo "============== SYSTEM INFO ==============="
echo "Python path: $PYTHON_PATH"
$PYTHON_PATH --version
echo "Script path: $SCRIPT_PATH"
echo "Working directory: $(pwd)"
echo "==========================================="

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"

    # Create dataset-specific directories
    mkdir -p logs/$DATASET
    mkdir -p logs/$DATASET/$ALGORITHM

    echo "Starting $NUM_RUNS experiments on $DATASET using $ALGORITHM algorithm..."

    # Run multiple experiments with different seeds
    for ((i=0; i<$NUM_RUNS; i++)); do
        # Calculate the seed for this run
        SEED=$((BASE_SEED + i))

        echo "Starting run $((i+1))/$NUM_RUNS with seed $SEED..."

        # Create a temporary file for capturing the output
        LOG_FILE="logs/$DATASET/$ALGORITHM/run_$((i+1))_seed_${SEED}.log"

        # Run the Python script with the specified parameters
        $PYTHON_PATH $SCRIPT_PATH \
            --dataset=$DATASET \
            --algorithm=$ALGORITHM \
            --slim-version=$SLIM_VERSION \
            --pop-size=$POP_SIZE \
            --n-iter=$N_ITER \
            --max-depth=$MAX_DEPTH \
            --p-inflate=$P_INFLATE \
            --sigmoid-scale=$SIGMOID_SCALE \
            --seed=$SEED \
            > $LOG_FILE 2>&1

        # Check if the script was successful
        RESULT=$?
        if [ $RESULT -eq 0 ]; then
            echo "Run $((i+1)) completed successfully."
        else
            echo "Run $((i+1)) failed with error code $RESULT."
            echo "Error details from $LOG_FILE:"
            echo "--------------------------------"
            tail -n 20 $LOG_FILE
            echo "--------------------------------"
            echo "See the full log at: $LOG_FILE"
        fi

        # Optional: add a short delay between runs
        sleep 1
    done

    # Print a summary of successes and failures for this dataset
    echo "============== SUMMARY FOR $DATASET ================="
    SUCCEEDED=0
    FAILED=0
    for ((i=0; i<$NUM_RUNS; i++)); do
        LOG_FILE="logs/$DATASET/$ALGORITHM/run_$((i+1))_seed_$((BASE_SEED + i)).log"
        if grep -q "Experiment completed successfully" $LOG_FILE; then
            ((SUCCEEDED++))
        else
            ((FAILED++))
        fi
    done

    echo "Successful runs: $SUCCEEDED"
    echo "Failed runs: $FAILED"
    echo "======================================================"
done

echo "All experiments completed!"
echo "Log files are available in the logs directory."
