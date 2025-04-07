#!/bin/bash
# Script to run multiple binary classification experiments with different seeds

PYTHON_PATH=""  # which python

# Set the full path to the script
SCRIPT_PATH="slim_gsgp/example_binary_classification.py"

# Set the number of runs
NUM_RUNS=1

# Set the base seed (each run will use base_seed + run_index)
BASE_SEED=42

# Set the dataset and algorithm
DATASET="eeg"
ALGORITHM="slim"
SLIM_VERSION="SLIM+SIG2"

# Set other parameters
POP_SIZE=50
N_ITER=5
MAX_DEPTH=12

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p logs/$DATASET
mkdir -p logs/$DATASET/$ALGORITHM

# Print Python version and path to help with debugging
echo "============== SYSTEM INFO ==============="
echo "Python path: $PYTHON_PATH"
$PYTHON_PATH --version
echo "Script path: $SCRIPT_PATH"
echo "Working directory: $(pwd)"
echo "==========================================="

echo "Starting $NUM_RUNS experiments on $DATASET using $ALGORITHM algorithm..."

# Run multiple experiments with different seeds
for ((i=0; i<$NUM_RUNS; i++)); do
    # Calculate the seed for this run
    SEED=$((BASE_SEED + i))

    echo "Starting run $((i+1))/$NUM_RUNS with seed $SEED..."

    # Create a temporary file for capturing the output
    LOG_FILE="logs/$DATASET/$ALGORITHM/run_${i+1}_seed_${SEED}.log"

#    echo "Command:"
#    echo "$PYTHON_PATH $SCRIPT_PATH --dataset=$DATASET --algorithm=$ALGORITHM --slim-version=$SLIM_VERSION --pop-size=$POP_SIZE --n-iter=$N_ITER --max-depth=$MAX_DEPTH --seed=$SEED"

    # Run the Python script with the specified parameters
    $PYTHON_PATH $SCRIPT_PATH \
        --dataset=$DATASET \
        --algorithm=$ALGORITHM \
        --slim-version=$SLIM_VERSION \
        --pop-size=$POP_SIZE \
        --n-iter=$N_ITER \
        --max-depth=$MAX_DEPTH \
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

echo "All experiments completed!"
echo "Log files are available in the logs directory."

# Print a summary of successes and failures
echo "============== SUMMARY ================="
SUCCEEDED=0
FAILED=0
for ((i=0; i<$NUM_RUNS; i++)); do
    LOG_FILE="logs/$DATASET/$ALGORITHM/run_${i+1}_seed_$((BASE_SEED + i)).log"
    if grep -q "Experiment completed successfully" $LOG_FILE; then
        ((SUCCEEDED++))
    else
        ((FAILED++))
    fi
done

echo "Successful runs: $SUCCEEDED"
echo "Failed runs: $FAILED"
echo "======================================="