#!/bin/bash
# Script to launch SLURM array jobs for the explainability experiment

TASK_LIST="config/task_list_explainability.csv"
SLURM_SCRIPT="run_experiments_explainability.slurm"
PYTHON_SCRIPT="scripts/run_explainability_exp.py"
EXPORT_FORMATS="${EXPORT_FORMATS:-html,svg,text}"

if [ ! -f "$TASK_LIST" ]; then
    echo "Task list not found! Please generate it using scripts/generate_explainability_exp_tasks.py"
    exit 1
fi

# Count the number of jobs (excluding the header line)
TOTAL_TASKS=$(($(wc -l < "$TASK_LIST") - 1))

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "Task list is empty!"
    exit 1
fi

echo "Total tasks to run: $TOTAL_TASKS"
echo "Export formats: $EXPORT_FORMATS"

# CINECA often has a limit on array sizes (e.g., 1000). 
MAX_ARRAY_SIZE=999

if [ "$TOTAL_TASKS" -gt "$MAX_ARRAY_SIZE" ]; then
    echo "Task count exceeds limits, dispatching in batches of $MAX_ARRAY_SIZE..."
    for (( i=0; i<$TOTAL_TASKS; i+=$MAX_ARRAY_SIZE+1 )); do
        START_IDX=$i
        END_IDX=$((i + MAX_ARRAY_SIZE))
        
        if [ "$END_IDX" -ge "$TOTAL_TASKS" ]; then
            END_IDX=$((TOTAL_TASKS - 1))
        fi
        
        echo "Submitting array batch from $START_IDX to $END_IDX"
        sbatch \
            --export=ALL,ALLOW_ENV_OVERRIDE=0,TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT",EXPORT_FORMATS="$EXPORT_FORMATS" \
            --array=${START_IDX}-${END_IDX} \
            $SLURM_SCRIPT
        
        # Adding a small delay to avoid overwhelming the scheduler
        sleep 2
    done
else
    echo "Submitting array from 0 to $((TOTAL_TASKS - 1))"
    sbatch \
        --export=ALL,ALLOW_ENV_OVERRIDE=0,TASK_LIST="$TASK_LIST",PYTHON_SCRIPT="$PYTHON_SCRIPT",EXPORT_FORMATS="$EXPORT_FORMATS" \
        --array=0-$((TOTAL_TASKS - 1)) \
        $SLURM_SCRIPT
fi

echo "All batches submitted."
