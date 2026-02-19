# Launching Experiments on Cineca

To launch these specific experiments using the existing `launch_cineca_flexible.sh` from the root directory, you can override the python script and task list via environment variables.

A helper task list at `config/experiments_task_list.csv` has been created, containing 30 seeds (rows 0-29).

## 1. Experiment 1: DARWIN Benchmark
```bash
# Launch 30 runs (seeds 0-29)
export PYTHON_SCRIPT="experiments/exp_1_darwin.py"
export TASK_LIST="config/experiments_task_list.csv"

./launch_cineca_flexible.sh --count 30
```

## 2. Experiment 2: HAND_STAT Benchmark
```bash
# Launch 30 runs (seeds 0-29)
export PYTHON_SCRIPT="experiments/exp_2_hand_stat.py"
export TASK_LIST="config/experiments_task_list.csv"

./launch_cineca_flexible.sh --count 30
```

## 3. Experiment 3: SLIM Ablation Study
```bash
# Launch 30 runs (seeds 0-29)
export PYTHON_SCRIPT="experiments/exp_3_ablation.py"
export TASK_LIST="config/experiments_task_list.csv"

./launch_cineca_flexible.sh --count 30
```

## Note on Dry Run
You can verify the setup locally before submitting by running:
```bash
python experiments/exp_1_darwin.py --dry-run
```
