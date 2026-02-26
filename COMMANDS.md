# COMMANDS

Concise CLI reference for running `slim_gsgp/example_binary_classification.py`.

## Quick start

```bash
# GP
python slim_gsgp/example_binary_classification.py --dataset=breast_cancer --algorithm=gp

# GSGP
python slim_gsgp/example_binary_classification.py --dataset=breast_cancer --algorithm=gsgp

# SLIM (choose a version)
python slim_gsgp/example_binary_classification.py --dataset=breast_cancer --algorithm=slim --slim-version=SLIM+SIG2
```

## Feature importance

```bash
# Fast (frequency only)
python slim_gsgp/example_binary_classification.py \
  --dataset=breast_cancer --algorithm=slim --slim-version=SLIM+SIG2 \
  --n-iter=50 --pop-size=100 \
  --feature-importance --importance-method=frequency --importance-top-n=10

# Full (frequency + depth + permutation)
python slim_gsgp/example_binary_classification.py \
  --dataset=breast_cancer --algorithm=gsgp \
  --n-iter=50 --pop-size=100 \
  --feature-importance --importance-method=all --importance-top-n=10 --importance-n-repeats=10
```

## Tree export

```bash
python slim_gsgp/example_binary_classification.py \
  --dataset=breast_cancer --algorithm=gp \
  --export-tree --export-format=all
```

## CINECA launch scripts

### Flexible launcher (recommended)

```bash
# Make executable once
chmod +x launch_cineca_flexible.sh

# Launch all task configurations from config/task_list.csv
./launch_cineca_flexible.sh --all

# Launch an inclusive task-id interval
./launch_cineca_flexible.sh --from 250 --to 1272

# Launch exactly 1023 tasks from the beginning (0..1022)
./launch_cineca_flexible.sh --count 1023

# Launch exactly 1023 tasks starting from a custom index
./launch_cineca_flexible.sh --from 500 --count 1023
```

Notes:
- Tasks are always submitted in array batches of 50.
- If total tasks is not a multiple of 50, the last batch is partial.
- Scripts used: `launch_cineca_flexible.sh` + `run_experiments_cineca_flexible.slurm`.

### Fixed block launcher (legacy)

```bash
# Launch a selected interval where size is a multiple of 50
./launch_cineca_block50.sh --from 0 --to 249
./launch_cineca_block50.sh --from 250 --to 499
```

Notes:
- This launcher enforces interval size as a multiple of 50.
- Scripts used: `launch_cineca_block50.sh` + `run_experiments_cineca_block50.slurm`.

## Experiments folder (30 runs)

### Local launchers

```bash
# Optional: activate project venv
source venv_slim/bin/activate

# Each command runs 30 runs by default (run_id 0..29)
./launch_exp_1_darwin.sh
./launch_exp_2_hand_stat.sh
./launch_exp_3_ablation.sh
./launch_exp_4_depth_ablation.sh
```

Optional overrides:

```bash
# Example: 30 runs starting from run_id 30
START_RUN=30 N_RUNS=30 ./launch_exp_1_darwin.sh

# Example: quick smoke setup
POP_SIZE=5 N_ITER=1 N_RUNS=1 ./launch_exp_2_hand_stat.sh
```

### CINECA launchers (per experiment)

```bash
# Each command submits 30 array tasks by default (task_id 0..29)
./launch_cineca_exp_1_darwin.sh
./launch_cineca_exp_2_hand_stat.sh
./launch_cineca_exp_3_ablation.sh
./launch_cineca_exp_4_depth_ablation.sh
```

Optional overrides:

```bash
# Example: submit next 30 tasks (30..59)
FROM_TASK=30 COUNT=30 ./launch_cineca_exp_1_darwin.sh
```

### Generic CINECA launcher (custom experiment script)

```bash
# Default behavior (if no range flags): 30 tasks from 0
./launch_custom_experiments.sh --script experiments/exp_1_darwin.py

# Explicit range/count variants
./launch_custom_experiments.sh --script experiments/exp_2_hand_stat.py --count 30
./launch_custom_experiments.sh --script experiments/exp_3_ablation.py --from 0 --to 29
./launch_custom_experiments.sh --script experiments/exp_4_depth_ablation.py --all
```

## Arguments (compact)

### Selection
- `--dataset` (str): dataset name (e.g. `breast_cancer`, `eeg`, ...)
- `--algorithm` (`gp|gsgp|slim`)
- `--slim-version` (SLIM only): `SLIM+SIG2|SLIM*SIG2|SLIM+ABS|SLIM*ABS|SLIM+SIG1|SLIM*SIG1`

### Training
- `--pop-size` (int): population size
- `--n-iter` (int): generations
- `--max-depth` (int or `None`): depth cap (use `--max-depth=None` to disable)
- `--seed` (int)
- `--device` (`auto|cpu|cuda`): tensor device selection

### Classification
- `--fitness-function` (str): `binary_rmse|binary_mse|binary_mae|binary_auc_roc|binary_mcc`
- `--use-sigmoid` (bool): pass `--use-sigmoid=True` or `--use-sigmoid=False`
- `--sigmoid-scale` (float)

### Feature importance
- `--feature-importance` (flag)
- `--importance-method` (`frequency|depth|permutation|all`)
- `--importance-top-n` (int)
- `--importance-n-repeats` (int): permutation repeats (higher = slower, more stable)

### SLIM-only
- `--p-inflate` (float): inflate mutation probability
- `--p-xo` (float): crossover probability (0 disables crossover)
- `--crossover-operator` (`one_point|uniform|none`): crossover operator used when `--p-xo > 0`

### Output
- `--verbose` (bool): pass `--verbose=True` or `--verbose=False`
- `--save-visualization` (bool): pass `--save-visualization=True` or `--save-visualization=False`

### Export (explainability)
- `--export-tree` (flag)
- `--export-format` (`html|svg|pdf|text|all`)
- `--export-path` (str)

### Checkpointing
- `--checkpoint-enabled` (flag)
- `--checkpoint-freq` (int)
- `--checkpoint-path` (str)
- `--resume` (flag)
- `--no-clean-checkpoint` (flag)

### Experiment registry
- `--registry-path` (str)
- `--experiment-id` (str)
- `--force-registry` (flag)
