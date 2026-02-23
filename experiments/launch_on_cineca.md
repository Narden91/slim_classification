# Launching Experiments on Cineca

To launch these specific experiments, use the dedicated `launch_custom_experiments.sh` script located in the root directory. This script will automatically map the appropriate custom Python configuration to the SLURM scheduler array, deploying runs perfectly segregated into `results/experiments/{experiment_name}`.

A helper task list at `config/experiments_task_list.csv` has been created, dictating the 30 seeds (rows 0-29).

## 1. Experiment 1: DARWIN Benchmark
```bash
# Launch 30 runs (seeds 0-29)
./launch_custom_experiments.sh --script experiments/exp_1_darwin.py --count 30
```

## 2. Experiment 2: HAND_STAT Benchmark
```bash
# Launch 30 runs (seeds 0-29)
./launch_custom_experiments.sh --script experiments/exp_2_hand_stat.py --count 30
```

## 3. Experiment 3: SLIM Ablation Study
```bash
# Launch 30 runs (seeds 0-29)
./launch_custom_experiments.sh --script experiments/exp_3_ablation.py --count 30
```

## Note on Dry Run
You can verify the setup locally before submitting by running:
```bash
python experiments/exp_1_darwin.py --dry-run
```
