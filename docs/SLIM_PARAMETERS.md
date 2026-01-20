# SLIM Algorithm Parameters Reference

Complete reference for SLIM algorithm configuration in the `slim_gsgp` framework.

---

## Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--algorithm` | str | `slim` | Algorithm type: `gp`, `gsgp`, `slim` |
| `--slim-version` | str | `SLIM+ABS` | SLIM variant (see [Versions](#slim-versions)) |
| `--dataset` | str | `eeg` | Dataset name to use |
| `--seed` | int | `42` | Random seed for reproducibility |

---

## Population & Evolution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--pop-size` | int | `50` | Population size |
| `--n-iter` | int | `10` | Number of generations |
| `--max-depth` | int/None | `None` | Maximum tree depth (`None` = unlimited) |

---

## SLIM-Specific Parameters

### Mutation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--p-inflate` | float | `0.5` | Probability of inflate mutation (vs deflate) |

### Crossover

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--p-xo` | float | `0.0` | Crossover probability (`0` = disabled) |
| `--crossover-operator` | str | `one_point` | Operator: `one_point`, `uniform`, `none` |

> [!NOTE]
> When `--p-xo > 0`, the mutation probability is automatically set to `1 - p_xo`.

---

## Classification Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-sigmoid` | bool | `True` | Apply sigmoid activation |
| `--sigmoid-scale` | float | `1.0` | Sigmoid scaling factor |
| `--fitness-function` | str | `binary_rmse` | Options: `binary_rmse`, `binary_mse`, `binary_mae` |

---

## Performance & Device

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda` |
| `--verbose` | bool | `False` | Enable verbose output |

---

## Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--checkpoint-enabled` | flag | `False` | Enable training checkpoints |
| `--checkpoint-freq` | int | `10` | Checkpoint frequency (generations) |
| `--checkpoint-path` | str | `None` | Directory for checkpoints |
| `--resume` | flag | `False` | Resume from existing checkpoint |

---

## Explainability

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--export-tree` | flag | `False` | Export final tree visualization |
| `--export-format` | str | `all` | Format: `html`, `svg`, `pdf`, `text`, `all` |
| `--feature-importance` | flag | `False` | Extract feature importance |
| `--importance-method` | str | `all` | Method: `frequency`, `depth`, `permutation`, `all` |

---

## SLIM Versions

| Version | Operator | Activation |
|---------|----------|------------|
| `SLIM+SIG2` | Addition (+) | Sigmoid² |
| `SLIM*SIG2` | Multiplication (*) | Sigmoid² |
| `SLIM+ABS` | Addition (+) | Absolute Value |
| `SLIM*ABS` | Multiplication (*) | Absolute Value |
| `SLIM+SIG1` | Addition (+) | Sigmoid |
| `SLIM*SIG1` | Multiplication (*) | Sigmoid |

---

## Example Configurations

### Basic SLIM Run

```bash
python slim_gsgp/example_binary_classification.py \
    --dataset=eeg \
    --algorithm=slim \
    --slim-version=SLIM+SIG2 \
    --pop-size=100 \
    --n-iter=500 \
    --seed=42
```

### SLIM with Crossover Enabled

```bash
python slim_gsgp/example_binary_classification.py \
    --dataset=eeg \
    --algorithm=slim \
    --slim-version=SLIM+ABS \
    --pop-size=500 \
    --n-iter=2000 \
    --p-xo=0.3 \
    --crossover-operator=one_point \
    --p-inflate=0.7 \
    --seed=42
```

### SLIM with GPU and Checkpointing

```bash
python slim_gsgp/example_binary_classification.py \
    --dataset=gina \
    --algorithm=slim \
    --slim-version=SLIM*SIG2 \
    --pop-size=500 \
    --n-iter=5000 \
    --device=cuda \
    --checkpoint-enabled \
    --checkpoint-freq=100 \
    --seed=42
```

### Full Feature Extraction Run

```bash
python slim_gsgp/example_binary_classification.py \
    --dataset=liver \
    --algorithm=slim \
    --slim-version=SLIM+SIG2 \
    --pop-size=200 \
    --n-iter=1000 \
    --export-tree \
    --feature-importance \
    --importance-method=all \
    --verbose=True \
    --seed=42
```

---

## Example Bash Scripts

### Multi-Seed Experiment (No Crossover)

```bash
#!/bin/bash
SCRIPT="slim_gsgp/example_binary_classification.py"
DATASETS=("eeg" "liver" "fertility")
SLIM_VERSIONS=("SLIM+SIG2" "SLIM+ABS")

for DATASET in "${DATASETS[@]}"; do
    for VERSION in "${SLIM_VERSIONS[@]}"; do
        for SEED in {42..71}; do
            python $SCRIPT \
                --dataset=$DATASET \
                --algorithm=slim \
                --slim-version=$VERSION \
                --pop-size=500 \
                --n-iter=2000 \
                --p-inflate=0.7 \
                --sigmoid-scale=0.01 \
                --seed=$SEED
        done
    done
done
```

### Crossover Comparison Experiment

```bash
#!/bin/bash
SCRIPT="slim_gsgp/example_binary_classification.py"
DATASET="eeg"

# Run without crossover (baseline)
for SEED in {42..51}; do
    python $SCRIPT \
        --dataset=$DATASET \
        --algorithm=slim \
        --slim-version=SLIM+SIG2 \
        --pop-size=500 \
        --n-iter=2000 \
        --p-xo=0.0 \
        --seed=$SEED
done

# Run with one-point crossover
for SEED in {42..51}; do
    python $SCRIPT \
        --dataset=$DATASET \
        --algorithm=slim \
        --slim-version=SLIM+SIG2 \
        --pop-size=500 \
        --n-iter=2000 \
        --p-xo=0.3 \
        --crossover-operator=one_point \
        --seed=$SEED
done

# Run with uniform crossover
for SEED in {42..51}; do
    python $SCRIPT \
        --dataset=$DATASET \
        --algorithm=slim \
        --slim-version=SLIM+SIG2 \
        --pop-size=500 \
        --n-iter=2000 \
        --p-xo=0.3 \
        --crossover-operator=uniform \
        --seed=$SEED
done
```

---

## Crossover Operators

### One-Point Block Crossover
- Offspring = prefix(parent1) + suffix(parent2)
- Fast and preserves semantic blocks

### Uniform Block Crossover
- Each block randomly selected from either parent
- Higher diversity, potentially more disruptive

> [!TIP]
> Start with `--p-xo=0.2` and `--crossover-operator=one_point` for a conservative crossover setting.
