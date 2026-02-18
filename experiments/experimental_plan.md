# Experimental Plan and Setup for ICPR Manuscript

## 1. Introduction and Objectives
The objective of this study is to benchmark the **SLIM (Semantic Learning via Iterative Mappings)** algorithm against standard **Genetic Programming (GP)** and **Geometric Semantic Genetic Programming (GSGP)** on two handwriting-based datasets: **DARWIN** and **HAND_STAT**. The study aims to demonstrate SLIM's effectiveness in terms of predictive accuracy, generalization, and computational efficiency.

## 2. Methodology

### 2.1. Datasets
Two binary classification datasets are used:
1.  **DARWIN**: 174 instances (85 Patient, 89 Healthy), 450 numerical features.
    -   *Preprocessing*: Removal of `ID` column. No normalization (algorithm-intrinsic).
2.  **HAND_STAT**: 174 instances, mixed statistical and categorical features.
    -   *Preprocessing*: One-Hot Encoding for `Sex`, `Work`, `Education`. `Age` and other features kept numerical.
    -   *Split*: Stratified Shuffle Split (70% Training, 30% Testing). 30 independent splits (seeded 0-29).

### 2.2. Algorithms
Three evolutionary algorithms are compared:
1.  **Standard GP**: Tree-based GP with subtree mutation and crossover.
    -   *Representation*: Syntax trees.
2.  **Standard GSGP**: Geometric Semantic GP.
    -   *Representation*: Linear combination of trees (implicit).
    -   *Mutation*: Geometric Semantic Mutation (GSM).
3.  **SLIM**: The proposed method.
    -   *Representation*: Dynamic semantic mappings.
    -   *Strategy*: `SLIM+SIG2` (Sigmoid-based mutation on 2 trees).

### 2.3. Hyperparameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Population Size** | 100 | Standard balance between diversity and speed. |
| **Generations** | 100 | Sufficient for convergence on these datasets. |
| **Selection** | Tournament (k=2) | Low selection pressure to maintain diversity. |
| **Elitism** | 1 | Preserves the best solution. |
| **Initialization** | Ramped Half-and-Half (Depth 6) | Standard initialization. |
| **Crossover (GP)** | 0.8 | High crossover is standard for GP. |
| **Mutation (GP)** | 0.2 | Standard background operator. |
| **P_Inflation (SLIM)** | 0.2 | Probability of inflating (adding complexity). |
| **P_Deflation (SLIM)** | 0.8 | Probability of deflating (simplifying). |

## 3. Experiments

### Experiment 1: Benchmarking on DARWIN
- **Goal**: Assess classification performance on high-dimensional data.
- **Metrics**: Accuracy, Precision, Recall, Specificity, F1-Score, RMSE, Tree Size, Training Time.
- **Statistical Test**: Wilcoxon Rank-Sum (p < 0.05).

### Experiment 2: Benchmarking on HAND_STAT
- **Goal**: Assess performance on mixed-type data (categorical + numerical).
- **Metric Focus**: Robustness to different feature types.

### Experiment 3: Ablation Study (SLIM Hyperparameters)
- **Goal**: Investigate the impact of the **Inflation Probability (`p_inflate`)** on SLIM's performance and model size.
- **Setup**: Run SLIM on DARWIN with `p_inflate` $\in \{0.2, 0.5, 0.8\}$.
- **Hypothesis**: Lower `p_inflate` (0.2) leads to smaller models with comparable accuracy, while higher `p_inflate` (0.8) may overfit or bloat without significant gain.

### Experiment 4: Ablation Study (SLIM Initial Depth)
- **Goal**: Investigate the impact of the **Initial Depth (`init_depth`)** on SLIM's starting population and convergence behavior.
- **Setup**: Run SLIM on DARWIN with `init_depth` $\in \{4, 6, 8, 10\}$.
    - *Note*: To accommodate deeper initial trees without artificial constraints, `max_depth` is set to `None` for this experiment.
- **Hypothesis**: Deeper initial trees provide a "richer" starting semantic space, potentially accelerating convergence. However, they significantly increase the computational cost (training time) and may lead to bloated final models if not controlled by deflation. Standard depth (6) is expected to offer the best trade-off.

## 4. Evaluation Metrics Definitions
For binary classification (Positive=Patient, Negative=Healthy):

1.  **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
2.  **Precision**: $\frac{TP}{TP + FP}$ (Positive Predictive Value)
3.  **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ (True Positive Rate)
4.  **Specificity**: $\frac{TN}{TN + FP}$ (True Negative Rate) (Crucial for medical diagnosis)
5.  **F1-Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$

## 5. Software and Hardware
- **Language**: Python 3.10
- **Libraries**: PyTorch (Backend), Pandas, NumPy.
- **Hardware**: [Insert Spec of User Machine]
