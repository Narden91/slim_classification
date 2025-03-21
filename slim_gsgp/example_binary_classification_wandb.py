"""
Binary Classification Example with W&B Support

This module provides a modular implementation for running binary classification experiments
with optional Weights & Biases (W&B) integration for tracking and visualization.

The module is organized into several components:
- Experiment configuration and execution
- Results tracking and metrics collection
- Visualization utilities
- W&B integration
- Command-line interface

Example usage:
    python binary_classification_wandb.py --use-wandb=True --dataset=breast_cancer --algorithm=gp --num-runs=3
"""

import time
import os
import csv
import torch
import numpy as np
import pandas as pd
import argparse
import wandb
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset
from slim_gsgp.classification import (
    train_binary_classifier,
    register_classification_fitness_functions,
    save_metrics_to_csv,
    BinaryClassifier
)
from slim_gsgp.tree_visualizer import visualize_gp_tree


# ===== Experiment Configuration Functions =====

def setup_experiment_params(
        dataset: str,
        algorithm: str,
        pop_size: int,
        n_iter: int,
        max_depth: int,
        seed: int
) -> Dict[str, Any]:
    """
    Set up experiment parameters based on the algorithm type and other settings.

    Parameters
    ----------
    dataset : str
        Dataset name
    algorithm : str
        Algorithm to use (gp, gsgp, slim)
    pop_size : int
        Population size
    n_iter : int
        Number of iterations
    max_depth : int
        Maximum tree depth
    seed : int
        Random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary of algorithm-specific parameters
    """
    # Set common algorithm parameters
    algo_params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'seed': seed,
        'dataset_name': dataset,
        'max_depth': max_depth,
    }

    # Create log directory if needed
    log_dir = os.path.join(os.getcwd(), "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set algorithm-specific parameters
    if algorithm == 'gsgp':
        # For GSGP, ensure reconstruct=True to enable prediction
        algo_params['reconstruct'] = True
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(log_dir, f"gsgp_{seed}.csv")
    elif algorithm == 'slim':
        # For SLIM, set appropriate version
        algo_params['slim_version'] = 'SLIM+ABS'
        algo_params['p_inflate'] = 0.5
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(log_dir, f"slim_{seed}.csv")

    return algo_params


def load_and_split_dataset(
        dataset: str,
        seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Load a dataset and split it into train, validation, and test sets.

    Parameters
    ----------
    dataset : str
        Dataset name to load
    seed : int
        Random seed for reproducible splits

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes
    """
    # Load dataset
    X, y, n_classes, class_labels = load_classification_dataset(dataset)

    # Check if dataset is binary
    if n_classes != 2:
        raise ValueError(f"This example is for binary classification only. Dataset {dataset} has {n_classes} classes.")

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes


# ===== Experiment Execution Functions =====

def run_single_experiment(
        dataset: str,
        algorithm: str,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        root_dir: str,
        seed: int = 42,
        pop_size: int = 50,
        n_iter: int = 10,
        max_depth: int = 8,
        use_sigmoid: bool = True,
        sigmoid_scale: float = 1.0,
        fitness_function: str = 'binary_rmse',
        verbose: bool = True,
        save_visualization: bool = True,
        run_index: Optional[int] = None,
        wandb_run: Optional[Any] = None,
) -> Tuple[Dict[str, Any], float, str, Optional[str]]:
    """
    Run a single binary classification experiment with optional W&B logging.

    Parameters
    ----------
    dataset : str
        Dataset name
    algorithm : str
        Algorithm to use (gp, gsgp, slim)
    X_train : torch.Tensor
        Training features
    y_train : torch.Tensor
        Training labels
    X_val : torch.Tensor
        Validation features
    y_val : torch.Tensor
        Validation labels
    X_test : torch.Tensor
        Test features
    y_test : torch.Tensor
        Test labels
    root_dir : str
        Project root directory
    seed : int
        Random seed
    pop_size : int
        Population size
    n_iter : int
        Number of iterations
    max_depth : int
        Maximum tree depth
    use_sigmoid : bool
        Whether to use sigmoid activation
    sigmoid_scale : float
        Scaling factor for sigmoid
    fitness_function : str
        Fitness function to use
    verbose : bool
        Whether to print detailed output
    save_visualization : bool
        Whether to save tree visualization
    run_index : int, optional
        Index of the current run (for multi-run experiments)
    wandb_run : wandb.Run, optional
        Active wandb run for logging metrics and artifacts

    Returns
    -------
    Tuple[Dict[str, Any], float, str, Optional[str]]
        Metrics, training time, metrics file path, and visualization path
    """
    # Set random seed
    torch.manual_seed(seed)

    # Create run identifier for logging
    run_label = f"Run {run_index}" if run_index is not None else f"Seed {seed}"

    # Log experiment parameters
    if verbose:
        print_experiment_params(
            run_label, dataset, algorithm, pop_size, n_iter,
            seed, fitness_function, use_sigmoid, sigmoid_scale, max_depth
        )

    # Set up algorithm parameters
    algo_params = setup_experiment_params(
        dataset, algorithm, pop_size, n_iter, max_depth, seed
    )

    # Train the classifier
    start_time = time.time()
    if verbose:
        print(f"{run_label}: Training binary classifier...")

    model = train_binary_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        algorithm=algorithm,
        use_sigmoid=use_sigmoid,
        sigmoid_scale=sigmoid_scale,
        fitness_function=fitness_function,
        **algo_params
    )

    training_time = time.time() - start_time

    # Log training completion
    if verbose:
        print(f"{run_label}: Training completed in {training_time:.2f} seconds")
        print(f"{run_label}: Evaluating on test set:")

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    # Print metrics
    if verbose:
        print_metrics(metrics)

    # Save metrics to CSV
    metrics_file = save_experiment_metrics(
        metrics, training_time, dataset, algorithm, root_dir,
        pop_size, n_iter, seed, use_sigmoid, sigmoid_scale,
        fitness_function, max_depth, run_index, verbose, run_label
    )

    # Create visualization if requested
    vis_path = None
    if save_visualization:
        vis_path = create_visualization(
            model, dataset, algorithm, root_dir,
            run_index, seed, verbose, run_label
        )

    # Log metrics to W&B if enabled
    if wandb_run is not None:
        log_to_wandb(
            wandb_run, metrics, training_time,
            seed, run_index, vis_path
        )

    return metrics, training_time, metrics_file, vis_path


def print_experiment_params(
        run_label: str,
        dataset: str,
        algorithm: str,
        pop_size: int,
        n_iter: int,
        seed: int,
        fitness_function: str,
        use_sigmoid: bool,
        sigmoid_scale: float,
        max_depth: int
) -> None:
    """
    Print experiment parameters in a readable format.

    Parameters
    ----------
    run_label : str
        Label for the current run
    dataset : str
        Dataset name
    algorithm : str
        Algorithm name
    pop_size : int
        Population size
    n_iter : int
        Number of iterations
    seed : int
        Random seed
    fitness_function : str
        Fitness function name
    use_sigmoid : bool
        Whether to use sigmoid activation
    sigmoid_scale : float
        Sigmoid scaling factor
    max_depth : int
        Maximum tree depth
    """
    print(f"\n{'=' * 60}")
    print(f"{run_label}: Running binary classification with {algorithm.upper()} on {dataset}")
    print(f"{'=' * 60}")
    print(f"Parameters:")
    print(f"  Population size: {pop_size}")
    print(f"  Iterations: {n_iter}")
    print(f"  Seed: {seed}")
    print(f"  Fitness function: {fitness_function}")
    print(f"  Use sigmoid: {use_sigmoid}")
    print(f"  Sigmoid scale: {sigmoid_scale}")
    print(f"  Max depth: {max_depth}")
    print()


def print_metrics(metrics: Dict[str, Any]) -> None:
    """
    Print metrics in a readable format.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of metrics to print
    """
    for name, value in metrics.items():
        if name != 'confusion_matrix':
            print(f"{name}: {value:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"[{cm[0, 0]}, {cm[0, 1]}]")
    print(f"[{cm[1, 0]}, {cm[1, 1]}]")


# ===== Results and Metrics Functions =====

def save_experiment_metrics(
        metrics: Dict[str, Any],
        training_time: float,
        dataset: str,
        algorithm: str,
        root_dir: str,
        pop_size: int,
        n_iter: int,
        seed: int,
        use_sigmoid: bool,
        sigmoid_scale: float,
        fitness_function: str,
        max_depth: int,
        run_index: Optional[int],
        verbose: bool,
        run_label: str
) -> str:
    """
    Save metrics from an experiment to a CSV file.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of metrics to save
    training_time : float
        Time taken for training
    dataset : str
        Dataset name
    algorithm : str
        Algorithm used
    root_dir : str
        Project root directory
    pop_size : int
        Population size
    n_iter : int
        Number of iterations
    seed : int
        Random seed
    use_sigmoid : bool
        Whether sigmoid activation was used
    sigmoid_scale : float
        Sigmoid scaling factor
    fitness_function : str
        Fitness function used
    max_depth : int
        Maximum tree depth
    run_index : int, optional
        Index of the current run
    verbose : bool
        Whether to print verbose output
    run_label : str
        Label for the current run

    Returns
    -------
    str
        Path to the saved metrics file
    """
    additional_info = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'seed': seed,
        'use_sigmoid': use_sigmoid,
        'sigmoid_scale': sigmoid_scale,
        'fitness_function': fitness_function,
        'max_depth': max_depth,
        'run_index': run_index if run_index is not None else 'N/A'
    }

    metrics_file = save_metrics_to_csv(
        metrics=metrics,
        training_time=training_time,
        dataset_name=dataset,
        algorithm=algorithm,
        root_dir=root_dir,
        additional_info=additional_info
    )

    if verbose:
        print(f"\n{run_label}: Metrics saved to: {metrics_file}")

    return metrics_file


def save_unified_metrics(
        all_metrics: List[Dict[str, Any]],
        training_times: List[float],
        seeds: List[int],
        dataset: str,
        algorithm: str,
        run_params: Dict[str, Any],
        root_dir: str
) -> str:
    """
    Save unified metrics from multiple runs to a single CSV file.

    Parameters
    ----------
    all_metrics : List[Dict[str, Any]]
        List of metrics from all runs
    training_times : List[float]
        List of training times for all runs
    seeds : List[int]
        List of seeds used for each run
    dataset : str
        Dataset name
    algorithm : str
        Algorithm used
    run_params : Dict[str, Any]
        Parameters for the runs
    root_dir : str
        Project root directory

    Returns
    -------
    str
        Path to the saved unified metrics file
    """
    # Create metrics directory
    metrics_dir = create_result_directory(
        root_dir=root_dir,
        dataset=dataset,
        algorithm=algorithm,
        result_type="metrics"
    )

    # Fixed filename as requested
    summary_path = os.path.join(metrics_dir, "summary_metrics.csv")

    # Extract key metrics to include in the file
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity',
                   'true_positives', 'true_negatives', 'false_positives', 'false_negatives']

    # Prepare rows for the CSV file
    rows = []

    # Add one row for each run
    for i, (metrics, training_time, seed) in enumerate(zip(all_metrics, training_times, seeds)):
        row = {
            'run_index': i + 1,
            'seed': seed,
            'dataset': dataset,
            'algorithm': algorithm,
            'training_time_seconds': training_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add run parameters
        for key, value in run_params.items():
            if key != 'seeds':  # Skip the seeds list
                if isinstance(value, list) and len(value) > i:
                    row[key] = value[i]
                else:
                    row[key] = value

        # Add metrics
        for metric in key_metrics:
            if metric in metrics and not isinstance(metrics[metric], (dict, list, np.ndarray)):
                row[metric] = float(metrics[metric])

        # Add confusion matrix elements if available
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            if cm.shape == (2, 2):
                row['cm_tn'] = int(cm[0, 0])
                row['cm_fp'] = int(cm[0, 1])
                row['cm_fn'] = int(cm[1, 0])
                row['cm_tp'] = int(cm[1, 1])

        rows.append(row)

    # Add a row with mean values
    mean_row = create_mean_metrics_row(rows, dataset, algorithm, training_times, run_params)
    rows.append(mean_row)

    # Write to CSV
    write_metrics_to_csv(rows, summary_path)

    return summary_path


def create_mean_metrics_row(
        rows: List[Dict[str, Any]],
        dataset: str,
        algorithm: str,
        training_times: List[float],
        run_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a row with mean values for metrics across all runs.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        List of metric rows from individual runs
    dataset : str
        Dataset name
    algorithm : str
        Algorithm used
    training_times : List[float]
        List of training times for all runs
    run_params : Dict[str, Any]
        Parameters for the runs

    Returns
    -------
    Dict[str, Any]
        Row with mean values
    """
    mean_row = {
        'run_index': 'mean',
        'seed': 'N/A',
        'dataset': dataset,
        'algorithm': algorithm,
        'training_time_seconds': np.mean(training_times),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add run parameters (same as others)
    for key, value in run_params.items():
        if key != 'seeds':  # Skip the seeds list
            if not isinstance(value, list):
                mean_row[key] = value

    # Calculate mean for all numeric metrics across all runs
    all_keys = set().union(*[row.keys() for row in rows])
    for key in all_keys:
        if key not in mean_row:
            values = []
            for row in rows:
                if key in row and isinstance(row[key], (int, float)) and not isinstance(row[key], bool):
                    values.append(row[key])
            if values:
                mean_row[key] = np.mean(values)

    return mean_row


def write_metrics_to_csv(rows: List[Dict[str, Any]], file_path: str) -> None:
    """
    Write metrics rows to a CSV file.

    Parameters
    ----------
    rows : List[Dict[str, Any]]
        List of metric rows to write
    file_path : str
        Path to the CSV file to create
    """
    # Get all fieldnames from all rows
    fieldnames = list(set().union(*[row.keys() for row in rows]))

    # Sort fieldnames for consistent ordering
    fieldnames.sort()

    # Move run_index to the front
    if 'run_index' in fieldnames:
        fieldnames.remove('run_index')
        fieldnames.insert(0, 'run_index')

    # Write to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary_results(
        all_metrics: List[Dict[str, Any]],
        all_training_times: List[float],
        num_runs: int,
        summary_file: str,
        all_metrics_files: List[str] = None,
        all_vis_paths: List[str] = None,
        save_individual_metrics: bool = False
) -> None:
    """
    Print summary results from multiple runs.

    Parameters
    ----------
    all_metrics : List[Dict[str, Any]]
        List of metrics from all runs
    all_training_times : List[float]
        List of training times for all runs
    num_runs : int
        Number of runs executed
    summary_file : str
        Path to the summary metrics file
    all_metrics_files : List[str], optional
        List of paths to individual metrics files
    all_vis_paths : List[str], optional
        List of paths to visualization files
    save_individual_metrics : bool, optional
        Whether individual metrics files were saved
    """
    print("\n" + "=" * 60)
    print(f"SUMMARY RESULTS FOR {num_runs} RUNS")
    print("=" * 60)

    # Calculate mean and standard deviation for key metrics
    key_metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in key_metrics:
        values = [float(m[metric]) for m in all_metrics if metric in m]
        if values:
            print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Calculate mean and standard deviation for training time
    print(f"Training time: {np.mean(all_training_times):.2f} ± {np.std(all_training_times):.2f} seconds")

    print(f"\nUnified metrics saved to: {summary_file}")

    if save_individual_metrics and all_metrics_files:
        print(f"Individual metrics files ({len(all_metrics_files)}):")
        for i, file_path in enumerate(all_metrics_files):
            print(f"  Run {i + 1}: {os.path.basename(file_path)}")

    # Print visualization paths if any
    if all_vis_paths:
        print(f"\nVisualization files ({len(all_vis_paths)}):")
        for i, file_path in enumerate(all_vis_paths):
            print(f"  Run {i + 1}: {os.path.basename(file_path)}.png")
    else:
        print("\nVisualization files (0):\n  None generated")


# ===== Visualization Functions =====

def create_visualization(
        model: BinaryClassifier,
        dataset: str,
        algorithm: str,
        root_dir: str,
        run_index: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
        run_label: str = "Run"
) -> Optional[str]:
    """
    Create a visualization of the model's tree structure.

    Parameters
    ----------
    model : BinaryClassifier
        The trained classifier model
    dataset : str
        Dataset name
    algorithm : str
        Algorithm used
    root_dir : str
        Project root directory
    run_index : int, optional
        Index of the current run
    seed : int
        Random seed used
    verbose : bool
        Whether to print verbose output
    run_label : str
        Label for the current run

    Returns
    -------
    Optional[str]
        Path to the visualization if created, None otherwise
    """
    try:
        # Create visualization directory
        vis_dir = create_result_directory(
            root_dir=root_dir,
            dataset=dataset,
            algorithm=algorithm,
            result_type="visualizations"
        )

        if verbose:
            print(f"\n{run_label}: Tree text representation:")
            model.print_tree_representation()

        # Create a unique filename for the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_suffix = f"run_{run_index}" if run_index is not None else f"seed_{seed}"
        filename_visualization = f"classification_{run_suffix}_{timestamp}"

        # Extract tree structure based on model type
        vis_path = None
        if hasattr(model.model, 'repr_'):
            # For GP models
            tree_structure = model.model.repr_
            vis_path = os.path.join(vis_dir, filename_visualization)
            visualize_gp_tree(tree_structure, vis_path, 'png')
        elif hasattr(model.model, 'structure'):
            # For GSGP models
            tree_structure = model.model.structure
            vis_path = os.path.join(vis_dir, filename_visualization)
            visualize_gp_tree(tree_structure, vis_path, 'png')
        elif hasattr(model.model, 'collection'):
            # For SLIM models
            tree_structure = [t.structure for t in model.model.collection]
            vis_path = os.path.join(vis_dir, filename_visualization)
            visualize_gp_tree(tree_structure, vis_path, 'png')

        if vis_path and verbose:
            print(f"{run_label}: Tree visualization saved to {vis_path}.png")

        return vis_path

    except Exception as e:
        if verbose:
            print(f"{run_label}: Could not visualize the model: {str(e)}")
        return None


# ===== W&B Integration Functions =====

def init_wandb(
        dataset: str,
        algorithm: str,
        num_runs: int,
        seeds: List[int],
        pop_size: int,
        n_iter: int,
        max_depth: int,
        use_sigmoid: bool,
        sigmoid_scale: float,
        fitness_function: str
) -> Optional[Any]:
    """
    Initialize Weights & Biases for experiment tracking.

    Parameters
    ----------
    dataset : str
        Dataset name
    algorithm : str
        Algorithm to use
    num_runs : int
        Number of runs to execute
    seeds : List[int]
        List of random seeds
    pop_size : int
        Population size
    n_iter : int
        Number of iterations
    max_depth : int
        Maximum tree depth
    use_sigmoid : bool
        Whether to use sigmoid activation
    sigmoid_scale : float
        Sigmoid scaling factor
    fitness_function : str
        Fitness function to use

    Returns
    -------
    Optional[Any]
        W&B run object if initialization was successful, None otherwise
    """
    try:
        wandb_run = wandb.init(
            project="slim-gsgp-binary-classification",
            name=f"{dataset}_{algorithm}_{num_runs}_runs",
            config={
                "dataset": dataset,
                "algorithm": algorithm,
                "num_runs": num_runs,
                "seeds": seeds[:num_runs],
                "pop_size": pop_size,
                "n_iter": n_iter,
                "max_depth": max_depth,
                "use_sigmoid": use_sigmoid,
                "sigmoid_scale": sigmoid_scale,
                "fitness_function": fitness_function,
            }
        )
        print(f"\nWeights & Biases initialized - tracking run at {wandb.run.url}")
        return wandb_run
    except Exception as e:
        print(f"Warning: Failed to initialize Weights & Biases: {e}")
        print("Running without W&B logging")
        return None


def log_to_wandb(
        wandb_run: Any,
        metrics: Dict[str, Any],
        training_time: float,
        seed: int,
        run_index: Optional[int] = None,
        vis_path: Optional[str] = None
) -> None:
    """
    Log metrics and artifacts to W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    metrics : Dict[str, Any]
        Dictionary of metrics to log
    training_time : float
        Training time in seconds
    seed : int
        Random seed used
    run_index : int, optional
        Index of the current run
    vis_path : str, optional
        Path to visualization file
    """
    # Store metrics in run config for later summary
    if 'run_metrics' not in wandb_run.config:
        wandb_run.config.update({'run_metrics': {}}, allow_val_change=True)

    # Store metrics in a dictionary for later summarization
    run_metrics = {}
    for name, value in metrics.items():
        if name != 'confusion_matrix' and not isinstance(value, (list, np.ndarray)):
            run_metrics[name] = float(value)

    # Add run index and training time
    run_metrics['run_index'] = run_index if run_index is not None else 1
    run_metrics['training_time'] = training_time
    run_metrics['seed'] = seed

    # Update wandb config with the metrics
    run_key = f"run_{run_index}" if run_index is not None else f"run_1"
    wandb_run.config.run_metrics[run_key] = run_metrics

    # Log confusion matrix as a wandb plot
    if "confusion_matrix" in metrics:
        log_confusion_matrix_to_wandb(wandb_run, metrics, run_index)

    # Save tree visualization as an artifact
    if vis_path:
        log_visualization_to_wandb(wandb_run, vis_path, run_index)


def log_confusion_matrix_to_wandb(
        wandb_run: Any,
        metrics: Dict[str, Any],
        run_index: Optional[int] = None
) -> None:
    """
    Log confusion matrix to W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    metrics : Dict[str, Any]
        Dictionary of metrics containing confusion matrix
    run_index : int, optional
        Index of the current run
    """
    # Create confusion matrix labels
    cm = metrics['confusion_matrix']

    # Create mock predictions and labels for W&B format
    preds = np.zeros(cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1], dtype=int)
    actuals = np.zeros_like(preds)

    # Fill with values based on confusion matrix
    pos = 0

    # True negatives (0, 0)
    preds[pos:pos + cm[0, 0]] = 0
    actuals[pos:pos + cm[0, 0]] = 0
    pos += cm[0, 0]

    # False positives (0, 1)
    preds[pos:pos + cm[0, 1]] = 1
    actuals[pos:pos + cm[0, 1]] = 0
    pos += cm[0, 1]

    # False negatives (1, 0)
    preds[pos:pos + cm[1, 0]] = 0
    actuals[pos:pos + cm[1, 0]] = 1
    pos += cm[1, 0]

    # True positives (1, 1)
    preds[pos:pos + cm[1, 1]] = 1
    actuals[pos:pos + cm[1, 1]] = 1

    # Create a confusion matrix plot
    run_key = f"run_{run_index}" if run_index is not None else "run_1"
    cm_plot = wandb.plot.confusion_matrix(
        preds=preds.tolist(),
        y_true=actuals.tolist(),
        class_names=["Negative", "Positive"]
    )
    wandb_run.log({f"{run_key}/confusion_matrix": cm_plot})


def log_visualization_to_wandb(
        wandb_run: Any,
        vis_path: str,
        run_index: Optional[int] = None
) -> None:
    """
    Log visualization to W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    vis_path : str
        Path to visualization file
    run_index : int, optional
        Index of the current run
    """
    try:
        # Append .png extension if not already present
        img_path = f"{vis_path}.png" if not vis_path.endswith(".png") else vis_path

        if os.path.exists(img_path):
            run_key = f"run_{run_index}" if run_index is not None else "run_1"

            # Create artifact name
            artifact_name = f"tree_viz_run_{run_index}" if run_index is not None else "tree_viz"

            # Log as artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type="tree_visualization"
            )
            artifact.add_file(img_path)
            wandb_run.log_artifact(artifact)

            # Also log as image for easier viewing
            wandb_run.log({
                f"{run_key}/tree_visualization": wandb.Image(img_path)
            })
    except Exception as e:
        print(f"Warning: Failed to log visualization to W&B: {e}")


def log_summary_to_wandb(
        wandb_run: Any,
        all_metrics: List[Dict[str, Any]],
        all_training_times: List[float],
        seeds: List[int],
        all_vis_paths: List[str]
) -> None:
    """
    Log summary metrics and visualizations to W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    all_metrics : List[Dict[str, Any]]
        List of metrics from all runs
    all_training_times : List[float]
        List of training times for all runs
    seeds : List[int]
        List of seeds used for each run
    all_vis_paths : List[str]
        List of paths to visualization files
    """
    try:
        # Create summary table
        create_summary_table(wandb_run, all_metrics, all_training_times, seeds, all_vis_paths)

        # Create metrics plots
        # create_metrics_plots(wandb_run, all_metrics, all_training_times)

        # Create visualization gallery
        if any(all_vis_paths):
            create_visualization_gallery(wandb_run, all_metrics, seeds, all_vis_paths)

        # Finish W&B run
        wandb_run.finish()
        print(f"\nWeights & Biases tracking completed - view results at {wandb_run.url}")

    except Exception as e:
        print(f"Warning: Error during W&B summary logging: {e}")
        try:
            if wandb_run is not None:
                wandb_run.finish()
        except:
            pass


def create_summary_table(
        wandb_run: Any,
        all_metrics: List[Dict[str, Any]],
        all_training_times: List[float],
        seeds: List[int],
        all_vis_paths: List[str]
) -> None:
    """
    Create a summary table of results in W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    all_metrics : List[Dict[str, Any]]
        List of metrics from all runs
    all_training_times : List[float]
        List of training times for all runs
    seeds : List[int]
        List of seeds used for each run
    all_vis_paths : List[str]
        List of paths to visualization files
    """
    summary_table = wandb.Table(
        columns=["Run", "Seed", "Accuracy", "Precision", "Recall", "F1", "Specificity",
                 "True Positives", "True Negatives", "False Positives", "False Negatives",
                 "Training Time (s)", "Tree Visualization"]
    )

    for i, (metrics_dict, training_time, seed) in enumerate(
            zip(all_metrics, all_training_times, seeds)):

        # Get the visualization for this run
        tree_image = None
        if i < len(all_vis_paths) and all_vis_paths[i]:
            try:
                # Append .png extension if not already present
                img_path = f"{all_vis_paths[i]}.png" if not all_vis_paths[i].endswith(".png") else all_vis_paths[i]
                if os.path.exists(img_path):
                    tree_image = wandb.Image(img_path, caption=f"Run {i + 1} Tree")
            except Exception as e:
                print(f"Error loading visualization for run {i + 1}: {str(e)}")

        # Add row to summary table
        summary_table.add_data(
            i + 1,
            seed,
            float(metrics_dict.get("accuracy", 0)),
            float(metrics_dict.get("precision", 0)),
            float(metrics_dict.get("recall", 0)),
            float(metrics_dict.get("f1", 0)),
            float(metrics_dict.get("specificity", 0)),
            int(metrics_dict.get("true_positives", 0)),
            int(metrics_dict.get("true_negatives", 0)),
            int(metrics_dict.get("false_positives", 0)),
            int(metrics_dict.get("false_negatives", 0)),
            training_time,
            tree_image
        )

    # Log the summary table to wandb
    wandb_run.log({"Results Summary": summary_table})


def create_visualization_gallery(
        wandb_run: Any,
        all_metrics: List[Dict[str, Any]],
        seeds: List[int],
        all_vis_paths: List[str]
) -> None:
    """
    Create a gallery of visualizations in W&B.

    Parameters
    ----------
    wandb_run : Any
        W&B run object
    all_metrics : List[Dict[str, Any]]
        List of metrics from all runs
    seeds : List[int]
        List of seeds used for each run
    all_vis_paths : List[str]
        List of paths to visualization files
    """
    viz_panel = {}
    for i, vis_path in enumerate(all_vis_paths):
        if vis_path:
            img_path = f"{vis_path}.png" if not vis_path.endswith(".png") else vis_path
            if os.path.exists(img_path):
                seed_val = seeds[i] if i < len(seeds) else 'N/A'
                acc_val = all_metrics[i].get('accuracy', 0) if i < len(all_metrics) else 0
                viz_panel[f"Tree_Run_{i + 1}"] = wandb.Image(
                    img_path,
                    caption=f"Run {i + 1} (Seed: {seed_val}, Accuracy: {acc_val:.4f})"
                )

    if viz_panel:
        wandb_run.log({"Tree Visualizations Gallery": viz_panel})


# ===== Main Function =====

def run_experiments(
        dataset: str = 'breast_cancer',
        algorithm: str = 'gp',
        num_runs: int = 3,
        pop_size: int = 50,
        n_iter: int = 10,
        max_depth: int = 8,
        use_sigmoid: bool = True,
        sigmoid_scale: float = 1.0,
        fitness_function: str = 'binary_rmse',
        seeds: Optional[List[int]] = None,
        use_wandb: bool = False,
        save_visualization: bool = True,
        save_individual_metrics: bool = False,
        verbose_individual_runs: bool = True
) -> Tuple[List[Dict[str, Any]], List[float], str]:
    """
    Run multiple binary classification experiments with optional W&B tracking.

    Parameters
    ----------
    dataset : str
        Dataset name to use
    algorithm : str
        Algorithm to use (gp, gsgp, slim)
    num_runs : int
        Number of runs to execute
    pop_size : int
        Population size for GP algorithm
    n_iter : int
        Number of iterations for GP algorithm
    max_depth : int
        Maximum depth of GP trees
    use_sigmoid : bool
        Whether to use sigmoid activation
    sigmoid_scale : float
        Scaling factor for sigmoid
    fitness_function : str
        Fitness function to use
    seeds : List[int], optional
        List of random seeds for reproducibility
    use_wandb : bool
        Whether to use Weights & Biases for tracking
    save_visualization : bool
        Whether to save tree visualizations
    save_individual_metrics : bool
        Whether to save individual metrics files
    verbose_individual_runs : bool
        Whether to print verbose output for individual runs

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[float], str]
        List of metrics, list of training times, and path to summary metrics file
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Set default seeds if not provided
    if seeds is None:
        seeds = [42, 123, 456]

    # Register binary fitness functions
    register_classification_fitness_functions()

    print(f"Running {num_runs} binary classification experiments with {algorithm.upper()} on {dataset}")
    print(f"Seeds: {seeds}")
    print()

    # Load and split dataset
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes = load_and_split_dataset(
        dataset, seeds[0]
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Initialize wandb if enabled
    wandb_run = None
    if use_wandb:
        wandb_run = init_wandb(
            dataset, algorithm, num_runs, seeds,
            pop_size, n_iter, max_depth, use_sigmoid,
            sigmoid_scale, fitness_function
        )

    # Run all experiments
    all_metrics = []
    all_training_times = []
    all_metrics_files = []
    all_vis_paths = []

    print(f"Starting {num_runs} experimental runs...")

    for i in range(num_runs):
        # Get the seed for this run
        seed = seeds[i] if i < len(seeds) else seeds[0] + i

        # Run a single experiment
        metrics, training_time, metrics_file, vis_path = run_single_experiment(
            dataset=dataset,
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            root_dir=root_dir,
            seed=seed,
            pop_size=pop_size,
            n_iter=n_iter,
            max_depth=max_depth,
            use_sigmoid=use_sigmoid,
            sigmoid_scale=sigmoid_scale,
            fitness_function=fitness_function,
            verbose=verbose_individual_runs,
            save_visualization=save_visualization,
            run_index=i + 1,
            wandb_run=wandb_run if use_wandb else None
        )

        # Collect results
        all_metrics.append(metrics)
        all_training_times.append(training_time)
        if metrics_file:
            all_metrics_files.append(metrics_file)
        if vis_path:
            all_vis_paths.append(vis_path)

    # Save unified metrics file
    run_params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'max_depth': max_depth,
        'use_sigmoid': use_sigmoid,
        'sigmoid_scale': sigmoid_scale,
        'fitness_function': fitness_function,
    }

    summary_file = save_unified_metrics(
        all_metrics=all_metrics,
        training_times=all_training_times,
        seeds=seeds[:num_runs],
        dataset=dataset,
        algorithm=algorithm,
        run_params=run_params,
        root_dir=root_dir
    )

    # Print summary results
    print_summary_results(
        all_metrics=all_metrics,
        all_training_times=all_training_times,
        num_runs=num_runs,
        summary_file=summary_file,
        all_metrics_files=all_metrics_files,
        all_vis_paths=all_vis_paths,
        save_individual_metrics=save_individual_metrics
    )

    # Log summary to W&B if enabled
    if wandb_run is not None:
        log_summary_to_wandb(
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            all_training_times=all_training_times,
            seeds=seeds[:num_runs],
            all_vis_paths=all_vis_paths
        )

    print("\nExperiment completed successfully.")
    return all_metrics, all_training_times, summary_file


# ===== Command Line Interface =====

def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run binary classification experiments with SLIM-GSGP")
    parser.add_argument("--use-wandb", type=bool, default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--dataset", type=str, default="breast_cancer",
                        help="Dataset to use (breast_cancer, iris, etc.)")
    parser.add_argument("--algorithm", type=str, default="gp",
                        help="Algorithm to use (gp, gsgp, slim)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs to execute")
    parser.add_argument("--pop-size", type=int, default=50,
                        help="Population size")
    parser.add_argument("--n-iter", type=int, default=10,
                        help="Number of iterations")
    parser.add_argument("--max-depth", type=int, default=8,
                        help="Maximum tree depth")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds, e.g., '42,123,456'")

    return parser.parse_args()


def main():
    """
    Main entry point for the script when run from the command line.
    """
    args = parse_arguments()

    # Parse seed list if provided
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Run experiments
    run_experiments(
        use_wandb=args.use_wandb,
        dataset=args.dataset,
        algorithm=args.algorithm,
        num_runs=args.num_runs,
        pop_size=args.pop_size,
        n_iter=args.n_iter,
        max_depth=args.max_depth,
        seeds=seeds
    )


if __name__ == "__main__":
    main()