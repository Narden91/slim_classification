# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Example script for binary classification using the SLIM-GSGP framework.

This script demonstrates how to use the specialized binary classification module
with integration of tree visualization and supports multiple experimental runs.
"""
import time
import os
import csv
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset
from slim_gsgp.classification import (
    train_binary_classifier,
    register_classification_fitness_functions,
    save_metrics_to_csv,
    BinaryClassifier
)
from slim_gsgp.tree_visualizer import visualize_gp_tree


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
) -> Tuple[Dict[str, Any], float, str, Optional[str]]:
    """
    Run a single binary classification experiment.

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

    Returns
    -------
    Tuple[Dict[str, Any], float, str, Optional[str]]
        Metrics, training time, metrics file path, and visualization path (if applicable)
    """
    # Set random seed
    torch.manual_seed(seed)

    # Label for run identifier (for logging)
    run_label = f"Run {run_index}" if run_index is not None else f"Seed {seed}"

    if verbose:
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

    # Set algorithm-specific parameters
    algo_params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'seed': seed,
        'dataset_name': dataset,
        'max_depth': max_depth,
    }

    # Create log directory if needed
    log_dir = os.path.join(root_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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

    if verbose:
        print(f"{run_label}: Training completed in {training_time:.2f} seconds")
        print()
        print(f"{run_label}: Evaluating on test set:")

    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)

    # Print metrics if verbose
    if verbose:
        for name, value in metrics.items():
            if name != 'confusion_matrix':
                print(f"{name}: {value:.4f}")

        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"[{cm[0, 0]}, {cm[0, 1]}]")
        print(f"[{cm[1, 0]}, {cm[1, 1]}]")

    # Save metrics to CSV file
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

    # Visualization path (if visualization is created)
    vis_path = None

    # Try to visualize the model
    if save_visualization:
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

            # Try to extract and visualize the tree
            if hasattr(model.model, 'repr_'):
                # For GP models
                tree_structure = model.model.repr_
                vis_path = os.path.join(vis_dir, filename_visualization)
                visualize_gp_tree(tree_structure, vis_path, 'png')
                if verbose:
                    print(f"{run_label}: Tree visualization saved to {vis_path}.png")
            elif hasattr(model.model, 'structure'):
                # For GSGP models
                tree_structure = model.model.structure
                vis_path = os.path.join(vis_dir, filename_visualization)
                visualize_gp_tree(tree_structure, vis_path, 'png')
                if verbose:
                    print(f"{run_label}: Tree visualization saved to {vis_path}.png")
            elif hasattr(model.model, 'collection'):
                # For SLIM models
                tree_structure = [t.structure for t in model.model.collection]
                vis_path = os.path.join(vis_dir, filename_visualization)
                visualize_gp_tree(tree_structure, vis_path, 'png')
                if verbose:
                    print(f"{run_label}: Tree visualization saved to {vis_path}.png")
        except Exception as e:
            if verbose:
                print(f"{run_label}: Could not visualize the model: {str(e)}")
            vis_path = None

    return metrics, training_time, metrics_file, vis_path


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
    Save all metrics from multiple runs to a single CSV file with one row per run
    and a final row with mean values.

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
        Path to the saved metrics file
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

    # Add the mean row
    rows.append(mean_row)

    # Write to CSV
    fieldnames = list(set().union(*[row.keys() for row in rows]))
    # Sort fieldnames for consistent ordering
    fieldnames.sort()
    # Move run_index to the front
    if 'run_index' in fieldnames:
        fieldnames.remove('run_index')
        fieldnames.insert(0, 'run_index')

    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return summary_path


def main():
    """
    Main function to run the binary classification example with multiple runs.
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Define parameters for all runs
    dataset = 'breast_cancer'  # Options: 'breast_cancer', 'iris', 'digits', 'wine'
    algorithm = 'gp'  # Options: 'gp', 'gsgp', 'slim'

    # Run configuration
    num_runs = 3  # Number of runs to execute
    seeds = [42, 123, 456]  # Random seeds for each run (should match num_runs)

    # Parameters for all runs
    pop_size = 50
    n_iter = 10
    max_depth = 8
    use_sigmoid = True
    sigmoid_scale = 1.0
    fitness_function = 'binary_rmse'

    # Whether to save visualizations for each run
    save_visualization = True

    # Whether to save individual metrics files (not necessary with unified summary)
    save_individual_metrics = False

    # Verbose output for individual runs
    verbose_individual_runs = True

    # Register binary fitness functions
    register_classification_fitness_functions()

    print(f"Running {num_runs} binary classification experiments with {algorithm.upper()} on {dataset}")
    print(f"Seeds: {seeds}")
    print()

    # Load the dataset
    print(f"Loading dataset: {dataset}")
    X, y, n_classes, class_labels = load_classification_dataset(dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    print()

    # Check if dataset is binary
    if n_classes != 2:
        raise ValueError(f"This example is for binary classification only. Dataset {dataset} has {n_classes} classes.")

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=seeds[0])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=seeds[0])

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

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
            run_index=i + 1
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

    print(f"\nVisualization files ({len(all_vis_paths)}):")
    if all_vis_paths:
        for i, file_path in enumerate(all_vis_paths):
            print(f"  Run {i + 1}: {os.path.basename(file_path)}.png")
    else:
        print("  None generated")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()