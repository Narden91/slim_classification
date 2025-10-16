#!/usr/bin/env python3
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
Binary Classification Example

This script runs a single binary classification experiment using the SLIM-GSGP framework.
It supports command-line arguments for configuring experiment parameters and properly
handles SLIM algorithm versions with dedicated subfolders for results.

Example usage:
    python binary_classification.py --dataset=breast_cancer --algorithm=gp --seed=42
    python binary_classification.py --dataset=breast_cancer --algorithm=slim --slim-version=SLIM+SIG2 --seed=42
"""

import os
import time
import argparse
import torch
import numpy as np
import csv
from datetime import datetime
from types import SimpleNamespace

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset, load_classification_benchmark_dataset
from slim_gsgp.classification import (
    train_binary_classifier,
    register_classification_fitness_functions
)
from slim_gsgp.tree_visualizer import visualize_gp_tree

# Valid SLIM algorithm versions
SLIM_VERSIONS = ["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"]


def get_project_root():
    """Get the project root directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, os.pardir))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a binary classification experiment")

    # Dataset and algorithm selection
    parser.add_argument("--dataset", type=str, default="eeg",
                        help="Dataset to use.")
    parser.add_argument("--algorithm", type=str, default="slim",
                        choices=["gp", "gsgp", "slim"],
                        help="Algorithm to use (gp, gsgp, slim)")
    parser.add_argument("--slim-version", type=str, default="SLIM+ABS",
                        # "SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"
                        choices=SLIM_VERSIONS,
                        help="SLIM algorithm version (only used when algorithm=slim)")

    # Training parameters
    parser.add_argument("--pop-size", type=int, default=50,
                        help="Population size")
    parser.add_argument("--n-iter", type=int, default=10,
                        help="Number of iterations")
    
    def parse_max_depth(value):
        if value.strip().lower() == "none":
            return None
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("max-depth must be an integer or 'None'")

    parser.add_argument("--max-depth", type=parse_max_depth, default=None,
                        help="Maximum tree depth")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Classification parameters
    parser.add_argument("--use-sigmoid", type=bool, default=True,
                        help="Whether to use sigmoid activation")
    parser.add_argument("--sigmoid-scale", type=float, default=1.0,
                        help="Scaling factor for sigmoid")
    parser.add_argument("--fitness-function", type=str, default="binary_rmse",
                        help="Fitness function to use: binary_rmse, binary_mse, binary_mae")

    # Output control
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Print detailed output")
    parser.add_argument("--save-visualization", type=bool, default=True,
                        help="Save tree visualization")

    # SLIM specific parameters
    parser.add_argument("--p-inflate", type=float, default=0.5,
                        help="Probability of inflate mutation for SLIM algorithm")

    return parser.parse_args()


def create_default_experiment_config():
    """Create a default configuration for the experiment."""
    return {
        # Dataset and algorithm selection
        "dataset": "eeg",
        "algorithm": "slim",
        "slim_version": "SLIM+ABS",

        # Training parameters
        "pop_size": 50,
        "n_iter": 10,
        "max_depth": None,
        "seed": 42,

        # Classification parameters
        "use_sigmoid": True,
        "sigmoid_scale": 1.0,
        "fitness_function": "binary_rmse",

        # Output control
        "verbose": False,
        "save_visualization": True,

        # SLIM specific parameters
        "p_inflate": 0.5
    }


def load_and_split_dataset(dataset_name, seed):
    """
    Load and split a dataset into train, validation, and test sets.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load
    seed : int
        Random seed for reproducible splits

    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes, class_labels
    """
    print(f"Loading dataset: {dataset_name}")
    try:
        X, y, n_classes, class_labels = load_classification_benchmark_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading benchmark dataset: {e}")
        print("Trying to load standard dataset...")
        X, y, n_classes, class_labels = load_classification_dataset(dataset_name)

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    print()

    # Check if dataset is binary
    if n_classes != 2:
        raise ValueError(
            f"This example is for binary classification only. Dataset {dataset_name} has {n_classes} classes.")

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=seed)

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test, n_classes, class_labels


def setup_algorithm_params(args, dataset_name):
    """
    Set up algorithm-specific parameters.

    Parameters
    ----------
    args : argparse.Namespace or SimpleNamespace
        Command line arguments or configuration object
    dataset_name : str
        Dataset name

    Returns
    -------
    dict
        Algorithm-specific parameters
    """
    # Set common algorithm parameters
    algo_params = {
        'pop_size': args.pop_size,
        'n_iter': args.n_iter,
        'seed': args.seed,
        'dataset_name': dataset_name,
        'max_depth': args.max_depth
    }

    # Create log directory if needed
    log_dir = os.path.join(get_project_root(), "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add algorithm-specific parameters
    if args.algorithm == 'gsgp':
        # For GSGP, ensure reconstruct=True to enable prediction
        algo_params['reconstruct'] = True
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(log_dir, dataset_name, args.algorithm , f"gsgp_run_seed_{args.seed}.csv")

    elif args.algorithm == 'slim':
        # For SLIM, set appropriate version
        algo_params['slim_version'] = args.slim_version
        algo_params['p_inflate'] = args.p_inflate
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1

        # Create SLIM version specific log directory
        slim_log_dir = os.path.join(log_dir, dataset_name, args.slim_version)
        if not os.path.exists(slim_log_dir):
            os.makedirs(slim_log_dir)

        # algo_params['log_path'] = os.path.join(slim_log_dir, f"slim_{args.slim_version}_{args.seed}.csv")
        algo_params['log_path'] = os.path.join(slim_log_dir, f"run_seed_{args.seed}.csv")

    return algo_params


def get_algorithm_identifier(args):
    """
    Get a proper algorithm identifier for directory creation.

    Parameters
    ----------
    args : argparse.Namespace or SimpleNamespace
        Command line arguments or configuration object

    Returns
    -------
    str
        Algorithm identifier for directory creation
    """
    if args.algorithm == 'slim':
        return f"slim_{args.slim_version}"
    else:
        return args.algorithm


def create_visualization(model, args, root_dir, seed, verbose=True):
    """
    Create a visualization of the model tree structure.

    Parameters
    ----------
    model : BinaryClassifier
        The trained classifier model
    args : argparse.Namespace or SimpleNamespace
        Command line arguments or configuration object
    root_dir : str
        Project root directory
    seed : int
        Random seed used
    verbose : bool, optional
        Whether to print verbose output

    Returns
    -------
    str or None
        Path to the visualization file if created, None otherwise
    """
    try:
        # Get algorithm identifier
        algorithm_id = get_algorithm_identifier(args)

        # Create visualization directory
        vis_dir = create_result_directory(
            root_dir=root_dir,
            dataset=args.dataset,
            algorithm=algorithm_id,
            result_type="visualizations"
        )

        if verbose:
            print("\nTree text representation:")
            model.print_tree_representation()

        # Create a unique filename for the visualization
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classification_tree_{seed}"

        # Try to extract and visualize the tree
        tree_structure = None
        vis_path = None

        if hasattr(model.model, 'repr_'):
            # For GP models
            tree_structure = model.model.repr_
        elif hasattr(model.model, 'structure'):
            # For GSGP models
            tree_structure = model.model.structure
        elif hasattr(model.model, 'collection'):
            # For SLIM models
            tree_structure = [t.structure for t in model.model.collection]

        if tree_structure:
            vis_path = os.path.join(vis_dir, filename)
            visualize_gp_tree(tree_structure, vis_path, 'png')
            if verbose:
                print(f"Tree visualization saved to {vis_path}.png")

        return vis_path

    except Exception as e:
        if verbose:
            print(f"Could not visualize the model: {str(e)}")
        return None


def run_experiment(config):
    """
    Run a single binary classification experiment.

    Parameters
    ----------
    config : argparse.Namespace or SimpleNamespace
        Command line arguments or configuration object

    Returns
    -------
    tuple
        metrics, training_time, metrics_file_path, visualization_path
    """
    root_dir = get_project_root()

    # Register binary fitness functions
    # register_classification_fitness_functions()

    # Load and split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes, class_labels = load_and_split_dataset(
        config.dataset, config.seed
    )

    # Set up algorithm parameters
    algo_params = setup_algorithm_params(config, config.dataset)

    # Get algorithm display name
    algorithm_display = config.algorithm.upper()
    if config.algorithm == 'slim':
        algorithm_display = f"{algorithm_display} ({config.slim_version})"

    # Print experiment information
    print(f"Running binary classification with {algorithm_display} on {config.dataset}")
    print(f"Parameters:")
    print(f"  Population size: {config.pop_size}")
    print(f"  Iterations: {config.n_iter}")
    print(f"  Seed: {config.seed}")
    print(f"  Fitness function: {config.fitness_function}")
    print(f"  Use sigmoid: {config.use_sigmoid}")
    print(f"  Sigmoid scale: {config.sigmoid_scale}")
    print(f"  Max depth: {config.max_depth}")
    if config.algorithm == 'slim':
        print(f"  P-inflate: {config.p_inflate}")
    print()

    # Train the classifier
    print("Training binary classifier...")
    start_time = time.time()

    model = train_binary_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        algorithm=config.algorithm,
        use_sigmoid=config.use_sigmoid,
        sigmoid_scale=config.sigmoid_scale,
        fitness_function=config.fitness_function,
        **algo_params
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on test set
    print("\nEvaluating on test set:")
    metrics = model.evaluate(X_test, y_test)

    # Print metrics
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
        'pop_size': config.pop_size,
        'n_iter': config.n_iter,
        'seed': config.seed,
        'use_sigmoid': config.use_sigmoid,
        'sigmoid_scale': config.sigmoid_scale,
        'fitness_function': config.fitness_function,
        'max_depth': config.max_depth
    }

    if config.algorithm == 'slim':
        additional_info['slim_version'] = config.slim_version
        additional_info['p_inflate'] = config.p_inflate

    # Get algorithm identifier for directory creation
    algorithm_id = get_algorithm_identifier(config)

    # Create a custom dictionary for metrics to avoid duplication
    custom_metrics = {}

    # Add performance metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        if key in metrics:
            custom_metrics[key] = metrics[key]

    # Add confusion matrix values only once with consistent naming
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        if cm.shape == (2, 2):  # Binary classification
            custom_metrics['true_negatives'] = int(cm[0, 0])
            custom_metrics['false_positives'] = int(cm[0, 1])
            custom_metrics['false_negatives'] = int(cm[1, 0])
            custom_metrics['true_positives'] = int(cm[1, 1])

    # Handle any additional metrics that might be present
    for key in metrics:
        if key not in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'confusion_matrix',
                       'true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            custom_metrics[key] = metrics[key]
            
    # Create metrics directory
    metrics_dir = create_result_directory(
        root_dir=root_dir,
        dataset=config.dataset,
        algorithm=algorithm_id,
        result_type="metrics",
        experiment_name="Experiment_1"
    )

    # Generate timestamp for filename
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(metrics_dir, f"metrics_{config.seed}.csv")

    # Prepare data for CSV
    metrics_data = {
        # Metadata
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': config.dataset,
        'algorithm': algorithm_id,
        'training_time_seconds': training_time,
    }

    # Add metrics
    for key, value in custom_metrics.items():
        metrics_data[key] = value

    # Add additional info
    for key, value in additional_info.items():
        metrics_data[key] = value

    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = list(metrics_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics_data)

    print(f"\nMetrics saved to: {csv_path}")

    # Create visualization if requested
    vis_path = None
    if config.save_visualization:
        vis_path = create_visualization(
            model=model,
            args=config,
            root_dir=root_dir,
            seed=config.seed,
            verbose=config.verbose
        )

    return metrics, training_time, csv_path, vis_path, model


def run_experiment_with_config(config_dict=None):
    """
    Run an experiment with a custom configuration dictionary.

    Parameters
    ----------
    config_dict : dict, optional
        Custom configuration parameters. If None, uses default configuration.

    Returns
    -------
    tuple
        metrics, training_time, metrics_file_path, visualization_path, model
    """
    # Start with default configuration
    default_config = create_default_experiment_config()

    # Override with custom configuration if provided
    if config_dict:
        default_config.update(config_dict)

    # Convert dictionary to SimpleNamespace for compatibility with existing code
    config = SimpleNamespace(**default_config)

    # Run the experiment with the configuration
    return run_experiment(config)


def main():
    """Main function to run a binary classification experiment."""
    args = parse_arguments()

    print(f"SLIM-GSGP Binary Classification Example")
    print("=" * 60)

    metrics, training_time, metrics_file, vis_path, _ = run_experiment(args)

    print("\nExperiment completed successfully.")
    print(f"Training time: {training_time:.2f} seconds")

    # Print visualization path if available
    if vis_path:
        print(f"Visualization saved to: {vis_path}.png")

    return 0


if __name__ == "__main__":
    main()