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

# Ensure local project imports take priority over installed slim_gsgp package
import sys
import os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

"""
Binary Classification Example

This script runs a single binary classification experiment using the SLIM-GSGP framework.
It supports command-line arguments for configuring experiment parameters and properly
handles SLIM algorithm versions with dedicated subfolders for results.

Example usage:
    python binary_classification.py --dataset=breast_cancer --algorithm=gp --seed=42
    python binary_classification.py --dataset=breast_cancer --algorithm=slim --slim-version=SLIM+SIG2 --seed=42
"""

import time
import argparse
import torch
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset, load_classification_benchmark_dataset
from slim_gsgp.classification import (
    train_binary_classifier,
    register_classification_fitness_functions
)
from slim_gsgp.tree_visualizer import visualize_gp_tree
from slim_gsgp.utils.experiment_registry import (
    ExperimentRegistry,
    experiment_run_from_config,
)

# Import explainability module (optional - graceful fallback if not available)
try:
    from slim_gsgp.explainability import TreeExporter
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    TreeExporter = None

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
    parser.add_argument("--save-visualization", type=bool, default=False,
                        help="Save tree visualization")

    # Device control
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for tensors (auto/cpu/cuda)",
    )

    # Explainability / Tree export arguments
    parser.add_argument("--export-tree", action="store_true", default=False,
                        help="Export final tree (visualization and formula)")
    parser.add_argument("--export-format", type=str, default="all",
                        choices=["html", "svg", "pdf", "text", "all"],
                        help="Export format for tree: html (interactive), svg, pdf, text, or all")
    parser.add_argument("--export-path", type=str, default=None,
                        help="Custom output directory for tree exports (default: results folder)")

    # SLIM specific parameters
    parser.add_argument("--p-inflate", type=float, default=0.5,
                        help="Probability of inflate mutation for SLIM algorithm")

    # SLIM crossover parameters
    parser.add_argument(
        "--p-xo",
        type=float,
        default=0.0,
        help="SLIM crossover probability (0 disables crossover)",
    )
    parser.add_argument(
        "--crossover-operator",
        type=str,
        default="one_point",
        choices=["one_point", "uniform", "none"],
        help="SLIM crossover operator to use when --p-xo > 0",
    )

    # Checkpoint parameters
    parser.add_argument("--checkpoint-enabled", action="store_true", default=False,
                        help="Enable checkpointing to save and resume training")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="How often to save checkpoints (every N generations)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Directory path for storing checkpoints")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Attempt to resume from existing checkpoint")
    parser.add_argument("--no-clean-checkpoint", action="store_true", default=False,
                        help="Keep checkpoint files after successful completion")

    # Experiment-level checkpointing (registry)
    parser.add_argument("--registry-path", type=str, default=None,
                        help="Optional path to the experiment registry JSON file")
    parser.add_argument("--experiment-id", type=str, default=None,
                        help="Override the auto-generated registry identifier")
    parser.add_argument("--force-registry", action="store_true", default=False,
                        help="Force execution even if the registry marks the run as completed or running")

    # Feature importance extraction
    parser.add_argument("--feature-importance", action="store_true", default=False,
                        help="Extract and display feature importance from the best individual")
    parser.add_argument("--importance-method", type=str, default="all",
                        choices=["frequency", "depth", "permutation", "all"],
                        help="Feature importance method: frequency (fast), depth (fast), permutation (accurate), or all")
    parser.add_argument("--importance-top-n", type=int, default=10,
                        help="Number of top important features to display")
    parser.add_argument("--importance-n-repeats", type=int, default=10,
                        help="Number of permutation repeats for permutation importance (higher = more accurate but slower)")

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
        "save_visualization": False,

        # Explainability / Tree export
        "export_tree": False,
        "export_format": "all",
        "export_path": None,

        # SLIM specific parameters
        "p_inflate": 0.5,

        # SLIM crossover parameters
        "p_xo": 0.0,
        "crossover_operator": "one_point",
        
        # Performance options
        "device": "auto",  # "auto", "cuda", or "cpu" - auto uses GPU if available
        
        # Checkpoint parameters
        "checkpoint_enabled": False,
        "checkpoint_freq": 10,
        "checkpoint_path": None,
        "resume": False,
        "no_clean_checkpoint": False,

        # Experiment registry defaults
        "registry_path": None,
        "experiment_id": None,
        "force_registry": False,
        
        # Feature importance
        "feature_importance": False,
        "importance_method": "all",
        "importance_top_n": 10,
        "importance_n_repeats": 10,
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
        'dataset_name': dataset_name
    }

    # Add max_depth for algorithms that support it
    if args.algorithm in ['gp', 'slim']:
        algo_params['max_depth'] = args.max_depth

    # Create log directory if needed
    algorithm_id = get_algorithm_identifier(args)
    results_dir = os.path.join(get_project_root(), "results", dataset_name, algorithm_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Add algorithm-specific parameters
    if args.algorithm == 'gsgp':
        # For GSGP, ensure reconstruct=True to enable prediction
        algo_params['reconstruct'] = True
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(results_dir, f"run_seed_{args.seed}.csv")

    elif args.algorithm == 'slim':
        # For SLIM, set appropriate version
        algo_params['slim_version'] = args.slim_version
        algo_params['p_inflate'] = args.p_inflate
        algo_params['p_xo'] = getattr(args, 'p_xo', 0.0)
        algo_params['crossover_operator'] = getattr(args, 'crossover_operator', 'one_point')
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1

        algo_params['log_path'] = os.path.join(results_dir, f"run_seed_{args.seed}.csv")

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
        return f"slim_{args.slim_version.replace('*', 'MUL')}"
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


def extract_and_display_feature_importance(model, X_test, y_test, config, dataset_name):
    """
    Extract and display feature importance from the best individual.
    
    Parameters
    ----------
    model : BinaryClassifier
        Trained classification model
    X_test : torch.Tensor
        Test input data
    y_test : torch.Tensor
        Test target data
    config : argparse.Namespace
        Configuration object with feature importance settings
    dataset_name : str
        Name of the dataset
    """
    try:
        from slim_gsgp.explainability import FeatureImportanceExtractor
        from slim_gsgp.evaluators.fitness_functions import binary_cross_entropy
        
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        # Get the best individual (tree) from the underlying model
        # The model could be a GP/GSGP/SLIM instance or directly a Tree
        if hasattr(model.model, 'elite'):
            # It's a GP/GSGP/SLIM instance
            best_tree = model.model.elite
        else:
            # It's already a Tree object
            best_tree = model.model
        
        n_features = X_test.shape[1]
        
        print(f"\nBest tree depth: {best_tree.depth}")
        
        # node_count may not exist for all tree types (e.g., SLIM Individual objects)
        if hasattr(best_tree, 'node_count'):
            print(f"Best tree nodes: {best_tree.node_count}")
        
        if hasattr(best_tree, 'fitness'):
            print(f"Train fitness: {best_tree.fitness:.4f}")
        
        if hasattr(best_tree, 'test_fitness') and best_tree.test_fitness:
            print(f"Test fitness: {best_tree.test_fitness:.4f}")
        
        # Create feature importance extractor
        extractor = FeatureImportanceExtractor(n_features=n_features)
        
        # Determine which methods to run
        methods = []
        if config.importance_method == "all":
            methods = ["frequency", "depth", "permutation"]
        else:
            methods = [config.importance_method]
        
        # Dictionary to store results
        importance_results = {}
        
        # Run each method
        for method in methods:
            if method == "frequency":
                print("\nMethod: Frequency-Based (Fast)")
                print("  Counts how many times each feature appears in the tree")
                importance = extractor.frequency_importance(best_tree, normalize=True)
                importance_results['frequency'] = importance
                
            elif method == "depth":
                print("\nMethod: Depth-Weighted (Fast)")
                print("  Features closer to root have higher importance")
                importance = extractor.depth_weighted_importance(best_tree, normalize=True)
                importance_results['depth'] = importance
                
            elif method == "permutation":
                print("\nMethod: Permutation-Based (Accurate)")
                print(f"  Measures performance degradation when features are shuffled")
                print(f"  Using {config.importance_n_repeats} permutation repeats...")
                importance = extractor.permutation_importance(
                    best_tree, X_test, y_test, binary_cross_entropy,
                    n_repeats=config.importance_n_repeats,
                    normalize=True
                )
                importance_results['permutation'] = importance
        
        # Display results
        print("\n" + "-" * 70)
        print(f"Top {config.importance_top_n} Most Important Features:")
        print("-" * 70)
        
        for method_name, importance in importance_results.items():
            top_features = extractor.get_top_features(importance, n=config.importance_top_n)
            
            print(f"\n{method_name.upper()}:")
            print(f"  {'Feature':<12} {'Score':<10} {'Percentage':<12}")
            print(f"  {'-'*40}")
            
            for feat, score, _ in top_features:
                print(f"  {feat:<12} {score:<10.4f} {score*100:<10.1f}%")
        
        # Summary statistics
        features_used = sum(1 for s in importance_results[methods[0]].values() if s > 0)
        print(f"\n{'-'*70}")
        print("Summary:")
        print(f"  Total features in dataset: {n_features}")
        print(f"  Features used in best tree: {features_used}")
        print(f"  Feature utilization: {features_used / n_features * 100:.1f}%")
        print("=" * 70)
        
    except ImportError:
        print("\nWarning: Feature importance module not available.")
        print("Install the explainability module to use this feature.")
    except Exception as e:
        print(f"\nWarning: Failed to extract feature importance: {str(e)}")
        import traceback
        traceback.print_exc()


def export_tree_and_formula(model, config, root_dir, algorithm_id, verbose=True):
    """
    Export the final tree structure and formula for SLIM models.
    
    Parameters
    ----------
    model : BinaryClassifier
        The trained classification model.
    config : SimpleNamespace or argparse.Namespace
        Configuration object with export settings.
    root_dir : str
        Root directory of the project.
    algorithm_id : str
        Algorithm identifier for directory organization.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    dict or None
        Dictionary of exported file paths, or None if export failed.
    """
    if not EXPLAINABILITY_AVAILABLE:
        if verbose:
            print("Warning: Explainability module not available. Skipping tree export.")
        return None
    
    # Only export for SLIM models
    if config.algorithm != 'slim':
        if verbose:
            print(f"Note: Tree export is optimized for SLIM models. Current algorithm: {config.algorithm}")
    
    # Check if model has the required structure
    if not hasattr(model, 'model') or not hasattr(model.model, 'collection'):
        if verbose:
            print("Warning: Model does not have exportable tree structure.")
        return None
    
    try:
        # Get the underlying SLIM individual
        individual = model.model
        
        # Set the version attribute if not present
        if not hasattr(individual, 'version') and hasattr(config, 'slim_version'):
            individual.version = config.slim_version
        
        # Determine output directory
        if getattr(config, 'export_path', None):
            output_dir = config.export_path
        else:
            output_dir = os.path.join(root_dir, "results", config.dataset, algorithm_id, "exports")
        
        # Create exporter
        slim_version = getattr(config, 'slim_version', None)
        exporter = TreeExporter(slim_version=slim_version)
        
        # Print formula to console
        if verbose:
            exporter.print_formula(individual)
        
        # Export to files
        export_format = getattr(config, 'export_format', 'all')
        seed = getattr(config, 'seed', None)
        
        print(f"\nExporting final tree (format: {export_format}):")
        results = exporter.export(
            individual=individual,
            output_dir=output_dir,
            format=export_format,
            filename="final_tree",
            seed=seed,
            verbose=verbose
        )
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to export tree: {str(e)}")
            import traceback
            traceback.print_exc()
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

    # Ensure configured device is applied consistently across:
    # - algorithm constants (gp/gsgp/slim config modules)
    # - dataset tensors (X/y)
    requested_device = getattr(config, "device", "auto")
    if requested_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device=cuda requested but CUDA is not available; falling back to CPU.")
        resolved_device = "cpu"
    else:
        resolved_device = requested_device

    # Keep global config modules in sync with the requested device.
    # These modules generate constants using their internal DEVICE.
    try:
        from slim_gsgp.config.gp_config import set_device as _gp_set_device
        from slim_gsgp.config.gsgp_config import set_device as _gsgp_set_device
        from slim_gsgp.config.slim_config import set_device as _slim_set_device

        _gp_set_device(resolved_device)
        _gsgp_set_device(resolved_device)
        _slim_set_device(resolved_device)
    except Exception:
        # Device config is a performance feature; do not fail the experiment
        # if some environments import config modules differently.
        pass

    # Register binary fitness functions
    # register_classification_fitness_functions()

    # Load and split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test, n_classes, class_labels = load_and_split_dataset(
        config.dataset, config.seed
    )

    # Move tensors to the selected device (prevents CPU/CUDA mismatch when
    # constants/operators are built on GPU).
    device_obj = torch.device(resolved_device)
    X_train = X_train.to(device_obj)
    y_train = y_train.to(device_obj)
    X_val = X_val.to(device_obj)
    y_val = y_val.to(device_obj)
    X_test = X_test.to(device_obj)
    y_test = y_test.to(device_obj)

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
    print(f"  Device: {requested_device} (resolved: {resolved_device})")
    if config.algorithm == 'slim':
        print(f"  P-inflate: {config.p_inflate}")
        print(f"  P-xo: {getattr(config, 'p_xo', 0.0)}")
        print(f"  Crossover operator: {getattr(config, 'crossover_operator', 'one_point')}")
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

    # Evaluate on train set
    print("\nEvaluating on train set:")
    train_metrics = model.evaluate(X_train, y_train)

    # Print train metrics
    for name, value in train_metrics.items():
        if name != 'confusion_matrix':
            print(f"{name}: {value:.4f}")

    # Print train confusion matrix
    print("\nTrain Confusion Matrix:")
    cm_train = train_metrics['confusion_matrix']
    print(f"[{cm_train[0, 0]}, {cm_train[0, 1]}]")
    print(f"[{cm_train[1, 0]}, {cm_train[1, 1]}]")

    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_metrics = model.evaluate(X_test, y_test)

    # Print test metrics
    for name, value in test_metrics.items():
        if name != 'confusion_matrix':
            print(f"{name}: {value:.4f}")

    # Print test confusion matrix
    print("\nTest Confusion Matrix:")
    cm_test = test_metrics['confusion_matrix']
    print(f"[{cm_test[0, 0]}, {cm_test[0, 1]}]")
    print(f"[{cm_test[1, 0]}, {cm_test[1, 1]}]")

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
        additional_info['p_xo'] = getattr(config, 'p_xo', 0.0)
        additional_info['crossover_operator'] = getattr(config, 'crossover_operator', 'one_point')

    # Get algorithm identifier for directory creation
    algorithm_id = get_algorithm_identifier(config)

    # Create results directory
    results_dir = os.path.join(root_dir, "results", config.dataset, algorithm_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics_path = os.path.join(results_dir, "metrics.csv")
    settings_path = os.path.join(results_dir, "settings.csv")

    # Prepare data for metrics CSV
    metrics_data = {
        # Metadata
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': config.dataset,
        'algorithm': algorithm_id,
        'seed': config.seed,
        'training_time_seconds': training_time,
    }

    # Add train metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        if key in train_metrics:
            metrics_data[f'train_{key}'] = train_metrics[key]

    # Add test metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        if key in test_metrics:
            metrics_data[f'test_{key}'] = test_metrics[key]

    # Add confusion matrix values
    if 'confusion_matrix' in train_metrics:
        cm = train_metrics['confusion_matrix']
        if cm.shape == (2, 2):  # Binary classification
            metrics_data['train_true_negatives'] = int(cm[0, 0])
            metrics_data['train_false_positives'] = int(cm[0, 1])
            metrics_data['train_false_negatives'] = int(cm[1, 0])
            metrics_data['train_true_positives'] = int(cm[1, 1])

    if 'confusion_matrix' in test_metrics:
        cm = test_metrics['confusion_matrix']
        if cm.shape == (2, 2):  # Binary classification
            metrics_data['test_true_negatives'] = int(cm[0, 0])
            metrics_data['test_false_positives'] = int(cm[0, 1])
            metrics_data['test_false_negatives'] = int(cm[1, 0])
            metrics_data['test_true_positives'] = int(cm[1, 1])

    # Add additional info
    for key, value in additional_info.items():
        metrics_data[key] = value

    # Write metrics to CSV (append mode)
    file_exists = os.path.isfile(metrics_path)
    with open(metrics_path, 'a', newline='') as csvfile:
        fieldnames = list(metrics_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_data)

    print(f"\nMetrics saved to: {metrics_path}")

    # Save settings to CSV
    settings_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': config.dataset,
        'algorithm': algorithm_id,
        'seed': config.seed,
    }
    for key, value in additional_info.items():
        settings_data[key] = value

    # Write settings to CSV (append mode)
    file_exists = os.path.isfile(settings_path)
    with open(settings_path, 'a', newline='') as csvfile:
        fieldnames = list(settings_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(settings_data)

    print(f"Settings saved to: {settings_path}")

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

    # Export tree and formula if requested
    export_paths = None
    if getattr(config, 'export_tree', False):
        export_paths = export_tree_and_formula(
            model=model,
            config=config,
            root_dir=root_dir,
            algorithm_id=algorithm_id,
            verbose=True
        )

    # Extract and display feature importance if requested
    if getattr(config, 'feature_importance', False):
        extract_and_display_feature_importance(
            model=model,
            X_test=X_test,
            y_test=y_test,
            config=config,
            dataset_name=config.dataset
        )

    return test_metrics, training_time, metrics_path, vis_path, model


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

    registry_entry = None
    registry = None
    if args.registry_path:
        registry = ExperimentRegistry(Path(args.registry_path))
        registry_entry = experiment_run_from_config(args)
        if args.experiment_id:
            registry_entry.run_id = args.experiment_id
        registry_entry = registry.register(registry_entry)

        if registry_entry.status == "completed" and not args.force_registry:
            print("Experiment already completed according to registry. Use --force-registry to rerun.")
            return 0
        if registry_entry.status == "running" and not args.force_registry:
            print("Experiment is currently marked as running. Use --force-registry to reset the entry.")
            return 1
        if args.force_registry:
            registry_entry = registry.reset(registry_entry.run_id)

    try:
        if registry_entry:
            registry.mark_started(registry_entry.run_id)
        metrics, training_time, metrics_file, vis_path, _ = run_experiment(args)
        if registry_entry:
            registry.mark_completed(registry_entry.run_id, metrics_path=metrics_file, duration=training_time)
    except Exception as exc:
        if registry_entry:
            registry.mark_failed(registry_entry.run_id, str(exc))
        raise

    print("\nExperiment completed successfully.")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Metrics saved to: {metrics_file}")

    # Print visualization path if available
    if vis_path:
        print(f"Visualization saved to: {vis_path}.png")

    return 0


if __name__ == "__main__":
    main()