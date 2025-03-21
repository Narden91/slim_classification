import time
import os
import csv
import torch
import numpy as np
import pandas as pd
import argparse
import wandb
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
        wandb_run: Optional[Any] = None,  # Add wandb_run parameter
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

    # Label for run identifier (for logging)
    run_label = f"Run {run_index}" if run_index is not None else f"Seed {seed}"
    run_prefix = f"run_{run_index}" if run_index is not None else f"seed_{seed}"

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

    # Log run start to wandb if available
    if wandb_run is not None:
        wandb_run.log({f"{run_prefix}/training_started": True})
        if verbose:
            print(f"{run_label}: Logging to Weights & Biases")

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

    # Log metrics and artifacts to wandb if available
    if wandb_run is not None:
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
        wandb_run.config.run_metrics[f"run_{run_index}"] = run_metrics

        # Log confusion matrix as a wandb plot
        if "confusion_matrix" in metrics:
            cm = metrics['confusion_matrix']
            preds = model.predict(X_test).cpu().numpy()
            y_true = y_test.cpu().numpy()

            # Create a confusion matrix plot
            cm_plot = wandb.plot.confusion_matrix(
                preds=preds.astype(int).tolist(),
                y_true=y_true.astype(int).tolist(),
                class_names=["Negative", "Positive"]
            )
            wandb_run.log({f"run_{run_index}/confusion_matrix": cm_plot})

        # Save tree visualization as an artifact
        if vis_path:
            artifact = wandb.Artifact(
                name=f"tree_viz_{dataset}_{algorithm}_run_{run_index}",
                type="tree_visualization"
            )
            artifact.add_file(f"{vis_path}.png")
            wandb_run.log_artifact(artifact)

            # Also log the image directly for easy viewing
            wandb_run.log({
                f"run_{run_index}/tree_visualization": wandb.Image(f"{vis_path}.png")
            })

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


def main(use_wandb=False, dataset='breast_cancer', algorithm='gp', num_runs=3,
         pop_size=50, n_iter=10, max_depth=8, seeds=None):
    """
    Main function to run the binary classification example with multiple runs
    and optional W&B tracking.

    Parameters
    ----------
    use_wandb : bool
        Whether to use Weights & Biases for tracking experiments
    dataset : str
        Dataset name to use (breast_cancer, iris, digits, wine)
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
    seeds : list
        List of random seeds for reproducibility
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

    # Initialize wandb if enabled
    wandb_run = None
    if use_wandb:
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
            print(f"Weights & Biases initialized - tracking run at {wandb.run.url}")
        except Exception as e:
            print(f"Warning: Failed to initialize Weights & Biases: {e}")
            print("Running without W&B logging")
            use_wandb = False

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

    # Log summary metrics and create visualizations in wandb
    if wandb_run is not None:
        try:
            # Create data for line plots showing trends across runs
            run_indices = list(range(1, num_runs + 1))

            # Prepare line chart data for metrics
            metrics_data = {metric: [] for metric in key_metrics}
            for i, metrics_dict in enumerate(all_metrics):
                for metric in key_metrics:
                    if metric in metrics_dict:
                        metrics_data[metric].append([i + 1, float(metrics_dict[metric])])

            # Create a single combined line chart for all metrics
            combined_data = []
            for metric in key_metrics:
                for run_idx, value in metrics_data[metric]:
                    combined_data.append([run_idx, value, metric])

            if combined_data:
                metrics_table = wandb.Table(columns=["Run", "Value", "Metric"], data=combined_data)
                wandb_run.log({"Metrics Across Runs": wandb.plot.line(
                    metrics_table, "Run", "Value", "Metric",
                    title="Classification Metrics Across Runs")
                })

            # Create a bar chart for training times
            time_data = [[i + 1, time] for i, time in enumerate(all_training_times)]
            if time_data:
                time_table = wandb.Table(columns=["Run", "Training Time (s)"], data=time_data)
                wandb_run.log({"Training Time by Run": wandb.plot.bar(
                    time_table, "Run", "Training Time (s)",
                    title="Training Time by Run")
                })

            # Create a summary table with all run results
            summary_table = wandb.Table(
                columns=["Run", "Seed", "Accuracy", "Precision", "Recall", "F1", "Specificity",
                         "True Positives", "True Negatives", "False Positives", "False Negatives", "Training Time (s)"]
            )

            for i, (metrics_dict, training_time, seed) in enumerate(
                    zip(all_metrics, all_training_times, seeds[:num_runs])):
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
                    training_time
                )

            wandb_run.log({"Results Summary": summary_table})

            # Create performance radar chart
            if num_runs > 0 and all_metrics:
                radar_data = []
                for i, metrics_dict in enumerate(all_metrics):
                    row = [f"Run {i + 1}"]
                    for metric in ["accuracy", "precision", "recall", "f1"]:
                        if metric in metrics_dict:
                            row.append(float(metrics_dict[metric]))
                        else:
                            row.append(0)
                    radar_data.append(row)

                radar_table = wandb.Table(
                    columns=["Run", "Accuracy", "Precision", "Recall", "F1"],
                    data=radar_data
                )

                wandb_run.log({"Performance Radar": wandb.plot.line(
                    radar_table,
                    "Run",
                    "Accuracy", "Precision", "Recall", "F1",
                    title="Performance Metrics by Run")
                })

            # Create confusion matrix comparison
            # Since we've already logged individual confusion matrices per run,
            # we can create a summary of key confusion matrix metrics
            cm_summary_data = []
            for i, metrics_dict in enumerate(all_metrics):
                if "confusion_matrix" in metrics_dict:
                    cm = metrics_dict["confusion_matrix"]
                    if cm.shape == (2, 2):
                        # Calculate metrics from confusion matrix
                        tn, fp, fn, tp = cm.ravel()
                        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                        cm_summary_data.append([
                            f"Run {i + 1}", int(tp), int(tn), int(fp), int(fn),
                            float(accuracy), float(precision), float(recall), float(specificity)
                        ])

            if cm_summary_data:
                cm_summary_table = wandb.Table(
                    columns=["Run", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "Specificity"],
                    data=cm_summary_data
                )
                wandb_run.log({"Confusion Matrix Summary": cm_summary_table})

            # Add statistical summary
            stats_data = []
            for metric in key_metrics:
                values = [float(m[metric]) for m in all_metrics if metric in m]
                if values:
                    stats_data.append([
                        metric.capitalize(),
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)
                    ])

            if stats_data:
                stats_table = wandb.Table(
                    columns=["Metric", "Mean", "Std Dev", "Min", "Max"],
                    data=stats_data
                )
                wandb_run.log({"Statistical Summary": stats_table})

            # Finish the wandb run
            wandb_run.finish()
            print(f"\nWeights & Biases tracking completed - view results at {wandb.run.url}")

        except Exception as e:
            print(f"Warning: Error during W&B summary logging: {e}")
            try:
                wandb_run.finish()
            except:
                pass

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
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

    args = parser.parse_args()

    # Parse seed list if provided
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    main(
        use_wandb=args.use_wandb,
        dataset=args.dataset,
        algorithm=args.algorithm,
        num_runs=args.num_runs,
        pop_size=args.pop_size,
        n_iter=args.n_iter,
        max_depth=args.max_depth,
        seeds=seeds
    )