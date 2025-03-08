"""
Utilities for saving and loading classification metrics.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_metrics(metrics, dataset, algorithm, strategy=None, balance=False,
                params=None, root_dir=None, create_csv=True):
    """
    Save classification metrics to file.

    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics from evaluate_classification_model
    dataset : str
        Dataset name
    algorithm : str
        Algorithm type (gp, gsgp, slim)
    strategy : str, optional
        Classification strategy (ovr, ovo)
    balance : bool, optional
        Whether data balancing was used
    params : dict, optional
        Additional parameters used in training
    root_dir : str, optional
        Project root directory
    create_csv : bool, optional
        Whether to also create a CSV summary file

    Returns:
    --------
    str
        Path to the saved metrics file
    """
    # Get project root directory if not provided
    if root_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Create metrics directory structure
    metrics_dir = os.path.join(root_dir, "results", "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    # Create subdirectories
    dataset_dir = os.path.join(metrics_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if strategy:
        strategy_dir = os.path.join(dataset_dir, strategy)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
    else:
        strategy_dir = dataset_dir

    algo_dir = os.path.join(strategy_dir, algorithm)
    if not os.path.exists(algo_dir):
        os.makedirs(algo_dir)

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format the filename
    balance_str = "balanced" if balance else "unbalanced"
    filename = f"{dataset}_{strategy or 'binary'}_{algorithm}_{balance_str}_{timestamp}.json"
    filepath = os.path.join(algo_dir, filename)

    # Create a metrics dictionary with metadata
    metrics_with_meta = {
        "dataset": dataset,
        "algorithm": algorithm,
        "strategy": strategy,
        "balanced": balance,
        "timestamp": timestamp,
        "parameters": params or {},
        "metrics": metrics
    }

    # Extract the confusion matrix as it can't be serialized directly
    if "confusion_matrix" in metrics:
        metrics_with_meta["metrics"]["confusion_matrix"] = metrics["confusion_matrix"].tolist()

    # Remove the classification report string as it's better formatted for console
    if "classification_report" in metrics:
        # Store only if needed for later analysis
        metrics_with_meta["metrics"]["classification_report_text"] = metrics["classification_report"]
        del metrics_with_meta["metrics"]["classification_report"]

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(metrics_with_meta, f, cls=NumpyEncoder, indent=2)

    # Create a CSV summary for easy loading into pandas
    if create_csv:
        # Create organized metrics dictionary for CSV
        csv_data = {
            # Experiment metadata (first columns)
            "exp_id": timestamp,
            "dataset": dataset,
            "algorithm": algorithm,
            "strategy": strategy if strategy else "binary",
            "balanced_data": balance,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
        }

        # Add main metrics with metric_ prefix (for grouping)
        for key, value in metrics.items():
            if key not in ["confusion_matrix", "classification_report"]:
                # Handle numeric vs string metrics
                if isinstance(value, (int, float, np.number)):
                    if key.startswith(("precision", "recall", "f1")):
                        # Split multiclass metrics into their own columns
                        if "_" in key:  # Like precision_macro
                            metric_name, aggregation = key.split("_")
                            csv_data[f"metric_{metric_name}_{aggregation}"] = float(f"{value:.4f}")
                        else:
                            csv_data[f"metric_{key}"] = float(f"{value:.4f}")
                    else:
                        csv_data[f"metric_{key}"] = float(f"{value:.4f}")
                else:
                    # Handle string metrics (or other non-numeric types)
                    csv_data[f"metric_{key}"] = str(value)

        # Add parameters with param_ prefix
        if params:
            for key, value in params.items():
                # Format params for better readability
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                    csv_data[f"param_{key}"] = value
                else:
                    csv_data[f"param_{key}"] = str(value)

        # Create or append to CSV
        summary_file = os.path.join(metrics_dir, "all_results.csv")

        # Convert to DataFrame for easier CSV handling
        df = pd.DataFrame([csv_data])

        # Check if file exists
        if os.path.exists(summary_file):
            # Check if existing CSV has the same columns
            existing_df = pd.read_csv(summary_file)

            # If columns don't match, merge them
            if set(existing_df.columns) != set(df.columns):
                # Create a new DataFrame with all columns from both
                all_columns = sorted(list(set(existing_df.columns) | set(df.columns)))

                # Read the file again, specifying all columns
                existing_df = pd.read_csv(summary_file, usecols=lambda x: x in all_columns)

                # Append new data and save entire dataframe
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Organize columns in a logical order
                ordered_columns = []

                # First experiment metadata columns
                meta_cols = [col for col in combined_df.columns if not (col.startswith("metric_") or col.startswith("param_"))]
                ordered_columns.extend(sorted(meta_cols))

                # Then metrics columns
                metric_cols = [col for col in combined_df.columns if col.startswith("metric_")]
                ordered_columns.extend(sorted(metric_cols))

                # Finally parameter columns
                param_cols = [col for col in combined_df.columns if col.startswith("param_")]
                ordered_columns.extend(sorted(param_cols))

                # Save with ordered columns
                combined_df = combined_df[ordered_columns]
                combined_df.to_csv(summary_file, index=False)
            else:
                # Same columns, just append
                df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            # New file, sort columns logically
            ordered_columns = []

            # First experiment metadata columns
            meta_cols = [col for col in df.columns if not (col.startswith("metric_") or col.startswith("param_"))]
            ordered_columns.extend(sorted(meta_cols))

            # Then metrics columns
            metric_cols = [col for col in df.columns if col.startswith("metric_")]
            ordered_columns.extend(sorted(metric_cols))

            # Finally parameter columns
            param_cols = [col for col in df.columns if col.startswith("param_")]
            ordered_columns.extend(sorted(param_cols))

            # Save with ordered columns
            df = df[ordered_columns]
            df.to_csv(summary_file, index=False)

    return filepath


def load_metrics(filepath):
    """
    Load metrics from JSON file.

    Parameters:
    -----------
    filepath : str
        Path to the metrics JSON file

    Returns:
    --------
    dict
        Dictionary of metrics and metadata
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_metrics(root_dir=None):
    """
    Load all metrics from CSV summary file into a DataFrame.

    Parameters:
    -----------
    root_dir : str, optional
        Project root directory

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all metrics results
    """
    # Get project root directory if not provided
    if root_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    metrics_file = os.path.join(root_dir, "results", "metrics", "all_results.csv")

    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    else:
        return pd.DataFrame()