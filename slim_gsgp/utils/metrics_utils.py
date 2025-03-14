"""
Utilities for saving and loading classification metrics.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from slim_gsgp.utils.utils import create_result_directory


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

    # Create metrics directory using the utility function
    metrics_dir = create_result_directory(
        root_dir=root_dir,
        dataset=dataset,
        algorithm=algorithm,
        result_type="metrics",
        strategy=strategy
    )

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format the filename
    balance_str = "balanced" if balance else "unbalanced"
    strategy_str = f"_{strategy}" if strategy else "_binary"
    filename = f"{dataset}{strategy_str}_{algorithm}_{balance_str}_{timestamp}.json"
    filepath = os.path.join(metrics_dir, filename)

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

        # Extract confusion matrix elements (TN, TP, FN, FP) for binary classification
        if "confusion_matrix" in metrics:
            conf_matrix = np.array(metrics["confusion_matrix"])  # Convert to numpy array for consistency

            # For binary classification (2x2 matrix)
            if conf_matrix.shape == (2, 2):
                csv_data["TN"] = int(conf_matrix[0, 0])
                csv_data["FP"] = int(conf_matrix[0, 1])
                csv_data["FN"] = int(conf_matrix[1, 0])
                csv_data["TP"] = int(conf_matrix[1, 1])

            # For multiclass, include total numbers
            else:
                # Calculate overall TP, TN, FP, FN for multiclass
                # (This is a simplification; multiclass confusion matrices are more complex)
                tp_sum = np.sum(np.diag(conf_matrix))  # Sum of diagonal elements (correct predictions)
                total = np.sum(conf_matrix)
                fp_sum = total - tp_sum  # All incorrect predictions

                csv_data["TP_total"] = int(tp_sum)
                csv_data["FP_total"] = int(fp_sum)

        # Add parameters with param_ prefix
        if params:
            for key, value in params.items():
                # Format params for better readability
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                    csv_data[f"param_{key}"] = value
                else:
                    csv_data[f"param_{key}"] = str(value)

        # Create or append to CSV
        # Save the CSV summary in the main results directory
        results_dir = os.path.join(root_dir, "results")
        summary_file = os.path.join(results_dir, "all_results.csv")

        # Convert to DataFrame for easier CSV handling
        df = pd.DataFrame([csv_data])

        # Define the preferred column order
        preferred_order = [
            # First metadata columns
            "exp_id", "dataset", "algorithm", "strategy", "balanced_data", "date", "time",
            # Then metrics columns (these will be dynamically identified)
            # Then confusion matrix elements
            "TN", "FP", "FN", "TP", "TP_total", "FP_total"
            # Parameter columns will be added at the end
        ]

        # Check if file exists
        if os.path.exists(summary_file):
            try:
                # Read existing file
                existing_df = pd.read_csv(summary_file)

                # Combine DataFrames
                combined_df = pd.concat([existing_df, df], ignore_index=True)

                # Get all columns and organize them
                all_columns = list(combined_df.columns)

                # Organize columns in a logical order
                ordered_columns = []

                # Add metadata columns in preferred order
                for col in preferred_order:
                    if col in all_columns:
                        ordered_columns.append(col)
                        all_columns.remove(col)

                # Add remaining metadata columns not in preferred order
                meta_cols = [col for col in all_columns if
                             not (col.startswith("metric_") or col.startswith("param_"))]
                ordered_columns.extend(sorted(meta_cols))

                # Add metric columns
                metric_cols = [col for col in all_columns if col.startswith("metric_")]
                ordered_columns.extend(sorted(metric_cols))

                # Add parameter columns
                param_cols = [col for col in all_columns if col.startswith("param_")]
                ordered_columns.extend(sorted(param_cols))

                # Reorder and save
                combined_df = combined_df[ordered_columns]
                combined_df.to_csv(summary_file, index=False)

            except Exception as e:
                # If there's an error reading the existing file, create a new one
                print(f"Error handling existing CSV file: {e}. Creating a new file.")
                os.rename(summary_file, f"{summary_file}.bak.{timestamp}")
                df.to_csv(summary_file, index=False)
        else:
            # New file, organize columns logically
            all_columns = list(df.columns)
            ordered_columns = []

            # Add metadata columns in preferred order
            for col in preferred_order:
                if col in all_columns:
                    ordered_columns.append(col)
                    all_columns.remove(col)

            # Add remaining metadata columns not in preferred order
            meta_cols = [col for col in all_columns if
                         not (col.startswith("metric_") or col.startswith("param_"))]
            ordered_columns.extend(sorted(meta_cols))

            # Add metric columns
            metric_cols = [col for col in all_columns if col.startswith("metric_")]
            ordered_columns.extend(sorted(metric_cols))

            # Add parameter columns
            param_cols = [col for col in all_columns if col.startswith("param_")]
            ordered_columns.extend(sorted(param_cols))

            # Reorder and save
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
    import json

    with open(filepath, 'r') as f:
        return json.load(f)


def load_all_metrics(root_dir=None, dataset=None, algorithm=None, strategy=None):
    """
    Load all metrics from CSV summary file into a DataFrame.

    Parameters:
    -----------
    root_dir : str, optional
        Project root directory
    dataset : str, optional
        Filter by dataset name
    algorithm : str, optional
        Filter by algorithm
    strategy : str, optional
        Filter by strategy

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all metrics results
    """
    import os
    import pandas as pd

    # Get project root directory if not provided
    if root_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    metrics_file = os.path.join(root_dir, "results", "all_results.csv")

    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)

        # Apply filters if provided
        if dataset:
            df = df[df['dataset'] == dataset]
        if algorithm:
            df = df[df['algorithm'] == algorithm]
        if strategy:
            df = df[df['strategy'] == strategy]

        return df
    else:
        return pd.DataFrame()
