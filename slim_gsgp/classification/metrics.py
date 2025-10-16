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
Evaluation metrics for classification tasks with SLIM-GSGP.

This module provides functions for evaluating classification performance.
"""
import csv
import datetime
import os

import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from slim_gsgp.utils.utils import create_result_directory


def calculate_binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    Calculate standard metrics for binary classification.

    Parameters
    ----------
    y_true : torch.Tensor
        True labels (0 or 1).
    y_pred : torch.Tensor
        Predicted labels (0 or 1).

    Returns
    -------
    Dict[str, float]
        Dictionary containing various classification metrics:
        - accuracy: Overall prediction accuracy
        - precision: Precision score (TP / (TP + FP))
        - recall: Recall/sensitivity score (TP / (TP + FN))
        - f1: F1 score (harmonic mean of precision and recall)
        - specificity: True negative rate (TN / (TN + FP))
        - confusion_matrix: Full confusion matrix
        - true_positives, true_negatives, false_positives, false_negatives: Counts
    """
    # Convert to numpy for sklearn metrics
    y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred_np),
        'precision': precision_score(y_true_np, y_pred_np, zero_division=0),
        'recall': recall_score(y_true_np, y_pred_np, zero_division=0),
        'f1': f1_score(y_true_np, y_pred_np, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_np, y_pred_np)
    }

    # Add derived metrics from confusion matrix
    cm = metrics['confusion_matrix']
    if cm.shape == (2, 2):  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)

        # Calculate specificity (true negative rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def save_metrics_to_csv(
        metrics: Dict[str, float],
        training_time: float,
        dataset_name: str,
        algorithm: str,
        root_dir: Optional[str] = None,
        additional_info: Optional[Dict] = None
) -> str:
    """
    Save classification metrics and experiment duration to a CSV file.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of classification metrics from calculate_binary_metrics
    training_time : float
        Training time in seconds
    dataset_name : str
        Name of the dataset used
    algorithm : str
        Algorithm used (gp, gsgp, slim)
    root_dir : str, optional
        Root directory for saving the file
    additional_info : Dict, optional
        Additional information to include in the CSV

    Returns
    -------
    str
        Path to the saved CSV file
    """
    # Get project root directory if not provided
    if root_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # Create metrics directory at the same level as visualizations
    metrics_dir = create_result_directory(
        root_dir=root_dir,
        dataset=dataset_name,
        algorithm=algorithm,
        result_type="metrics"
    )

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(metrics_dir, f"metrics_{timestamp}.csv")

    # Prepare data for CSV
    metrics_data = {
        # Metadata
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': dataset_name,
        'algorithm': algorithm,
        'training_time_seconds': training_time,
    }

    # Add metrics, excluding confusion matrix
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            metrics_data[key] = value

    # Add confusion matrix elements for binary classification
    # if 'confusion_matrix' in metrics:
    #     cm = metrics['confusion_matrix']
    #     if cm.shape == (2, 2):
    #         metrics_data['cm_tn'] = int(cm[0, 0])
    #         metrics_data['cm_fp'] = int(cm[0, 1])
    #         metrics_data['cm_fn'] = int(cm[1, 0])
    #         metrics_data['cm_tp'] = int(cm[1, 1])

    # Add additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            metrics_data[key] = value

    # Write to CSV
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = list(metrics_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(metrics_data)

    return csv_path