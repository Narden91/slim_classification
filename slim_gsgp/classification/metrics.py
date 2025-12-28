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
import logging

import torch
import numpy as np
from typing import Dict, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from slim_gsgp.utils.utils import create_result_directory
from .results import BinaryMetrics

logger = logging.getLogger(__name__)


def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor to numpy array efficiently.
    
    Parameters
    ----------
    tensor : Union[torch.Tensor, np.ndarray]
        Input tensor or array.
        
    Returns
    -------
    np.ndarray
        Numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def calculate_binary_metrics(
    y_true: Union[torch.Tensor, np.ndarray], 
    y_pred: Union[torch.Tensor, np.ndarray]
) -> BinaryMetrics:
    """
    Calculate standard metrics for binary classification.

    Parameters
    ----------
    y_true : Union[torch.Tensor, np.ndarray]
        True labels (0 or 1).
    y_pred : Union[torch.Tensor, np.ndarray]
        Predicted labels (0 or 1).

    Returns
    -------
    BinaryMetrics
        Dataclass containing various classification metrics:
        - accuracy: Overall prediction accuracy
        - precision: Precision score (TP / (TP + FP))
        - recall: Recall/sensitivity score (TP / (TP + FN))
        - f1: F1 score (harmonic mean of precision and recall)
        - specificity: True negative rate (TN / (TN + FP))
        - confusion_matrix: Full confusion matrix
        - true_positives, true_negatives, false_positives, false_negatives: Counts
        
    Examples
    --------
    >>> y_true = torch.tensor([0., 1., 1., 0., 1.])
    >>> y_pred = torch.tensor([0., 1., 0., 0., 1.])
    >>> metrics = calculate_binary_metrics(y_true, y_pred)
    >>> metrics.accuracy
    0.8
    >>> metrics['accuracy']  # Dict-like access also works
    0.8
    >>> metrics.true_positives
    2
    
    Notes
    -----
    Returns BinaryMetrics dataclass which supports both attribute access
    (metrics.accuracy) and dict-like access (metrics['accuracy']) for
    backward compatibility.
    """
    # Convert to numpy efficiently (single conversion per input)
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)

    # Calculate confusion matrix first
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    # Extract components for binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Handle edge case where only one class is present
        tp = tn = fp = fn = 0
        specificity = 0.0

    # Calculate basic metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)

    # Return typed dataclass
    return BinaryMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        specificity=float(specificity),
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        confusion_matrix=cm
    )


def save_metrics_to_csv(
        metrics: Dict[str, Union[float, int]],
        training_time: float,
        dataset_name: str,
        algorithm: str,
        root_dir: Optional[str] = None,
        additional_info: Optional[Dict[str, Union[str, int, float]]] = None
) -> str:
    """
    Save classification metrics and experiment duration to a CSV file.

    Parameters
    ----------
    metrics : Dict[str, Union[float, int]]
        Dictionary of classification metrics from calculate_binary_metrics
    training_time : float
        Training time in seconds
    dataset_name : str
        Name of the dataset used
    algorithm : str
        Algorithm used (gp, gsgp, slim)
    root_dir : str, optional
        Root directory for saving the file
    additional_info : Dict[str, Union[str, int, float]], optional
        Additional information to include in the CSV

    Returns
    -------
    str
        Path to the saved CSV file
        
    Examples
    --------
    >>> metrics = {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.97}
    >>> path = save_metrics_to_csv(metrics, 120.5, 'breast_cancer', 'gp')
    >>> assert os.path.exists(path)
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

    # Add additional info if provided
    if additional_info:
        for key, value in additional_info.items():
            metrics_data[key] = value

    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = list(metrics_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(metrics_data)

    return csv_path