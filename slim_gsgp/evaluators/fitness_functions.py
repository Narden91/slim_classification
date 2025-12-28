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
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    if y_true.device != y_pred.device or y_true.dtype != y_pred.dtype:
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MSE value.
    """
    if y_true.device != y_pred.device or y_true.dtype != y_pred.dtype:
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
    return torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value.
    """
    if y_true.device != y_pred.device or y_true.dtype != y_pred.dtype:
        y_true = y_true.to(device=y_pred.device, dtype=y_pred.dtype)
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value for integer predictions.
    """
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1)


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Signed error values.
    """
    return torch.sub(y_true, y_pred)


def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy (BCE) loss with numerical stability.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        BCE value.
        
    Notes
    -----
    Predictions are clamped to [1e-7, 1 - 1e-7] to prevent log(0) errors
    and ensure numerical stability.
    """
    # Clamp predictions to prevent log(0) errors
    eps = 1e-7
    y_pred_clamped = torch.clamp(y_pred, eps, 1 - eps)
    return torch.mean(
        -y_true * torch.log(y_pred_clamped) - (1 - y_true) * torch.log(1 - y_pred_clamped),
        dim=len(y_pred.shape) - 1
    )


def auc_roc_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Area Under the ROC Curve (AUC-ROC) for binary classification.
    
    This implementation uses the Mann-Whitney U statistic, which is equivalent
    to AUC-ROC. Returns 1 - AUC so it can be used as a minimization objective.

    Parameters
    ----------
    y_true : torch.Tensor
        True binary labels (0 or 1).
    y_pred : torch.Tensor
        Predicted probabilities or scores (continuous values).

    Returns
    -------
    torch.Tensor
        1 - AUC-ROC value (for minimization). Lower is better.
        Returns 0.5 if all predictions are identical or only one class present.
        
    Notes
    -----
    - For a perfect classifier, returns ~0.0 (AUC = 1.0)
    - For a random classifier, returns ~0.5 (AUC = 0.5)
    - For the worst classifier, returns ~1.0 (AUC = 0.0)
    - Handles edge cases gracefully (single class, identical predictions)
    
    Examples
    --------
    >>> y_true = torch.tensor([0., 0., 1., 1.])
    >>> y_pred = torch.tensor([0.1, 0.3, 0.7, 0.9])
    >>> loss = auc_roc_score(y_true, y_pred)
    >>> # Returns value close to 0.0 (perfect separation)
    """
    # Flatten tensors if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Get indices of positive and negative samples
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    # Handle edge cases
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    if n_pos == 0 or n_neg == 0:
        # Only one class present - return 0.5 (random performance)
        return torch.tensor(0.5, dtype=y_pred.dtype, device=y_pred.device)
    
    # Get predictions for positive and negative samples
    pos_preds = y_pred[pos_mask]
    neg_preds = y_pred[neg_mask]
    
    # Check if all predictions are identical
    if torch.all(pos_preds == pos_preds[0]) and torch.all(neg_preds == neg_preds[0]) and pos_preds[0] == neg_preds[0]:
        return torch.tensor(0.5, dtype=y_pred.dtype, device=y_pred.device)
    
    # Compute Mann-Whitney U statistic (equivalent to AUC)
    # Count how many times pos_pred > neg_pred
    pos_preds_expanded = pos_preds.unsqueeze(1)  # Shape: (n_pos, 1)
    neg_preds_expanded = neg_preds.unsqueeze(0)  # Shape: (1, n_neg)
    
    # Count comparisons: pos > neg (1.0), pos == neg (0.5), pos < neg (0.0)
    comparisons = (pos_preds_expanded > neg_preds_expanded).float() + \
                  0.5 * (pos_preds_expanded == neg_preds_expanded).float()
    
    # AUC is the average of all pairwise comparisons
    auc = comparisons.sum() / (n_pos * n_neg)
    
    # Return 1 - AUC for minimization (0 = perfect, 1 = worst)
    return 1.0 - auc


def matthews_correlation_coefficient(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Matthews Correlation Coefficient (MCC) for binary classification.
    
    MCC is a balanced measure that works well even with imbalanced classes.
    Returns 1 - ((MCC + 1) / 2) to convert to a minimization objective.

    Parameters
    ----------
    y_true : torch.Tensor
        True binary labels (0 or 1).
    y_pred : torch.Tensor
        Predicted binary labels (0 or 1) or probabilities that will be thresholded at 0.5.

    Returns
    -------
    torch.Tensor
        1 - normalized MCC (for minimization). Lower is better.
        - Returns 0.0 for perfect correlation (MCC = 1.0)
        - Returns 0.5 for random predictions (MCC = 0.0)
        - Returns 1.0 for perfect anti-correlation (MCC = -1.0)
        
    Notes
    -----
    MCC ranges from -1 (total disagreement) to +1 (perfect prediction).
    - MCC = 1: Perfect prediction
    - MCC = 0: Random prediction
    - MCC = -1: Total disagreement
    
    This implementation handles edge cases where denominators might be zero.
    
    Examples
    --------
    >>> y_true = torch.tensor([0., 0., 1., 1.])
    >>> y_pred = torch.tensor([0., 0., 1., 1.])
    >>> loss = matthews_correlation_coefficient(y_true, y_pred)
    >>> # Returns 0.0 (perfect prediction)
    
    >>> y_true = torch.tensor([0., 0., 1., 1.])
    >>> y_pred = torch.tensor([1., 1., 0., 0.])
    >>> loss = matthews_correlation_coefficient(y_true, y_pred)
    >>> # Returns 1.0 (worst prediction)
    """
    # Flatten and threshold predictions if they're probabilities
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Threshold predictions at 0.5 if they appear to be probabilities
    y_pred_binary = (y_pred >= 0.5).float()
    
    # Compute confusion matrix elements
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum().float()
    tn = ((y_true == 0) & (y_pred_binary == 0)).sum().float()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum().float()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum().float()
    
    # Compute MCC numerator
    numerator = tp * tn - fp * fn
    
    # Compute MCC denominator
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Handle edge cases where denominator is zero
    if denominator == 0:
        # If denominator is 0, either:
        # - All predictions are one class, or
        # - All true labels are one class
        # Return 0.5 (equivalent to random guessing)
        mcc = torch.tensor(0.0, dtype=y_pred.dtype, device=y_pred.device)
    else:
        mcc = numerator / denominator
    
    # Convert MCC from [-1, 1] to [0, 1] scale: (MCC + 1) / 2
    # Then return 1 - normalized_mcc for minimization
    normalized_mcc = (mcc + 1.0) / 2.0
    return 1.0 - normalized_mcc
