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
Input validation functions for classification module.

This module provides comprehensive validation for tensors, labels,
and configuration parameters used in classification tasks.
"""

import torch
from typing import Optional

from .exceptions import (
    InvalidLabelError,
    InvalidThresholdError,
    InvalidShapeError
)


def validate_binary_labels(
    labels: torch.Tensor,
    name: str = "labels"
) -> None:
    """
    Validate that labels are binary (only 0 and 1).
    
    Parameters
    ----------
    labels : torch.Tensor
        Labels to validate.
    name : str
        Name of the labels for error messages.
        
    Raises
    ------
    InvalidLabelError
        If labels are not binary.
    InvalidShapeError
        If labels tensor is empty.
        
    Examples
    --------
    >>> labels = torch.tensor([0., 1., 1., 0.])
    >>> validate_binary_labels(labels)  # No error
    
    >>> labels = torch.tensor([0., 1., 2.])
    >>> validate_binary_labels(labels)  # Raises InvalidLabelError
    """
    if labels.numel() == 0:
        raise InvalidShapeError(f"{name} tensor is empty")
    
    unique_values = torch.unique(labels)
    
    if len(unique_values) > 2:
        raise InvalidLabelError(
            f"{name} must be binary (0 or 1), but found {len(unique_values)} unique values: "
            f"{unique_values.tolist()}"
        )
    
    if not torch.all((labels == 0) | (labels == 1)):
        invalid_values = unique_values[~torch.isin(unique_values, torch.tensor([0., 1.]))]
        raise InvalidLabelError(
            f"{name} must contain only 0 or 1, but found invalid values: "
            f"{invalid_values.tolist()}"
        )


def validate_threshold(threshold: float) -> None:
    """
    Validate classification threshold.
    
    Parameters
    ----------
    threshold : float
        Threshold value to validate.
        
    Raises
    ------
    InvalidThresholdError
        If threshold is not in (0, 1) range.
        
    Examples
    --------
    >>> validate_threshold(0.5)  # No error
    >>> validate_threshold(1.5)  # Raises InvalidThresholdError
    """
    if not isinstance(threshold, (int, float)):
        raise InvalidThresholdError(
            f"Threshold must be numeric, got {type(threshold).__name__}"
        )
    
    if not 0 < threshold < 1:
        raise InvalidThresholdError(
            f"Threshold must be in range (0, 1), got {threshold}"
        )


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_features: Optional[int] = None,
    name: str = "tensor"
) -> None:
    """
    Validate tensor shape and dimensions.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to validate.
    expected_features : int, optional
        Expected number of features (columns).
    name : str
        Name of tensor for error messages.
        
    Raises
    ------
    InvalidShapeError
        If tensor has invalid shape.
        
    Examples
    --------
    >>> X = torch.randn(100, 10)
    >>> validate_tensor_shape(X, expected_features=10)  # No error
    
    >>> validate_tensor_shape(X, expected_features=5)  # Raises InvalidShapeError
    """
    if tensor.numel() == 0:
        raise InvalidShapeError(f"{name} tensor is empty")
    
    if tensor.ndim == 0:
        raise InvalidShapeError(
            f"{name} must have at least 1 dimension, got scalar"
        )
    
    if tensor.ndim == 1:
        # Reshape to 2D if needed
        tensor = tensor.reshape(-1, 1)
    
    if tensor.ndim > 2:
        raise InvalidShapeError(
            f"{name} must be 1D or 2D, got {tensor.ndim}D with shape {tuple(tensor.shape)}"
        )
    
    if expected_features is not None:
        actual_features = tensor.shape[1] if tensor.ndim == 2 else 1
        if actual_features != expected_features:
            raise InvalidShapeError(
                f"{name} has {actual_features} features, expected {expected_features}"
            )


def validate_matching_shapes(
    X: torch.Tensor,
    y: torch.Tensor,
    X_name: str = "X",
    y_name: str = "y"
) -> None:
    """
    Validate that X and y have matching sample dimensions.
    
    Parameters
    ----------
    X : torch.Tensor
        Feature tensor.
    y : torch.Tensor
        Label tensor.
    X_name : str
        Name of X for error messages.
    y_name : str
        Name of y for error messages.
        
    Raises
    ------
    InvalidShapeError
        If sample dimensions don't match.
        
    Examples
    --------
    >>> X = torch.randn(100, 10)
    >>> y = torch.randn(100)
    >>> validate_matching_shapes(X, y)  # No error
    
    >>> y = torch.randn(50)
    >>> validate_matching_shapes(X, y)  # Raises InvalidShapeError
    """
    X_samples = X.shape[0] if X.ndim > 0 else 1
    y_samples = y.shape[0] if y.ndim > 0 else 1
    
    if X_samples != y_samples:
        raise InvalidShapeError(
            f"{X_name} and {y_name} must have same number of samples. "
            f"Got {X_name}: {X_samples}, {y_name}: {y_samples}"
        )


def validate_scaling_factor(scale: float, name: str = "scaling_factor") -> None:
    """
    Validate sigmoid scaling factor.
    
    Parameters
    ----------
    scale : float
        Scaling factor to validate.
    name : str
        Name for error messages.
        
    Raises
    ------
    ValueError
        If scale is not positive.
        
    Examples
    --------
    >>> validate_scaling_factor(1.0)  # No error
    >>> validate_scaling_factor(-1.0)  # Raises ValueError
    """
    if not isinstance(scale, (int, float)):
        raise ValueError(
            f"{name} must be numeric, got {type(scale).__name__}"
        )
    
    if scale <= 0:
        raise ValueError(
            f"{name} must be positive, got {scale}"
        )
