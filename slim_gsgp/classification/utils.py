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
Utility functions for classification with SLIM-GSGP.

This module provides helper functions for working with
classification tasks across GP, GSGP, and SLIM models.
"""

import logging
import torch
from typing import Callable, Dict

from .exceptions import FitnessRegistrationError
from .validators import validate_scaling_factor

logger = logging.getLogger(__name__)


def apply_sigmoid(tensor: torch.Tensor, scaling_factor: float = 1.0) -> torch.Tensor:
    """
    Apply scaled sigmoid transformation to tensor.
    
    This function directly applies sigmoid without creating closures,
    improving performance by avoiding unnecessary function objects.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to transform.
    scaling_factor : float, default=1.0
        Controls the steepness of the sigmoid curve. Higher values make the transition
        between 0 and 1 more abrupt. Must be positive.

    Returns
    -------
    torch.Tensor
        Transformed tensor with values in (0, 1).
        
    Raises
    ------
    ValueError
        If scaling_factor is not positive.
        
    Examples
    --------
    >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    >>> apply_sigmoid(x, scaling_factor=1.0)
    tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
    
    >>> apply_sigmoid(x, scaling_factor=2.0)  # Steeper curve
    tensor([0.0180, 0.1192, 0.5000, 0.8808, 0.9820])
    """
    validate_scaling_factor(scaling_factor)
    return torch.sigmoid(scaling_factor * tensor)


def modified_sigmoid(scaling_factor: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    DEPRECATED: Use apply_sigmoid() instead.
    
    Creates a scaled sigmoid function for transforming outputs.
    This function is kept for backward compatibility but creates unnecessary closures.

    Parameters
    ----------
    scaling_factor : float
        Controls the steepness of the sigmoid curve.

    Returns
    -------
    Callable
        A sigmoid function with specified scaling factor.
    """
    import warnings
    warnings.warn(
        "modified_sigmoid() is deprecated and creates unnecessary closures. "
        "Use apply_sigmoid() instead for better performance.",
        DeprecationWarning,
        stacklevel=2
    )
    
    def sigmoid_func(tensor: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(scaling_factor * tensor)

    return sigmoid_func


def binary_threshold_transform(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Transforms a tensor to binary values based on a threshold.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to transform.
    threshold : float, default=0.5
        The threshold value for binarization.

    Returns
    -------
    torch.Tensor
        Tensor containing 0s and 1s based on threshold comparison.
        
    Examples
    --------
    >>> probs = torch.tensor([0.1, 0.4, 0.5, 0.6, 0.9])
    >>> binary_threshold_transform(probs, threshold=0.5)
    tensor([0., 0., 1., 1., 1.])
    """
    return (tensor >= threshold).float()


def binary_sign_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transforms a tensor to binary values based on sign.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to transform.

    Returns
    -------
    torch.Tensor
        Tensor containing 0s and 1s (0 for negative, 1 for non-negative).
        
    Examples
    --------
    >>> x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    >>> binary_sign_transform(x)
    tensor([0., 0., 1., 1., 1.])
    """
    return (tensor >= 0).float()


def create_binary_fitness_function(
    base_fitness_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    transform_func: Callable[[torch.Tensor], torch.Tensor] = None,
    scaling_factor: float = 1.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Creates a fitness function for binary classification by wrapping a base fitness function.

    Parameters
    ----------
    base_fitness_func : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Base fitness function (e.g., rmse, mse).
    transform_func : Callable[[torch.Tensor], torch.Tensor], optional
        Function to transform outputs before fitness calculation.
        If None, sigmoid with the specified scaling factor is used.
    scaling_factor : float, default=1.0
        Scaling factor for the sigmoid function (if transform_func is None).

    Returns
    -------
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The wrapped fitness function for binary classification.
        
    Raises
    ------
    ValueError
        If scaling_factor is not positive.
        
    Examples
    --------
    >>> from slim_gsgp.evaluators.fitness_functions import rmse
    >>> binary_rmse = create_binary_fitness_function(rmse)
    >>> y_true = torch.tensor([0., 1., 1., 0.])
    >>> y_pred = torch.tensor([-2., 1., 2., -1.])
    >>> loss = binary_rmse(y_true, y_pred)
    """
    validate_scaling_factor(scaling_factor)
    
    # Pre-compute transformation if using default
    use_default_sigmoid = transform_func is None
    
    def binary_fitness(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Ensure y_true is float type (single conversion)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        elif y_true.dtype != torch.float32:
            y_true = y_true.float()

        # Apply transformation to prediction
        if use_default_sigmoid:
            # Direct sigmoid application - more efficient than closure
            transformed_pred = torch.sigmoid(scaling_factor * y_pred)
        else:
            transformed_pred = transform_func(y_pred)

        # Calculate fitness using the base function
        return base_fitness_func(y_true, transformed_pred)

    return binary_fitness


def register_classification_fitness_functions() -> bool:
    """
    Register binary classification fitness functions with the fitness function options dictionaries.

    This function adds binary classification fitness functions to the fitness function
    options dictionaries in the GP, GSGP, and SLIM config modules.

    Returns
    -------
    bool
        True if registration was successful.
        
    Raises
    ------
    FitnessRegistrationError
        If required modules cannot be imported or registration fails.
        
    Examples
    --------
    >>> success = register_classification_fitness_functions()
    >>> assert success is True
    """
    try:
        # Import fitness function dictionaries
        from slim_gsgp.config.gp_config import fitness_function_options as gp_fitness
        from slim_gsgp.config.gsgp_config import fitness_function_options as gsgp_fitness
        from slim_gsgp.config.slim_config import fitness_function_options as slim_fitness

        # Import base fitness functions
        from slim_gsgp.evaluators.fitness_functions import rmse, mse, mae

        # Create binary versions of common fitness functions
        binary_rmse = create_binary_fitness_function(rmse)
        binary_mse = create_binary_fitness_function(mse)
        binary_mae = create_binary_fitness_function(mae)

        # Register with each dictionary
        for fitness_dict in [gp_fitness, gsgp_fitness, slim_fitness]:
            fitness_dict['binary_rmse'] = binary_rmse
            fitness_dict['binary_mse'] = binary_mse
            fitness_dict['binary_mae'] = binary_mae

        logger.info("Successfully registered binary classification fitness functions")
        return True
    except ImportError as e:
        error_msg = f"Could not register classification fitness functions: {str(e)}"
        logger.error(error_msg)
        raise FitnessRegistrationError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error during fitness function registration: {str(e)}"
        logger.error(error_msg)
        raise FitnessRegistrationError(error_msg) from e