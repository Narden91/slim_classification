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

import torch
from typing import Callable


def modified_sigmoid(scaling_factor: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a scaled sigmoid function for transforming outputs.

    Parameters
    ----------
    scaling_factor : float
        Controls the steepness of the sigmoid curve. Higher values make the transition
        between 0 and 1 more abrupt.

    Returns
    -------
    Callable
        A sigmoid function with specified scaling factor.
    """

    def sigmoid_func(tensor: torch.Tensor) -> torch.Tensor:
        return torch.div(
            1.0,
            torch.add(1.0, torch.exp(torch.mul(-scaling_factor, tensor)))
        )

    return sigmoid_func


def binary_threshold_transform(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Transforms a tensor to binary values based on a threshold.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to transform.
    threshold : float
        The threshold value for binarization (default: 0.5)

    Returns
    -------
    torch.Tensor
        Tensor containing 0s and 1s based on threshold comparison.
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
    """
    return (tensor >= 0).float()


def create_binary_fitness_function(base_fitness_func: Callable,
                                   transform_func: Callable = None,
                                   scaling_factor: float = 1.0) -> Callable:
    """
    Creates a fitness function for binary classification by wrapping a base fitness function.

    Parameters
    ----------
    base_fitness_func : Callable
        Base fitness function (e.g., rmse, mse).
    transform_func : Callable, optional
        Function to transform outputs before fitness calculation.
        If None, a sigmoid function with the specified scaling factor is used.
    scaling_factor : float
        Scaling factor for the sigmoid function (if transform_func is None).

    Returns
    -------
    Callable
        The wrapped fitness function for binary classification.
    """
    if transform_func is None:
        transform_func = modified_sigmoid(scaling_factor)

    def binary_fitness(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Ensure y_true is float type
        y_true = y_true.float() if isinstance(y_true, torch.Tensor) else torch.tensor(y_true, dtype=torch.float32)

        # Apply transformation to prediction
        transformed_pred = transform_func(y_pred)

        # Calculate fitness using the base function
        return base_fitness_func(y_true, transformed_pred)

    return binary_fitness


def register_classification_fitness_functions():
    """
    Register binary classification fitness functions with the fitness function options dictionaries.

    This function adds binary classification fitness functions to the fitness function
    options dictionaries in the GP, GSGP, and SLIM config modules.

    Returns
    -------
    bool
        True if registration was successful, False otherwise.
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

        return True
    except ImportError as e:
        print(f"Warning: Could not register classification fitness functions: {str(e)}")
        return False