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
Binary Classification Utilities for GP-based methods (GP, GSGP, SLIM).

This module provides functions for adapting regression-based GP variants to
binary classification problems.
"""

import torch
from typing import Callable, Union, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Type alias for model instances
GPModel = Any  # Any of GP, GSGP, or SLIM models


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


class BinaryClassifier:
    """
    Wrapper class for binary classification using GP-based models.

    This class wraps GP, GSGP, or SLIM models to provide a consistent interface
    for binary classification tasks.
    """

    def __init__(self, model: GPModel, threshold: float = 0.5, use_sigmoid: bool = True,
                 sigmoid_scale: float = 1.0):
        """
        Initialize a binary classifier wrapper.

        Parameters
        ----------
        model : GPModel
            The trained GP, GSGP, or SLIM model.
        threshold : float
            Threshold for binary classification (default: 0.5).
        use_sigmoid : bool
            Whether to apply sigmoid to model outputs (default: True).
            If False, classification is done based on the sign of outputs.
        sigmoid_scale : float
            Scaling factor for sigmoid function (default: 1.0).
        """
        self.model = model
        self.threshold = threshold
        self.use_sigmoid = use_sigmoid
        self.sigmoid_scale = sigmoid_scale

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Predicted class probabilities (shape: [n_samples, 2]).
        """
        # Get raw predictions
        raw_preds = self.model.predict(X)

        # Apply sigmoid if needed
        if self.use_sigmoid:
            probs = modified_sigmoid(self.sigmoid_scale)(raw_preds)
        else:
            # For sign-based prediction, we'll map output to [0,1] range
            probs = binary_sign_transform(raw_preds)

        # Return probabilities for both classes
        return torch.stack([1 - probs, probs], dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Predicted class labels (0 or 1).
        """
        if self.use_sigmoid:
            # Get raw predictions and apply sigmoid
            raw_preds = self.model.predict(X)
            probs = modified_sigmoid(self.sigmoid_scale)(raw_preds)
            return (probs > self.threshold).float()
        else:
            # Sign-based prediction (negative -> 0, non-negative -> 1)
            raw_preds = self.model.predict(X)
            return binary_sign_transform(raw_preds)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Evaluate the classifier on test data.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        y : torch.Tensor
            True labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """
        # Ensure y is a numpy array
        y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        # Make predictions
        y_pred = self.predict(X)
        y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_np, y_pred_np),
            'precision': precision_score(y_np, y_pred_np, zero_division=0),
            'recall': recall_score(y_np, y_pred_np, zero_division=0),
            'f1': f1_score(y_np, y_pred_np, zero_division=0)
        }

        return metrics

    def print_tree_representation(self):
        """
        Print the tree representation.
        """
        self.model.print_tree_representation()


def register_binary_fitness_functions():
    """
    Register binary fitness functions with the fitness function options dictionaries.

    This function adds binary classification fitness functions to the fitness function
    options dictionaries in the GP, GSGP, and SLIM config modules.

    Returns
    -------
    None
    """
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


def train_binary_classifier(X_train, y_train, X_val=None, y_val=None, algorithm='gp',
                            use_sigmoid=True, sigmoid_scale=1.0, threshold=0.5,
                            fitness_function='binary_rmse', **kwargs):
    """
    Train a binary classifier using GP-based methods.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels (0 or 1).
    X_val : torch.Tensor, optional
        Validation features.
    y_val : torch.Tensor, optional
        Validation labels.
    algorithm : str
        Algorithm to use: 'gp', 'gsgp', or 'slim' (default: 'gp').
    use_sigmoid : bool
        Whether to use sigmoid activation (default: True).
    sigmoid_scale : float
        Scaling factor for sigmoid (default: 1.0).
    threshold : float
        Threshold for binary classification (default: 0.5).
    fitness_function : str
        Fitness function to use (default: 'binary_rmse').
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    BinaryClassifier
        Trained binary classifier.
    """
    # Register fitness functions
    register_binary_fitness_functions()

    # Ensure binary labels
    if len(torch.unique(y_train)) > 2:
        raise ValueError("Training labels must be binary (0 or 1).")

    # Convert targets to float
    y_train = y_train.float()
    if y_val is not None:
        y_val = y_val.float()

    # Select the algorithm
    if algorithm.lower() == 'gp':
        from slim_gsgp.main_gp import gp
        model = gp(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                   fitness_function=fitness_function, **kwargs)
    elif algorithm.lower() == 'gsgp':
        from slim_gsgp.main_gsgp import gsgp
        # Ensure reconstruct=True for GSGP to enable predict method
        kwargs['reconstruct'] = True
        model = gsgp(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                     fitness_function=fitness_function, **kwargs)
    elif algorithm.lower() == 'slim':
        from slim_gsgp.main_slim import slim
        model = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                     fitness_function=fitness_function, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'gp', 'gsgp', or 'slim'.")

    # Return wrapped model
    return BinaryClassifier(model, threshold=threshold, use_sigmoid=use_sigmoid,
                            sigmoid_scale=sigmoid_scale)