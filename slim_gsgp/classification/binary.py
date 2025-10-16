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
Binary Classification module for SLIM-GSGP.

This module provides tools for binary classification using GP, GSGP, or SLIM models.
"""

import torch
from typing import Any, Dict, Union

from .metrics import calculate_binary_metrics
from .utils import modified_sigmoid, binary_sign_transform, register_classification_fitness_functions

# Register fitness functions once at module import
_register_success = register_classification_fitness_functions()
from ..main_gp import gp
from ..main_gsgp import gsgp
from ..main_slim import slim

# Type alias for model instances
GPModel = Any  # Any of GP, GSGP, or SLIM models


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

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
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
        Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        return calculate_binary_metrics(y, y_pred)

    def print_tree_representation(self):
        """
        Print the tree representation of the underlying model.
        """
        if hasattr(self.model, 'print_tree_representation'):
            self.model.print_tree_representation()
        else:
            print("Tree representation not available for this model type")


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
    # Fitness functions are registered at module import
    # register_classification_fitness_functions()

    # Ensure binary labels
    if len(torch.unique(y_train)) > 2:
        raise ValueError("Training labels must be binary (0 or 1).")

    # Convert targets to float
    y_train = y_train.float()
    if y_val is not None:
        y_val = y_val.float()

    # Select the algorithm
    if algorithm.lower() == 'gp':
        model = gp(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                   fitness_function=fitness_function, **kwargs)
    elif algorithm.lower() == 'gsgp':
        # Ensure reconstruct=True for GSGP to enable predict method
        if 'reconstruct' not in kwargs:
            kwargs['reconstruct'] = True

        # GSGP supports max_depth, don't remove it
        # kwargs = {k: v for k, v in kwargs.items() if k != 'max_depth'}

        model = gsgp(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                     fitness_function=fitness_function, **kwargs)
    elif algorithm.lower() == 'slim':
        model = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                     fitness_function=fitness_function, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'gp', 'gsgp', or 'slim'.")

    # Return wrapped model
    return BinaryClassifier(model, threshold=threshold, use_sigmoid=use_sigmoid,
                            sigmoid_scale=sigmoid_scale)