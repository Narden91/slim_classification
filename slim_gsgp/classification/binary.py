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

import logging
import torch
from typing import Dict, Optional, Protocol, TYPE_CHECKING, Union

from .metrics import calculate_binary_metrics
from .utils import apply_sigmoid
from .validators import (
    validate_binary_labels,
    validate_threshold,
    validate_scaling_factor,
    validate_tensor_shape,
    validate_matching_shapes,
)
from .exceptions import AlgorithmNotFoundError

if TYPE_CHECKING:
    from .config import ClassifierConfig

from ..main_gp import gp
from ..main_gsgp import gsgp
from ..main_slim import slim

logger = logging.getLogger(__name__)


class GPModelProtocol(Protocol):
    """Protocol defining the interface for GP-based models.
    
    This protocol ensures type safety when working with different model types.
    """
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for input features."""
        ...
    
    def print_tree_representation(self) -> None:
        """Print a representation of the model tree."""
        ...


class BinaryClassifier:
    """
    Wrapper class for binary classification using GP-based models.

    This class wraps GP, GSGP, or SLIM models to provide a consistent interface
    for binary classification tasks.
    
    Attributes
    ----------
    model : GPModelProtocol
        The underlying GP-based model.
    threshold : float
        Classification threshold (default: 0.5).
    use_sigmoid : bool
        Whether to apply sigmoid to outputs.
    sigmoid_scale : float
        Scaling factor for sigmoid function.
        
    Examples
    --------
    >>> from slim_gsgp.classification import train_binary_classifier
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train, X_val, y_val, algorithm='gp'
    ... )
    >>> predictions = classifier.predict(X_test)
    >>> metrics = classifier.evaluate(X_test, y_test)
    
    >>> # Using ClassifierConfig
    >>> from slim_gsgp.classification import ClassifierConfig
    >>> config = ClassifierConfig(threshold=0.6, sigmoid_scale=1.5)
    >>> classifier = BinaryClassifier(model, config=config)
    """

    __slots__ = ('model', 'threshold', 'use_sigmoid', 'sigmoid_scale')

    def __init__(
        self, 
        model: GPModelProtocol, 
        threshold: float = 0.5, 
        use_sigmoid: bool = True,
        sigmoid_scale: float = 1.0,
        *,
        config: Optional['ClassifierConfig'] = None
    ) -> None:
        """
        Initialize a binary classifier wrapper.

        Parameters
        ----------
        model : GPModelProtocol
            The trained GP, GSGP, or SLIM model.
        threshold : float, default=0.5
            Threshold for binary classification.
        use_sigmoid : bool, default=True
            Whether to apply sigmoid to model outputs.
            If False, classification is done based on the sign of outputs.
        sigmoid_scale : float, default=1.0
            Scaling factor for sigmoid function.
        config : ClassifierConfig, optional
            Configuration object. If provided, overrides threshold, use_sigmoid,
            and sigmoid_scale parameters.
            
        Raises
        ------
        InvalidThresholdError
            If threshold is not in (0, 1).
        ValueError
            If sigmoid_scale is not positive.
        """
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            threshold = config.threshold
            use_sigmoid = config.use_sigmoid
            sigmoid_scale = config.sigmoid_scale
        else:
            validate_threshold(threshold)
            validate_scaling_factor(sigmoid_scale)
        
        self.model = model
        self.threshold = threshold
        self.use_sigmoid = use_sigmoid
        self.sigmoid_scale = sigmoid_scale
        
        logger.debug(
            f"BinaryClassifier initialized: threshold={threshold}, "
            f"use_sigmoid={use_sigmoid}, scale={sigmoid_scale}"
        )

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Predicted class probabilities of shape (n_samples, 2).
            First column is probability of class 0, second is class 1.
            
        Examples
        --------
        >>> X_test = torch.randn(10, 5)
        >>> probs = classifier.predict_proba(X_test)
        >>> probs.shape
        torch.Size([10, 2])
        >>> torch.all((probs >= 0) & (probs <= 1))
        True
        """
        validate_tensor_shape(X, name="X")
        
        # Get raw predictions
        raw_preds = self.model.predict(X)

        # Apply transformation based on configuration
        if self.use_sigmoid:
            # Skip validation - sigmoid_scale was validated at init
            probs = apply_sigmoid(raw_preds, self.sigmoid_scale, _skip_validation=True)
        else:
            # For sign-based prediction, map output to [0,1] range
            probs = (raw_preds >= 0).float()

        # Return probabilities for both classes [P(class=0), P(class=1)]
        return torch.stack([1 - probs, probs], dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Predicted class labels (0 or 1) of shape (n_samples,).
            
        Examples
        --------
        >>> X_test = torch.randn(10, 5)
        >>> predictions = classifier.predict(X_test)
        >>> predictions.shape
        torch.Size([10])
        >>> torch.all((predictions == 0) | (predictions == 1))
        True
        """
        validate_tensor_shape(X, name="X")
        
        # Get raw predictions
        raw_preds = self.model.predict(X)
        
        if self.use_sigmoid:
            # Skip validation - sigmoid_scale was validated at init
            probs = apply_sigmoid(raw_preds, self.sigmoid_scale, _skip_validation=True)
            return (probs > self.threshold).float()
        else:
            # Sign-based prediction (negative -> 0, non-negative -> 1)
            return (raw_preds >= 0).float()

    def evaluate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> Dict[str, Union[float, int]]:
        """
        Evaluate the classifier on test data.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features).
        y : torch.Tensor
            True labels (0 or 1) of shape (n_samples,).

        Returns
        -------
        Dict[str, Union[float, int]]
            Dictionary containing evaluation metrics including:
            - accuracy, precision, recall, f1, specificity
            - true_positives, true_negatives, false_positives, false_negatives
            
        Raises
        ------
        InvalidShapeError
            If X and y have mismatched dimensions.
        InvalidLabelError
            If y contains non-binary values.
            
        Examples
        --------
        >>> metrics = classifier.evaluate(X_test, y_test)
        >>> print(f\"Accuracy: {metrics['accuracy']:.4f}\")
        Accuracy: 0.9234
        >>> print(f\"F1 Score: {metrics['f1']:.4f}\")
        F1 Score: 0.9156
        """
        validate_tensor_shape(X, name="X")
        validate_tensor_shape(y, name="y")
        validate_matching_shapes(X, y, "X", "y")
        validate_binary_labels(y, name="y")
        
        # Make predictions
        y_pred = self.predict(X)

        # Calculate and return metrics
        return calculate_binary_metrics(y, y_pred)

    def print_tree_representation(self) -> None:
        """
        Print the tree representation of the underlying model.
        
        This method delegates to the model's tree representation method if available.
        """
        if hasattr(self.model, 'print_tree_representation'):
            self.model.print_tree_representation()
        else:
            logger.warning(
                f"Tree representation not available for model type {type(self.model).__name__}"
            )
            print(f"Tree representation not available for {type(self.model).__name__}")


def train_binary_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    algorithm: str = 'gp',
    use_sigmoid: bool = True,
    sigmoid_scale: float = 1.0,
    threshold: float = 0.5,
    fitness_function: str = 'binary_rmse',
    **kwargs
) -> BinaryClassifier:
    """
    Train a binary classifier using GP-based methods.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features of shape (n_samples, n_features).
    y_train : torch.Tensor
        Training labels (0 or 1) of shape (n_samples,).
    X_val : torch.Tensor, optional
        Validation features of shape (n_samples, n_features).
    y_val : torch.Tensor, optional
        Validation labels (0 or 1) of shape (n_samples,).
    algorithm : str, default='gp'
        Algorithm to use: 'gp', 'gsgp', or 'slim'.
    use_sigmoid : bool, default=True
        Whether to use sigmoid activation.
    sigmoid_scale : float, default=1.0
        Scaling factor for sigmoid.
    threshold : float, default=0.5
        Threshold for binary classification.
    fitness_function : str, default='binary_rmse'
        Fitness function to use. Should be one of the registered
        binary classification fitness functions.
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    BinaryClassifier
        Trained binary classifier.
        
    Raises
    ------
    InvalidLabelError
        If labels are not binary (0 or 1).
    AlgorithmNotFoundError
        If the specified algorithm is not supported.
    InvalidShapeError
        If input shapes are incompatible.
    InvalidThresholdError
        If threshold is not in (0, 1).
        
    Examples
    --------
    >>> import torch
    >>> from slim_gsgp.classification import train_binary_classifier
    >>> X_train = torch.randn(100, 10)
    >>> y_train = torch.randint(0, 2, (100,)).float()
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train, 
    ...     algorithm='gp',
    ...     pop_size=50,
    ...     n_iter=10
    ... )
    >>> predictions = classifier.predict(X_train)
    
    >>> # With validation set
    >>> X_val = torch.randn(30, 10)
    >>> y_val = torch.randint(0, 2, (30,)).float()
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train, X_val, y_val,
    ...     algorithm='slim',
    ...     slim_version='SLIM+ABS'
    ... )
    >>> metrics = classifier.evaluate(X_val, y_val)
    >>> print(f\"Validation Accuracy: {metrics['accuracy']:.4f}\")
    """
    # Input validation
    validate_tensor_shape(X_train, name="X_train")
    validate_tensor_shape(y_train, name="y_train")
    validate_matching_shapes(X_train, y_train, "X_train", "y_train")
    validate_binary_labels(y_train, name="y_train")
    validate_threshold(threshold)
    validate_scaling_factor(sigmoid_scale)
    
    if X_val is not None:
        validate_tensor_shape(X_val, expected_features=X_train.shape[1], name="X_val")
    
    if y_val is not None:
        validate_tensor_shape(y_val, name="y_val")
        if X_val is not None:
            validate_matching_shapes(X_val, y_val, "X_val", "y_val")
        validate_binary_labels(y_val, name="y_val")
    
    logger.info(
        f"Training binary classifier with algorithm={algorithm}, "
        f"fitness={fitness_function}, X_train shape={tuple(X_train.shape)}"
    )

    # Convert targets to float
    y_train = y_train.float()
    if y_val is not None:
        y_val = y_val.float()

    # Select and train the algorithm
    algorithm_lower = algorithm.lower()
    
    if algorithm_lower == 'gp':
        model = gp(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_val, 
            y_test=y_val,
            fitness_function=fitness_function, 
            **kwargs
        )
    elif algorithm_lower == 'gsgp':
        # Ensure reconstruct=True for GSGP to enable predict method
        if 'reconstruct' not in kwargs:
            kwargs['reconstruct'] = True

        model = gsgp(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_val, 
            y_test=y_val,
            fitness_function=fitness_function, 
            **kwargs
        )
    elif algorithm_lower == 'slim':
        model = slim(
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_val, 
            y_test=y_val,
            fitness_function=fitness_function, 
            **kwargs
        )
    else:
        raise AlgorithmNotFoundError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Supported algorithms are: 'gp', 'gsgp', 'slim'"
        )
    
    logger.info(f"Successfully trained {algorithm} model")

    # Return wrapped model
    return BinaryClassifier(
        model, 
        threshold=threshold, 
        use_sigmoid=use_sigmoid,
        sigmoid_scale=sigmoid_scale
    )