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
from .results import BinaryMetrics
from .utils import apply_sigmoid
from .validators import (
    validate_binary_labels,
    validate_threshold,
    validate_scaling_factor,
    validate_tensor_shape,
    validate_matching_shapes,
)
from .exceptions import AlgorithmNotFoundError
from .strategies import (
    PredictionStrategy,
    SigmoidStrategy,
    SignBasedStrategy,
    PredictionContext,
    get_strategy,
)
from .factories import AlgorithmFactory, get_default_factory
from .config import ClassifierConfig, TrainingConfig

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
    for binary classification tasks. Supports both traditional configuration
    and Strategy Pattern for prediction behavior.
    
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
    _strategy : PredictionStrategy
        Internal strategy used for predictions (auto-created from config).
        
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
    
    >>> # Using Strategy Pattern directly
    >>> from slim_gsgp.classification.strategies import SigmoidStrategy
    >>> strategy = SigmoidStrategy(scale=2.0, threshold=0.6)
    >>> classifier = BinaryClassifier(model, strategy=strategy)
    """

    __slots__ = ('model', 'threshold', 'use_sigmoid', 'sigmoid_scale', '_strategy')

    def __init__(
        self, 
        model: GPModelProtocol, 
        threshold: float = 0.5, 
        use_sigmoid: bool = True,
        sigmoid_scale: float = 1.0,
        *,
        config: Optional['ClassifierConfig'] = None,
        strategy: Optional[PredictionStrategy] = None
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
        strategy : PredictionStrategy, optional
            Prediction strategy to use. If provided, overrides all other
            prediction-related parameters (threshold, use_sigmoid, sigmoid_scale).
            
        Raises
        ------
        InvalidThresholdError
            If threshold is not in (0, 1).
        ValueError
            If sigmoid_scale is not positive.
        """
        self.model = model
        
        # Priority: strategy > config > individual parameters
        if strategy is not None:
            # Use the provided strategy directly
            self._strategy = strategy
            # Extract parameters for backward compatibility attributes
            if isinstance(strategy, SigmoidStrategy):
                self.threshold = strategy.threshold
                self.use_sigmoid = True
                self.sigmoid_scale = strategy.scale
            elif isinstance(strategy, SignBasedStrategy):
                self.threshold = 0.5  # Not used for sign-based
                self.use_sigmoid = False
                self.sigmoid_scale = 1.0
            else:
                # Custom strategy - use defaults for attributes
                self.threshold = 0.5
                self.use_sigmoid = True
                self.sigmoid_scale = 1.0
        elif config is not None:
            threshold = config.threshold
            use_sigmoid = config.use_sigmoid
            sigmoid_scale = config.sigmoid_scale
            self.threshold = threshold
            self.use_sigmoid = use_sigmoid
            self.sigmoid_scale = sigmoid_scale
            self._strategy = self._create_strategy()
        else:
            validate_threshold(threshold)
            validate_scaling_factor(sigmoid_scale)
            self.threshold = threshold
            self.use_sigmoid = use_sigmoid
            self.sigmoid_scale = sigmoid_scale
            self._strategy = self._create_strategy()
        
        logger.debug(
            f"BinaryClassifier initialized: threshold={self.threshold}, "
            f"use_sigmoid={self.use_sigmoid}, scale={self.sigmoid_scale}, "
            f"strategy={self._strategy.name}"
        )
    
    def __repr__(self) -> str:
        """
        Return detailed string representation of the classifier.
        
        Returns
        -------
        str
            String representation showing configuration.
            
        Examples
        --------
        >>> print(repr(classifier))
        BinaryClassifier(threshold=0.5, use_sigmoid=True, sigmoid_scale=1.0, strategy=sigmoid(scale=1.0, threshold=0.5))
        """
        return (
            f"BinaryClassifier("
            f"threshold={self.threshold}, "
            f"use_sigmoid={self.use_sigmoid}, "
            f"sigmoid_scale={self.sigmoid_scale}, "
            f"strategy={self._strategy.name})"
        )
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality with another BinaryClassifier.
        
        Parameters
        ----------
        other : object
            Object to compare with.
            
        Returns
        -------
        bool
            True if classifiers have same configuration (model comparison excluded).
            
        Examples
        --------
        >>> classifier1 = BinaryClassifier(model, threshold=0.5)
        >>> classifier2 = BinaryClassifier(model, threshold=0.5)
        >>> classifier1 == classifier2
        True
        """
        if not isinstance(other, BinaryClassifier):
            return NotImplemented
        
        return (
            self.threshold == other.threshold and
            self.use_sigmoid == other.use_sigmoid and
            self.sigmoid_scale == other.sigmoid_scale and
            self._strategy.name == other._strategy.name
        )
    
    @classmethod
    def from_config(
        cls,
        model: GPModelProtocol,
        config: 'ClassifierConfig'
    ) -> 'BinaryClassifier':
        """
        Create a BinaryClassifier from a ClassifierConfig.
        
        Parameters
        ----------
        model : GPModelProtocol
            The trained GP-based model.
        config : ClassifierConfig
            Configuration for the classifier.
            
        Returns
        -------
        BinaryClassifier
            New classifier instance.
            
        Examples
        --------
        >>> config = ClassifierConfig(threshold=0.6, sigmoid_scale=2.0)
        >>> classifier = BinaryClassifier.from_config(model, config)
        >>> classifier.threshold
        0.6
        """
        return cls(model, config=config)
    
    def with_strategy(self, strategy: PredictionStrategy) -> 'BinaryClassifier':
        """
        Create a new classifier with a different prediction strategy.
        
        This method returns a new instance with the same model but
        different prediction strategy, following immutable design.
        
        Parameters
        ----------
        strategy : PredictionStrategy
            New prediction strategy to use.
            
        Returns
        -------
        BinaryClassifier
            New classifier instance with updated strategy.
            
        Examples
        --------
        >>> from slim_gsgp.classification.strategies import SignBasedStrategy
        >>> new_classifier = classifier.with_strategy(SignBasedStrategy())
        >>> new_classifier.strategy.name
        'sign-based'
        """
        return BinaryClassifier(self.model, strategy=strategy)
    
    def _create_strategy(self) -> PredictionStrategy:
        """Create a prediction strategy from current configuration."""
        if self.use_sigmoid:
            return SigmoidStrategy(scale=self.sigmoid_scale, threshold=self.threshold)
        else:
            return SignBasedStrategy()
    
    @property
    def strategy(self) -> PredictionStrategy:
        """Get the current prediction strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, new_strategy: PredictionStrategy) -> None:
        """
        Set a new prediction strategy.
        
        Parameters
        ----------
        new_strategy : PredictionStrategy
            The new strategy to use.
            
        Examples
        --------
        >>> from slim_gsgp.classification.strategies import SignBasedStrategy
        >>> classifier.strategy = SignBasedStrategy()
        """
        self._strategy = new_strategy
        logger.debug(f"Strategy changed to: {new_strategy.name}")

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
        
        # Get raw predictions and delegate to strategy
        raw_preds = self.model.predict(X)
        return self._strategy.predict_proba(raw_preds)

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
        
        # Get raw predictions and delegate to strategy
        raw_preds = self.model.predict(X)
        return self._strategy.predict(raw_preds)

    def evaluate(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> BinaryMetrics:
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
        BinaryMetrics
            Dataclass containing evaluation metrics including:
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
        >>> print(f\"Accuracy: {metrics.accuracy:.4f}\")
        Accuracy: 0.9234
        >>> print(f\"F1 Score: {metrics.f1:.4f}\")
        F1 Score: 0.9156
        >>> # Dict-like access also works
        >>> print(f\"Accuracy: {metrics['accuracy']:.4f}\")
        Accuracy: 0.9234
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
    algorithm: Optional[str] = None,
    use_sigmoid: Optional[bool] = None,
    sigmoid_scale: Optional[float] = None,
    threshold: Optional[float] = None,
    fitness_function: Optional[str] = None,
    *,
    training_config: Optional[TrainingConfig] = None,
    classifier_config: Optional[ClassifierConfig] = None,
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
    algorithm : str, optional
        Algorithm to use: 'gp', 'gsgp', or 'slim'.
        If None and training_config is provided, uses training_config.algorithm.
        Otherwise defaults to 'gp'.
    use_sigmoid : bool, optional
        Whether to use sigmoid activation. Defaults to True.
        Overridden by classifier_config if provided.
    sigmoid_scale : float, optional
        Scaling factor for sigmoid. Defaults to 1.0.
        Overridden by classifier_config if provided.
    threshold : float, optional
        Threshold for binary classification. Defaults to 0.5.
        Overridden by classifier_config if provided.
    fitness_function : str, optional
        Fitness function to use. Defaults to 'binary_rmse'.
        Overridden by training_config if provided.
    training_config : TrainingConfig, optional
        Training configuration object. If provided, its values take precedence
        over individual parameters (algorithm, fitness_function).
    classifier_config : ClassifierConfig, optional
        Classifier configuration object. If provided, its values take precedence
        over individual parameters (threshold, use_sigmoid, sigmoid_scale).
    **kwargs
        Additional arguments passed to the underlying algorithm (e.g., pop_size, n_iter).

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
    >>> # Traditional parameter passing
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train, 
    ...     algorithm='gp',
    ...     pop_size=50,
    ...     n_iter=10
    ... )
    
    >>> # Using TrainingConfig (recommended for complex setups)
    >>> from slim_gsgp.classification import TrainingConfig
    >>> train_cfg = TrainingConfig(algorithm='slim', pop_size=100, n_iter=50)
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train,
    ...     training_config=train_cfg
    ... )
    
    >>> # Using both configs
    >>> from slim_gsgp.classification import ClassifierConfig
    >>> clf_cfg = ClassifierConfig(threshold=0.6, sigmoid_scale=2.0)
    >>> classifier = train_binary_classifier(
    ...     X_train, y_train,
    ...     training_config=train_cfg,
    ...     classifier_config=clf_cfg
    ... )
    
    Notes
    -----
    Priority order for configuration:
    1. Explicit config objects (training_config, classifier_config)
    2. Individual keyword arguments
    3. Default values
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
    >>> print(f\"Validation Accuracy: {metrics.accuracy:.4f}\")
    """
    # Resolve configuration with priority: config objects > individual params > defaults
    
    # Training parameters
    if training_config is not None:
        final_algorithm = training_config.algorithm
        final_fitness = training_config.fitness_function
        # Add training config params to kwargs if not already present
        if 'pop_size' not in kwargs:
            kwargs['pop_size'] = training_config.pop_size
        if 'n_iter' not in kwargs:
            kwargs['n_iter'] = training_config.n_iter
        if 'seed' not in kwargs and training_config.seed is not None:
            kwargs['seed'] = training_config.seed
        if 'verbose' not in kwargs:
            kwargs['verbose'] = training_config.verbose
    else:
        final_algorithm = algorithm if algorithm is not None else 'gp'
        final_fitness = fitness_function if fitness_function is not None else 'binary_rmse'
    
    # Classifier parameters
    if classifier_config is not None:
        final_threshold = classifier_config.threshold
        final_use_sigmoid = classifier_config.use_sigmoid
        final_sigmoid_scale = classifier_config.sigmoid_scale
    else:
        final_threshold = threshold if threshold is not None else 0.5
        final_use_sigmoid = use_sigmoid if use_sigmoid is not None else True
        final_sigmoid_scale = sigmoid_scale if sigmoid_scale is not None else 1.0
    
    # Input validation
    validate_tensor_shape(X_train, name="X_train")
    validate_tensor_shape(y_train, name="y_train")
    validate_matching_shapes(X_train, y_train, "X_train", "y_train")
    validate_binary_labels(y_train, name="y_train")
    validate_threshold(final_threshold)
    validate_scaling_factor(final_sigmoid_scale)
    
    if X_val is not None:
        validate_tensor_shape(X_val, expected_features=X_train.shape[1], name="X_val")
    
    if y_val is not None:
        validate_tensor_shape(y_val, name="y_val")
        if X_val is not None:
            validate_matching_shapes(X_val, y_val, "X_val", "y_val")
        validate_binary_labels(y_val, name="y_val")
    
    logger.info(
        f"Training binary classifier with algorithm={final_algorithm}, "
        f"fitness={final_fitness}, X_train shape={tuple(X_train.shape)}"
    )

    # Convert targets to float
    y_train = y_train.float()
    if y_val is not None:
        y_val = y_val.float()

    # Use factory to create the model
    factory = get_default_factory()
    model = factory.create(
        algorithm=final_algorithm,
        X_train=X_train,
        y_train=y_train,
        X_test=X_val,
        y_test=y_val,
        fitness_function=final_fitness,
        **kwargs
    )
    
    logger.info(f"Successfully trained {final_algorithm} model")

    # Return wrapped model
    return BinaryClassifier(
        model, 
        threshold=final_threshold, 
        use_sigmoid=final_use_sigmoid,
        sigmoid_scale=final_sigmoid_scale
    )