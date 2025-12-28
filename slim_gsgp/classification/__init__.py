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
Classification module for SLIM-GSGP.

This module provides tools for binary classification using the SLIM-GSGP framework,
wrapping underlying algorithms (GP, GSGP, SLIM) with appropriate interfaces.

Main Components
---------------
- BinaryClassifier: Wrapper class for binary classification
- train_binary_classifier: Function to train a binary classifier
- ClassifierConfig: Configuration dataclass for classifier parameters

Example
-------
>>> from slim_gsgp.classification import train_binary_classifier
>>> 
>>> # Train a classifier
>>> classifier = train_binary_classifier(
...     X_train, y_train, X_val, y_val,
...     algorithm='gp',
...     fitness_function='binary_rmse'
... )
>>> 
>>> # Evaluate the model
>>> metrics = classifier.evaluate(X_test, y_test)
>>> print(f"Accuracy: {metrics['accuracy']:.4f}")
"""

import logging

from .binary import BinaryClassifier, train_binary_classifier
from .config import ClassifierConfig, TrainingConfig
from .metrics import calculate_binary_metrics, save_metrics_to_csv
from .results import BinaryMetrics
from .utils import (
    register_classification_fitness_functions,
    apply_sigmoid,
    create_binary_fitness_function,
)
from .exceptions import (
    ClassificationError,
    InvalidLabelError,
    AlgorithmNotFoundError,
    InvalidThresholdError,
    InvalidShapeError,
    FitnessRegistrationError,
    InvalidScalingFactorError,
)
from .factories import (
    AlgorithmFactory,
    get_default_factory,
    create_algorithm,
    register_algorithm,
)
from .strategies import (
    PredictionStrategy,
    SigmoidStrategy,
    SignBasedStrategy,
    SoftmaxStrategy,
    PredictionContext,
    get_strategy,
    register_strategy,
)

# Backward compatibility exports (deprecated)
# Note: modified_sigmoid is deprecated and no longer exported.
# Use apply_sigmoid() instead for better performance.
from .utils import (
    binary_threshold_transform,
    binary_sign_transform,
)

logger = logging.getLogger(__name__)

# Auto-register fitness functions at module import
try:
    register_classification_fitness_functions()
except FitnessRegistrationError as e:
    logger.warning(f"Auto-registration failed: {e}")
    logger.warning("Call register_classification_fitness_functions() explicitly before training")

__all__ = [
    # Main API
    'BinaryClassifier',
    'train_binary_classifier',
    
    # Configuration
    'ClassifierConfig',
    'TrainingConfig',
    
    # Metrics
    'BinaryMetrics',
    'calculate_binary_metrics',
    'save_metrics_to_csv',
    
    # Utilities
    'apply_sigmoid',
    'register_classification_fitness_functions',
    'create_binary_fitness_function',
    
    # Factory Pattern
    'AlgorithmFactory',
    'get_default_factory',
    'create_algorithm',
    'register_algorithm',
    
    # Strategy Pattern
    'PredictionStrategy',
    'SigmoidStrategy',
    'SignBasedStrategy',
    'SoftmaxStrategy',
    'PredictionContext',
    'get_strategy',
    'register_strategy',
    
    # Exceptions
    'ClassificationError',
    'InvalidLabelError',
    'AlgorithmNotFoundError',
    'InvalidThresholdError',
    'InvalidShapeError',
    'FitnessRegistrationError',
    'InvalidScalingFactorError',
    
    # Backward compatibility transforms (binary_threshold_transform, binary_sign_transform)
    # Note: modified_sigmoid has been removed (use apply_sigmoid instead)
    'binary_threshold_transform',
    'binary_sign_transform',
]
