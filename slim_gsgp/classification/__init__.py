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

This module provides tools for binary and multiclass classification
using the SLIM-GSGP framework, wrapping underlying algorithms (GP, GSGP, SLIM)
with appropriate interfaces for classification tasks.

Components:
- BinaryClassifier: Wrapper class for binary classification
- train_binary_classifier: Function to train a binary classifier
- register_classification_fitness_functions: Utility to register fitness functions

Example:
    >>> from slim_gsgp.classification import train_binary_classifier, register_classification_fitness_functions
    >>> # Register fitness functions
    >>> register_classification_fitness_functions()
    >>> # Train a classifier
    >>> model = train_binary_classifier(
    ...     X_train, y_train, X_val, y_val,
    ...     algorithm='gp',
    ...     fitness_function='binary_rmse',
    ...     dataset_name='breast_cancer'
    ... )
    >>> # Evaluate the model
    >>> metrics = model.evaluate(X_test, y_test)
    >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
"""

from .binary import BinaryClassifier, train_binary_classifier
from .metrics import calculate_binary_metrics, save_metrics_to_csv
from .utils import register_classification_fitness_functions

__all__ = [
    'BinaryClassifier',
    'train_binary_classifier',
    'register_classification_fitness_functions',
    'calculate_binary_metrics',
    'save_metrics_to_csv'
]
