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
Custom exceptions for classification module.

This module defines specific exception types for various error conditions
that can occur during classification tasks.
"""


class ClassificationError(Exception):
    """Base exception for all classification-related errors."""
    pass


class InvalidLabelError(ClassificationError):
    """Raised when labels are invalid for the classification task.
    
    Examples:
        - Non-binary labels for binary classification
        - Labels outside expected range
        - Inconsistent label dimensions
    """
    pass


class AlgorithmNotFoundError(ClassificationError):
    """Raised when a requested algorithm is not available or not recognized.
    
    Examples:
        - Unknown algorithm name
        - Algorithm not properly registered
    """
    pass


class InvalidThresholdError(ClassificationError):
    """Raised when classification threshold is invalid.
    
    Examples:
        - Threshold outside (0, 1) range
        - Non-numeric threshold value
    """
    pass


class InvalidShapeError(ClassificationError):
    """Raised when tensor shapes are incompatible.
    
    Examples:
        - Mismatched feature dimensions between train and test
        - Empty tensors
        - Incompatible label dimensions
    """
    pass


class FitnessRegistrationError(ClassificationError):
    """Raised when fitness function registration fails.
    
    Examples:
        - Missing dependencies
        - Invalid fitness function configuration
    """
    pass


class InvalidScalingFactorError(ClassificationError):
    """Raised when sigmoid scaling factor is invalid.
    
    Examples:
        - Non-positive scaling factor
        - Non-numeric scaling factor value
    """
    pass
