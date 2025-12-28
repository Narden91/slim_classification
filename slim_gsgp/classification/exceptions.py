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
    
    This exception is raised when:
    - Non-binary labels are provided for binary classification
    - Labels are outside the expected range [0, 1]
    - Label dimensions are inconsistent with features
    - Labels contain NaN or infinite values
    
    Examples
    --------
    >>> raise InvalidLabelError("Labels contain values other than 0 and 1: [0, 1, 2]")
    """
    def __init__(self, message: str = "Invalid labels for classification task") -> None:
        """Initialize with custom error message.
        
        Parameters
        ----------
        message : str
            Error message describing the validation failure.
        """
        super().__init__(message)
        self.message = message


class AlgorithmNotFoundError(ClassificationError):
    """Raised when a requested algorithm is not available or not recognized.
    
    This exception is raised when:
    - An unknown algorithm name is specified
    - An algorithm is not properly registered with the factory
    - Required dependencies for an algorithm are missing
    
    Examples
    --------
    >>> raise AlgorithmNotFoundError("Algorithm 'custom_gp' not found. Available: ['gp', 'gsgp', 'slim']")
    """
    def __init__(self, message: str = "Algorithm not found", available: list = None) -> None:
        """Initialize with custom error message and available algorithms.
        
        Parameters
        ----------
        message : str
            Error message describing the issue.
        available : list, optional
            List of available algorithm names.
        """
        if available:
            message = f"{message}. Available algorithms: {available}"
        super().__init__(message)
        self.message = message
        self.available = available or []


class InvalidThresholdError(ClassificationError):
    """Raised when classification threshold is invalid.
    
    This exception is raised when:
    - Threshold is outside the valid range (0, 1)
    - Threshold is not a numeric value
    - Threshold is NaN or infinite
    
    Examples
    --------
    >>> raise InvalidThresholdError("Threshold must be in range (0, 1), got 1.5")
    """
    def __init__(self, message: str = "Invalid threshold value", threshold: float = None) -> None:
        """Initialize with custom error message and threshold value.
        
        Parameters
        ----------
        message : str
            Error message describing the validation failure.
        threshold : float, optional
            The invalid threshold value that caused the error.
        """
        super().__init__(message)
        self.message = message
        self.threshold = threshold


class InvalidShapeError(ClassificationError):
    """Raised when tensor shapes are incompatible.
    
    This exception is raised when:
    - Feature dimensions don't match between train and test sets
    - Tensors are empty when data is required
    - Label dimensions don't match feature sample counts
    - Tensor dimensions exceed expected dimensionality
    
    Examples
    --------
    >>> raise InvalidShapeError("X has 100 samples but y has 50 samples")
    """
    def __init__(self, message: str = "Invalid tensor shape", expected: tuple = None, actual: tuple = None) -> None:
        """Initialize with custom error message and shape information.
        
        Parameters
        ----------
        message : str
            Error message describing the shape mismatch.
        expected : tuple, optional
            Expected tensor shape.
        actual : tuple, optional
            Actual tensor shape that caused the error.
        """
        if expected and actual:
            message = f"{message}. Expected {expected}, got {actual}"
        super().__init__(message)
        self.message = message
        self.expected = expected
        self.actual = actual


class FitnessRegistrationError(ClassificationError):
    """Raised when fitness function registration fails.
    
    This exception is raised when:
    - Required modules cannot be imported for registration
    - Fitness function configuration is invalid
    - Circular dependencies prevent registration
    - Registry is in an inconsistent state
    
    Examples
    --------
    >>> raise FitnessRegistrationError("Cannot import slim_gsgp.config.gp_config")
    """
    def __init__(self, message: str = "Fitness function registration failed", cause: Exception = None) -> None:
        """Initialize with custom error message and underlying cause.
        
        Parameters
        ----------
        message : str
            Error message describing the registration failure.
        cause : Exception, optional
            The underlying exception that caused the registration to fail.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause


class InvalidScalingFactorError(ClassificationError):
    """Raised when sigmoid scaling factor is invalid.
    
    This exception is raised when:
    - Scaling factor is not positive (≤ 0)
    - Scaling factor is not a numeric value
    - Scaling factor is NaN or infinite
    
    Examples
    --------
    >>> raise InvalidScalingFactorError("Scaling factor must be positive, got -1.0")
    """
    def __init__(self, message: str = "Invalid scaling factor", scale: float = None) -> None:
        """Initialize with custom error message and scaling factor.
        
        Parameters
        ----------
        message : str
            Error message describing the validation failure.
        scale : float, optional
            The invalid scaling factor that caused the error.
        """
        super().__init__(message)
        self.message = message
        self.scale = scale
