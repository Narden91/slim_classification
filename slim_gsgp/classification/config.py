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
Configuration dataclasses for classification module.

This module provides typed configuration objects for classifier parameters,
ensuring type safety and validation at construction time.
"""

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

from .validators import validate_threshold, validate_scaling_factor


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Configuration for binary classifier parameters.
    
    This immutable dataclass encapsulates all configuration options
    for the BinaryClassifier, providing type safety and validation.
    
    Attributes
    ----------
    threshold : float
        Classification threshold for binary prediction (default: 0.5).
        Must be in range (0, 1).
    use_sigmoid : bool
        Whether to apply sigmoid transformation to model outputs (default: True).
        If False, classification uses sign-based prediction.
    sigmoid_scale : float
        Scaling factor for sigmoid function (default: 1.0).
        Higher values make the sigmoid steeper. Must be positive.
        
    Examples
    --------
    >>> config = ClassifierConfig()
    >>> config.threshold
    0.5
    
    >>> config = ClassifierConfig(threshold=0.7, sigmoid_scale=2.0)
    >>> config.use_sigmoid
    True
    
    >>> # Invalid configurations raise errors at construction
    >>> config = ClassifierConfig(threshold=1.5)  # Raises InvalidThresholdError
    
    Notes
    -----
    The dataclass is frozen (immutable) to prevent accidental modification
    after construction.
    """
    threshold: float = 0.5
    use_sigmoid: bool = True
    sigmoid_scale: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        validate_threshold(self.threshold)
        validate_scaling_factor(self.sigmoid_scale)
    
    def replace(self, **changes) -> 'ClassifierConfig':
        """Create a new config with specified changes.
        
        This is a convenience method that wraps dataclasses.replace().
        
        Parameters
        ----------
        **changes
            Fields to change and their new values.
            
        Returns
        -------
        ClassifierConfig
            New config instance with modifications.
            
        Examples
        --------
        >>> config = ClassifierConfig(threshold=0.5)
        >>> new_config = config.replace(threshold=0.6, sigmoid_scale=2.0)
        >>> new_config.threshold
        0.6
        >>> config.threshold  # Original unchanged
        0.5
        """
        return replace(self, **changes)


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration for binary classifier training.
    
    This dataclass encapsulates training hyperparameters for GP-based
    classification algorithms.
    
    Attributes
    ----------
    algorithm : str
        Algorithm to use: 'gp', 'gsgp', or 'slim' (default: 'gp').
    fitness_function : str
        Fitness function to use (default: 'binary_cross_entropy').
    pop_size : int
        Population size for evolutionary algorithms (default: 100).
    n_iter : int
        Number of iterations/generations (default: 100).
    seed : Optional[int]
        Random seed for reproducibility (default: None).
    verbose : int
        Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1).
        
    Examples
    --------
    >>> config = TrainingConfig(algorithm='slim', pop_size=50, n_iter=20)
    >>> config.fitness_function
    'binary_cross_entropy'
    
    >>> config = TrainingConfig(seed=42, verbose=0)
    >>> config.algorithm
    'gp'
    """
    algorithm: Literal['gp', 'gsgp', 'slim'] = 'gp'
    fitness_function: str = 'binary_cross_entropy'
    pop_size: int = 100
    n_iter: int = 100
    seed: Optional[int] = None
    verbose: int = 1
    
    def __post_init__(self) -> None:
        """Validate training configuration."""
        valid_algorithms = ('gp', 'gsgp', 'slim')
        if self.algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got '{self.algorithm}'"
            )
        if self.pop_size <= 0:
            raise ValueError(f"pop_size must be positive, got {self.pop_size}")
        if self.n_iter <= 0:
            raise ValueError(f"n_iter must be positive, got {self.n_iter}")
        if self.verbose < 0:
            raise ValueError(f"verbose must be non-negative, got {self.verbose}")
    
    def replace(self, **changes) -> 'TrainingConfig':
        """Create a new config with specified changes.
        
        This is a convenience method that wraps dataclasses.replace().
        
        Parameters
        ----------
        **changes
            Fields to change and their new values.
            
        Returns
        -------
        TrainingConfig
            New config instance with modifications.
            
        Examples
        --------
        >>> config = TrainingConfig(pop_size=100)
        >>> new_config = config.replace(pop_size=50, n_iter=20)
        >>> new_config.pop_size
        50
        >>> config.pop_size  # Original unchanged
        100
        """
        return replace(self, **changes)
