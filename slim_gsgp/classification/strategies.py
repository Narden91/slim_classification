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
Strategy Pattern implementations for prediction strategies.

This module provides strategy classes for different prediction approaches
in binary classification, enabling runtime selection of prediction behavior.

Examples
--------
>>> from slim_gsgp.classification.strategies import (
...     SigmoidStrategy, SignBasedStrategy, PredictionContext
... )
>>> strategy = SigmoidStrategy(scale=1.0, threshold=0.5)
>>> context = PredictionContext(strategy)
>>> predictions = context.predict(raw_outputs)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch

from .validators import validate_threshold, validate_scaling_factor

logger = logging.getLogger(__name__)


class PredictionStrategy(ABC):
    """
    Abstract base class for prediction strategies.
    
    This class defines the interface for different prediction strategies
    in binary classification. Implementations determine how raw model
    outputs are transformed into class predictions and probabilities.
    
    Examples
    --------
    >>> class CustomStrategy(PredictionStrategy):
    ...     def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
    ...         return (raw_outputs > 0).float()
    ...     def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
    ...         probs = torch.sigmoid(raw_outputs)
    ...         return torch.stack([1 - probs, probs], dim=1)
    """
    
    @abstractmethod
    def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Convert raw model outputs to class predictions.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw outputs from the model.
            
        Returns
        -------
        torch.Tensor
            Binary class predictions (0 or 1).
        """
        ...
    
    @abstractmethod
    def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Convert raw model outputs to class probabilities.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw outputs from the model.
            
        Returns
        -------
        torch.Tensor
            Class probabilities of shape (n_samples, 2).
            Column 0 is P(class=0), column 1 is P(class=1).
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...


@dataclass(frozen=True)
class SigmoidStrategy(PredictionStrategy):
    """
    Sigmoid-based prediction strategy.
    
    This strategy applies a scaled sigmoid function to convert
    raw outputs to probabilities, then thresholds for predictions.
    
    Attributes
    ----------
    scale : float
        Scaling factor applied before sigmoid. Higher values
        create sharper decision boundaries.
    threshold : float
        Probability threshold for classification.
        
    Examples
    --------
    >>> strategy = SigmoidStrategy(scale=1.0, threshold=0.5)
    >>> raw = torch.tensor([0.0, 1.0, -1.0, 2.0])
    >>> strategy.predict(raw)
    tensor([0., 1., 0., 1.])
    >>> strategy.predict_proba(raw)
    tensor([[0.5000, 0.5000],
            [0.2689, 0.7311],
            [0.7311, 0.2689],
            [0.1192, 0.8808]])
    """
    scale: float = 1.0
    threshold: float = 0.5
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        validate_scaling_factor(self.scale)
        validate_threshold(self.threshold)
    
    def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels using sigmoid + threshold.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Binary predictions (0 or 1).
        """
        probs = torch.sigmoid(raw_outputs * self.scale)
        return (probs > self.threshold).float()
    
    def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using sigmoid.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Probabilities of shape (n_samples, 2).
        """
        probs = torch.sigmoid(raw_outputs * self.scale)
        return torch.stack([1 - probs, probs], dim=1)
    
    @property
    def name(self) -> str:
        return f"sigmoid(scale={self.scale}, threshold={self.threshold})"


@dataclass(frozen=True)
class SignBasedStrategy(PredictionStrategy):
    """
    Sign-based prediction strategy.
    
    This strategy uses the sign of raw outputs for classification:
    - Negative values -> class 0
    - Non-negative values -> class 1
    
    For probabilities, this returns hard 0/1 values since there's
    no natural probability interpretation for sign-based prediction.
    
    Examples
    --------
    >>> strategy = SignBasedStrategy()
    >>> raw = torch.tensor([-0.5, 0.0, 0.5, -1.0])
    >>> strategy.predict(raw)
    tensor([0., 1., 1., 0.])
    """
    
    def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels based on sign.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Binary predictions (0 for negative, 1 for non-negative).
        """
        return (raw_outputs >= 0).float()
    
    def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (hard 0/1 for sign-based).
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Hard probabilities of shape (n_samples, 2).
        """
        probs = (raw_outputs >= 0).float()
        return torch.stack([1 - probs, probs], dim=1)
    
    @property
    def name(self) -> str:
        return "sign-based"


@dataclass(frozen=True)
class SoftmaxStrategy(PredictionStrategy):
    """
    Softmax-based prediction strategy.
    
    This strategy treats raw outputs as logits and applies
    a softmax-like transformation for binary classification.
    Useful when outputs are expected to be in a specific range.
    
    Attributes
    ----------
    temperature : float
        Temperature parameter for softmax. Lower values create
        sharper distributions.
        
    Examples
    --------
    >>> strategy = SoftmaxStrategy(temperature=1.0)
    >>> raw = torch.tensor([0.0, 1.0, -1.0])
    >>> strategy.predict(raw)
    tensor([0., 1., 0.])
    """
    temperature: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate temperature."""
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
    
    def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Binary predictions.
        """
        # Create logits for binary classification
        scaled = raw_outputs / self.temperature
        probs = torch.sigmoid(scaled)
        return (probs > 0.5).float()
    
    def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using temperature-scaled sigmoid.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Probabilities of shape (n_samples, 2).
        """
        scaled = raw_outputs / self.temperature
        probs = torch.sigmoid(scaled)
        return torch.stack([1 - probs, probs], dim=1)
    
    @property
    def name(self) -> str:
        return f"softmax(temperature={self.temperature})"


class PredictionContext:
    """
    Context class for using prediction strategies.
    
    This class implements the Strategy Pattern context,
    allowing runtime selection and switching of prediction strategies.
    
    Attributes
    ----------
    strategy : PredictionStrategy
        The current prediction strategy.
        
    Examples
    --------
    >>> context = PredictionContext(SigmoidStrategy(scale=1.0))
    >>> predictions = context.predict(raw_outputs)
    
    >>> # Switch strategy at runtime
    >>> context.strategy = SignBasedStrategy()
    >>> predictions = context.predict(raw_outputs)
    """
    
    __slots__ = ('_strategy',)
    
    def __init__(self, strategy: PredictionStrategy) -> None:
        """
        Initialize with a prediction strategy.
        
        Parameters
        ----------
        strategy : PredictionStrategy
            The prediction strategy to use.
        """
        self._strategy = strategy
        logger.debug(f"PredictionContext initialized with strategy: {strategy.name}")
    
    @property
    def strategy(self) -> PredictionStrategy:
        """Get the current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: PredictionStrategy) -> None:
        """Set a new strategy."""
        self._strategy = strategy
        logger.debug(f"PredictionContext strategy changed to: {strategy.name}")
    
    def predict(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Predict using the current strategy.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Binary predictions.
        """
        return self._strategy.predict(raw_outputs)
    
    def predict_proba(self, raw_outputs: torch.Tensor) -> torch.Tensor:
        """
        Get probabilities using the current strategy.
        
        Parameters
        ----------
        raw_outputs : torch.Tensor
            Raw model outputs.
            
        Returns
        -------
        torch.Tensor
            Class probabilities of shape (n_samples, 2).
        """
        return self._strategy.predict_proba(raw_outputs)


# Strategy registry for easy lookup by name
_STRATEGY_REGISTRY: dict[str, Type[PredictionStrategy]] = {
    'sigmoid': SigmoidStrategy,
    'sign': SignBasedStrategy,
    'softmax': SoftmaxStrategy,
}


def get_strategy(
    name: str,
    **kwargs
) -> PredictionStrategy:
    """
    Get a prediction strategy by name.
    
    Parameters
    ----------
    name : str
        Strategy name: 'sigmoid', 'sign', or 'softmax'.
    **kwargs
        Arguments passed to the strategy constructor.
        
    Returns
    -------
    PredictionStrategy
        The requested strategy instance.
        
    Raises
    ------
    ValueError
        If the strategy name is not recognized.
        
    Examples
    --------
    >>> strategy = get_strategy('sigmoid', scale=2.0, threshold=0.6)
    >>> strategy = get_strategy('sign')
    """
    name_lower = name.lower()
    
    if name_lower not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: '{name}'. Available: {available}"
        )
    
    strategy_cls = _STRATEGY_REGISTRY[name_lower]
    return strategy_cls(**kwargs)


def register_strategy(name: str, strategy_class: Type[PredictionStrategy]) -> None:
    """
    Register a custom prediction strategy.
    
    Parameters
    ----------
    name : str
        Name for the strategy.
    strategy_class : Type[PredictionStrategy]
        The strategy class.
        
    Examples
    --------
    >>> class MyStrategy(PredictionStrategy):
    ...     def predict(self, raw): return (raw > 0).float()
    ...     def predict_proba(self, raw): ...
    ...     @property
    ...     def name(self): return "my_strategy"
    >>> register_strategy('my', MyStrategy)
    """
    _STRATEGY_REGISTRY[name.lower()] = strategy_class
    logger.info(f"Registered prediction strategy: {name}")
