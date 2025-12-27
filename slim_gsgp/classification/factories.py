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
Factory Pattern implementations for algorithm creation.

This module provides factory classes for creating GP-based models,
enabling extensible algorithm registration and creation.

Examples
--------
>>> from slim_gsgp.classification.factories import AlgorithmFactory
>>> factory = AlgorithmFactory()
>>> model = factory.create('gp', X_train, y_train, pop_size=50)

>>> # Register custom algorithm
>>> factory.register('custom_gp', my_custom_algorithm_func)
>>> model = factory.create('custom_gp', X_train, y_train)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from .exceptions import AlgorithmNotFoundError

logger = logging.getLogger(__name__)


# Type alias for algorithm creator functions
AlgorithmCreator = Callable[..., Any]


class AlgorithmFactoryBase(ABC):
    """Abstract base class for algorithm factories.
    
    This class defines the interface for algorithm factories,
    enabling different factory implementations.
    """
    
    @abstractmethod
    def create(self, algorithm: str, *args: Any, **kwargs: Any) -> Any:
        """Create an algorithm instance."""
        ...
    
    @abstractmethod
    def register(self, name: str, creator: AlgorithmCreator) -> None:
        """Register a new algorithm creator."""
        ...
    
    @abstractmethod
    def available_algorithms(self) -> List[str]:
        """Return list of available algorithm names."""
        ...


class AlgorithmFactory(AlgorithmFactoryBase):
    """
    Factory for creating GP-based algorithm instances.
    
    This factory implements the Factory Pattern to create different
    GP-based models (GP, GSGP, SLIM) with a unified interface.
    It supports dynamic registration of new algorithms.
    
    Attributes
    ----------
    _creators : Dict[str, AlgorithmCreator]
        Registry mapping algorithm names to their creator functions.
    _default_kwargs : Dict[str, Dict[str, Any]]
        Default keyword arguments for each algorithm.
        
    Examples
    --------
    >>> factory = AlgorithmFactory()
    >>> print(factory.available_algorithms())
    ['gp', 'gsgp', 'slim']
    
    >>> # Create with default settings
    >>> model = factory.create('gp', X_train, y_train)
    
    >>> # Create with custom settings
    >>> model = factory.create('slim', X_train, y_train, 
    ...                        slim_version='SLIM+ABS', pop_size=100)
    
    >>> # Register custom algorithm
    >>> def my_gp_variant(X_train, y_train, **kwargs):
    ...     return custom_implementation(X_train, y_train, **kwargs)
    >>> factory.register('my_gp', my_gp_variant)
    """
    
    # Class-level default registry for singleton-like behavior
    _global_creators: Dict[str, AlgorithmCreator] = {}
    _global_defaults: Dict[str, Dict[str, Any]] = {}
    _initialized: bool = False
    
    def __init__(self) -> None:
        """Initialize the algorithm factory with built-in algorithms."""
        # Instance-level overrides (optional)
        self._creators: Dict[str, AlgorithmCreator] = {}
        self._default_kwargs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize global creators once
        if not AlgorithmFactory._initialized:
            self._register_builtin_algorithms()
            AlgorithmFactory._initialized = True
    
    def _register_builtin_algorithms(self) -> None:
        """Register the built-in GP-based algorithms."""
        # Lazy imports to avoid circular dependencies
        from ..main_gp import gp
        from ..main_gsgp import gsgp
        from ..main_slim import slim
        
        # Register creators
        AlgorithmFactory._global_creators['gp'] = gp
        AlgorithmFactory._global_creators['gsgp'] = gsgp
        AlgorithmFactory._global_creators['slim'] = slim
        
        # Set default kwargs for each algorithm
        AlgorithmFactory._global_defaults['gp'] = {}
        AlgorithmFactory._global_defaults['gsgp'] = {'reconstruct': True}
        AlgorithmFactory._global_defaults['slim'] = {}
        
        logger.debug("Registered built-in algorithms: gp, gsgp, slim")
    
    def register(
        self, 
        name: str, 
        creator: AlgorithmCreator,
        default_kwargs: Optional[Dict[str, Any]] = None,
        *,
        instance_only: bool = False
    ) -> None:
        """
        Register a new algorithm creator.
        
        Parameters
        ----------
        name : str
            Name to register the algorithm under (case-insensitive).
        creator : AlgorithmCreator
            Callable that creates the algorithm. Should accept
            X_train, y_train, and optional X_test, y_test parameters.
        default_kwargs : Dict[str, Any], optional
            Default keyword arguments for this algorithm.
        instance_only : bool, default=False
            If True, register only for this instance. If False,
            register globally for all factory instances.
            
        Examples
        --------
        >>> factory = AlgorithmFactory()
        >>> def my_algorithm(X_train, y_train, **kwargs):
        ...     return MyModel(X_train, y_train, **kwargs)
        >>> factory.register('my_algo', my_algorithm, {'param': 10})
        """
        name_lower = name.lower()
        
        if instance_only:
            self._creators[name_lower] = creator
            self._default_kwargs[name_lower] = default_kwargs or {}
        else:
            AlgorithmFactory._global_creators[name_lower] = creator
            AlgorithmFactory._global_defaults[name_lower] = default_kwargs or {}
        
        logger.info(f"Registered algorithm: {name_lower}")
    
    def unregister(self, name: str, *, instance_only: bool = False) -> bool:
        """
        Unregister an algorithm.
        
        Parameters
        ----------
        name : str
            Name of the algorithm to unregister.
        instance_only : bool, default=False
            If True, only remove from instance registry.
            
        Returns
        -------
        bool
            True if algorithm was found and removed, False otherwise.
        """
        name_lower = name.lower()
        
        if instance_only:
            if name_lower in self._creators:
                del self._creators[name_lower]
                self._default_kwargs.pop(name_lower, None)
                return True
        else:
            if name_lower in AlgorithmFactory._global_creators:
                del AlgorithmFactory._global_creators[name_lower]
                AlgorithmFactory._global_defaults.pop(name_lower, None)
                return True
        
        return False
    
    def available_algorithms(self) -> List[str]:
        """
        Get list of available algorithm names.
        
        Returns
        -------
        List[str]
            Sorted list of registered algorithm names.
            
        Examples
        --------
        >>> factory = AlgorithmFactory()
        >>> factory.available_algorithms()
        ['gp', 'gsgp', 'slim']
        """
        all_names = set(AlgorithmFactory._global_creators.keys())
        all_names.update(self._creators.keys())
        return sorted(all_names)
    
    def get_creator(self, name: str) -> AlgorithmCreator:
        """
        Get the creator function for an algorithm.
        
        Parameters
        ----------
        name : str
            Algorithm name.
            
        Returns
        -------
        AlgorithmCreator
            The creator function.
            
        Raises
        ------
        AlgorithmNotFoundError
            If algorithm is not registered.
        """
        name_lower = name.lower()
        
        # Check instance-level first, then global
        if name_lower in self._creators:
            return self._creators[name_lower]
        if name_lower in AlgorithmFactory._global_creators:
            return AlgorithmFactory._global_creators[name_lower]
        
        available = self.available_algorithms()
        raise AlgorithmNotFoundError(
            f"Unknown algorithm: '{name}'. "
            f"Supported algorithms are: {', '.join(repr(a) for a in available)}"
        )
    
    def get_default_kwargs(self, name: str) -> Dict[str, Any]:
        """
        Get default keyword arguments for an algorithm.
        
        Parameters
        ----------
        name : str
            Algorithm name.
            
        Returns
        -------
        Dict[str, Any]
            Default kwargs (empty dict if none registered).
        """
        name_lower = name.lower()
        
        if name_lower in self._default_kwargs:
            return self._default_kwargs[name_lower].copy()
        return AlgorithmFactory._global_defaults.get(name_lower, {}).copy()
    
    def create(
        self,
        algorithm: str,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Any:
        """
        Create a model using the specified algorithm.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name ('gp', 'gsgp', 'slim', or custom registered).
        X_train : torch.Tensor
            Training features.
        y_train : torch.Tensor
            Training labels.
        X_test : torch.Tensor, optional
            Test/validation features.
        y_test : torch.Tensor, optional
            Test/validation labels.
        **kwargs
            Additional arguments passed to the algorithm creator.
            
        Returns
        -------
        Any
            The created model instance.
            
        Raises
        ------
        AlgorithmNotFoundError
            If the specified algorithm is not registered.
            
        Examples
        --------
        >>> factory = AlgorithmFactory()
        >>> model = factory.create(
        ...     'gp', X_train, y_train, X_test, y_test,
        ...     pop_size=100, n_iter=50
        ... )
        """
        creator = self.get_creator(algorithm)
        default_kwargs = self.get_default_kwargs(algorithm)
        
        # Merge defaults with provided kwargs (provided takes precedence)
        merged_kwargs = {**default_kwargs, **kwargs}
        
        logger.debug(
            f"Creating {algorithm} model with kwargs: {list(merged_kwargs.keys())}"
        )
        
        return creator(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            **merged_kwargs
        )


# Global factory instance for convenience
_default_factory: Optional[AlgorithmFactory] = None


def get_default_factory() -> AlgorithmFactory:
    """
    Get the default global algorithm factory.
    
    Returns
    -------
    AlgorithmFactory
        The default factory instance.
        
    Examples
    --------
    >>> factory = get_default_factory()
    >>> model = factory.create('gp', X_train, y_train)
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = AlgorithmFactory()
    return _default_factory


def create_algorithm(
    algorithm: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: Optional[torch.Tensor] = None,
    y_test: Optional[torch.Tensor] = None,
    **kwargs: Any
) -> Any:
    """
    Convenience function to create an algorithm using the default factory.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name ('gp', 'gsgp', 'slim', or custom registered).
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels.
    X_test : torch.Tensor, optional
        Test/validation features.
    y_test : torch.Tensor, optional
        Test/validation labels.
    **kwargs
        Additional arguments passed to the algorithm.
        
    Returns
    -------
    Any
        The created model instance.
        
    Examples
    --------
    >>> from slim_gsgp.classification.factories import create_algorithm
    >>> model = create_algorithm('gp', X_train, y_train, pop_size=50)
    """
    return get_default_factory().create(
        algorithm, X_train, y_train, X_test, y_test, **kwargs
    )


def register_algorithm(
    name: str,
    creator: AlgorithmCreator,
    default_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a custom algorithm with the default factory.
    
    Parameters
    ----------
    name : str
        Name for the algorithm.
    creator : AlgorithmCreator
        Function that creates the algorithm.
    default_kwargs : Dict[str, Any], optional
        Default keyword arguments.
        
    Examples
    --------
    >>> def my_custom_gp(X_train, y_train, **kwargs):
    ...     return MyCustomGP(X_train, y_train, **kwargs)
    >>> register_algorithm('custom_gp', my_custom_gp, {'depth': 5})
    >>> model = create_algorithm('custom_gp', X_train, y_train)
    """
    get_default_factory().register(name, creator, default_kwargs)
