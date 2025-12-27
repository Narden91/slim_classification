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
Typed result structures for classification metrics.

This module provides strongly-typed dataclasses for classification results,
replacing untyped dictionaries while maintaining backward compatibility.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Union
import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    """
    Typed container for binary classification metrics.
    
    This immutable dataclass provides type-safe access to classification
    metrics while maintaining backward compatibility with dict-based access
    through __getitem__.
    
    Attributes
    ----------
    accuracy : float
        Overall prediction accuracy (TP + TN) / (TP + TN + FP + FN).
    precision : float
        Precision score TP / (TP + FP).
    recall : float
        Recall/sensitivity score TP / (TP + FN).
    f1 : float
        F1 score (harmonic mean of precision and recall).
    specificity : float
        True negative rate TN / (TN + FP).
    true_positives : int
        Count of true positive predictions.
    true_negatives : int
        Count of true negative predictions.
    false_positives : int
        Count of false positive predictions.
    false_negatives : int
        Count of false negative predictions.
    confusion_matrix : np.ndarray
        Full confusion matrix of shape (2, 2).
        
    Examples
    --------
    >>> metrics = BinaryMetrics(
    ...     accuracy=0.85, precision=0.82, recall=0.88, f1=0.85,
    ...     specificity=0.82, true_positives=44, true_negatives=41,
    ...     false_positives=9, false_negatives=6,
    ...     confusion_matrix=np.array([[41, 9], [6, 44]])
    ... )
    >>> metrics.accuracy
    0.85
    >>> metrics['accuracy']  # Dict-like access for backward compatibility
    0.85
    >>> dict(metrics)  # Convert to dict
    {'accuracy': 0.85, ...}
    
    Notes
    -----
    The dataclass is frozen (immutable) to prevent accidental modification.
    Use __getitem__ for dict-like access to maintain backward compatibility
    with existing code expecting dict results.
    """
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: np.ndarray
    
    def __getitem__(self, key: str) -> Union[float, int, np.ndarray]:
        """
        Enable dict-like access for backward compatibility.
        
        Parameters
        ----------
        key : str
            Attribute name to access.
            
        Returns
        -------
        Union[float, int, np.ndarray]
            The requested metric value.
            
        Raises
        ------
        KeyError
            If key is not a valid metric name.
            
        Examples
        --------
        >>> metrics['accuracy']
        0.85
        >>> metrics['true_positives']
        44
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' is not a valid metric name")
    
    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists (for 'in' operator support).
        
        Parameters
        ----------
        key : str
            Attribute name to check.
            
        Returns
        -------
        bool
            True if key is a valid attribute name.
            
        Examples
        --------
        >>> 'accuracy' in metrics
        True
        >>> 'invalid_key' in metrics
        False
        """
        return hasattr(self, key)
    
    def keys(self):
        """Return field names for dict compatibility."""
        return asdict(self).keys()
    
    def values(self):
        """Return field values for dict compatibility."""
        return asdict(self).values()
    
    def items(self):
        """Return (key, value) pairs for dict compatibility."""
        return asdict(self).items()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get metric value with default fallback.
        
        Parameters
        ----------
        key : str
            Metric name to retrieve.
        default : Any, optional
            Default value if key not found.
            
        Returns
        -------
        Any
            Metric value or default.
            
        Examples
        --------
        >>> metrics.get('accuracy')
        0.85
        >>> metrics.get('nonexistent', 0.0)
        0.0
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def to_dict(self) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Convert to plain dictionary.
        
        Returns
        -------
        Dict[str, Union[float, int, np.ndarray]]
            Dictionary representation of metrics.
            
        Examples
        --------
        >>> metrics_dict = metrics.to_dict()
        >>> isinstance(metrics_dict, dict)
        True
        """
        return asdict(self)
    
    def __iter__(self):
        """Enable iteration over keys for dict compatibility."""
        return iter(asdict(self))
    
    def __repr__(self) -> str:
        """Provide detailed string representation."""
        return (
            f"BinaryMetrics("
            f"accuracy={self.accuracy:.4f}, "
            f"precision={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"f1={self.f1:.4f}, "
            f"specificity={self.specificity:.4f}, "
            f"TP={self.true_positives}, "
            f"TN={self.true_negatives}, "
            f"FP={self.false_positives}, "
            f"FN={self.false_negatives})"
        )
