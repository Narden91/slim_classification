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
Feature Importance Extraction for GP Trees.

Provides methods to extract and analyze feature importance from evolved GP trees.
Supports multiple importance metrics:
- Frequency-based: Count of feature occurrences
- Depth-weighted: Features closer to root have higher importance
- Permutation-based: Measure performance degradation when feature is shuffled
"""

from typing import Any, Dict, List, Tuple, Optional, Union
import torch
from collections import Counter


def _unwrap_tree_repr(obj: Any) -> Any:
    """Best-effort conversion of common SLIM-GSGP objects to a traversable tree representation."""
    if isinstance(obj, (tuple, list, str)):
        return obj
    if hasattr(obj, "repr_"):
        return getattr(obj, "repr_")
    if hasattr(obj, "structure"):
        return getattr(obj, "structure")
    return obj


def _has_collection(obj: Any) -> bool:
    return hasattr(obj, "collection") and getattr(obj, "collection") is not None


def flatten_tree(tree_repr: Union[tuple, str, list, Any]) -> List[Any]:
    """
    Flatten a tree representation into a list of nodes.
    
    Parameters
    ----------
    tree_repr : tuple or str
        Tree representation (nested tuples or terminal string).
    
    Returns
    -------
    list
        Flattened list of all nodes in the tree.
    
    Examples
    --------
    >>> flatten_tree(('add', 'x0', ('mul', 'x1', 'x2')))
    ['add', 'x0', 'mul', 'x1', 'x2']
    """
    # Support SLIM Individuals (collection of trees)
    if _has_collection(tree_repr):
        result: List[Any] = []
        for sub_tree in getattr(tree_repr, "collection"):
            result.extend(flatten_tree(sub_tree))
        return result

    tree_repr = _unwrap_tree_repr(tree_repr)

    if isinstance(tree_repr, tuple):
        result = []
        for item in tree_repr:
            result.extend(flatten_tree(item))
        return result
    elif isinstance(tree_repr, list):
        result = []
        for item in tree_repr:
            result.extend(flatten_tree(item))
        return result
    else:
        return [tree_repr]


def extract_features(tree_repr: Union[tuple, str, list, Any]) -> List[str]:
    """
    Extract all feature names (terminals starting with 'x') from a tree.
    
    Parameters
    ----------
    tree_repr : tuple or str
        Tree representation.
    
    Returns
    -------
    list
        List of feature names found in the tree.
    
    Examples
    --------
    >>> extract_features(('add', 'x0', ('mul', 'x1', 'x2')))
    ['x0', 'x1', 'x2']
    """
    nodes = flatten_tree(tree_repr)
    return [node for node in nodes if isinstance(node, str) and node.startswith('x')]


def count_feature_occurrences(tree_repr: Union[tuple, str, list, Any]) -> Dict[str, int]:
    """
    Count how many times each feature appears in the tree.
    
    Parameters
    ----------
    tree_repr : tuple or str
        Tree representation.
    
    Returns
    -------
    dict
        Dictionary mapping feature names to occurrence counts.
        
    Examples
    --------
    >>> count_feature_occurrences(('add', 'x0', ('mul', 'x0', 'x1')))
    {'x0': 2, 'x1': 1}
    """
    features = extract_features(tree_repr)
    return dict(Counter(features))


def normalize_importance_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize importance scores to sum to 1.0.
    
    Parameters
    ----------
    scores : dict
        Dictionary of feature names to importance scores.
    
    Returns
    -------
    dict
        Normalized scores (sum to 1.0).
        
    Examples
    --------
    >>> normalize_importance_scores({'x0': 2, 'x1': 3, 'x2': 5})
    {'x0': 0.2, 'x1': 0.3, 'x2': 0.5}
    """
    total = sum(scores.values())
    if total == 0:
        return {k: 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


class FeatureImportanceExtractor:
    """
    Extract feature importance from GP trees.
    
    Attributes
    ----------
    n_features : int
        Number of features in the dataset.
    feature_names : list
        List of feature names (e.g., ['x0', 'x1', 'x2']).
    """
    
    def __init__(self, n_features: int, feature_names: Optional[List[str]] = None):
        """
        Initialize the feature importance extractor.
        
        Parameters
        ----------
        n_features : int
            Number of features in the dataset.
        feature_names : list, optional
            Custom feature names. If None, uses ['x0', 'x1', ..., 'xN'].
        """
        self.n_features = n_features
        
        if feature_names is None:
            self.feature_names = [f"x{i}" for i in range(n_features)]
        else:
            if len(feature_names) != n_features:
                raise ValueError(f"Number of feature names ({len(feature_names)}) "
                               f"must match n_features ({n_features})")
            self.feature_names = feature_names
    
    def frequency_importance(self, tree: Any, normalize: bool = True) -> Dict[str, float]:
        """
        Calculate feature importance based on frequency of occurrence.
        
        Parameters
        ----------
        tree : Tree
            A Tree object with repr_ attribute.
        normalize : bool, optional
            If True, normalize scores to sum to 1.0. Default is True.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to importance scores.
        
        Examples
        --------
        >>> extractor = FeatureImportanceExtractor(n_features=3)
        >>> tree = Tree(('add', 'x0', ('mul', 'x0', 'x1')))
        >>> extractor.frequency_importance(tree)
        {'x0': 0.667, 'x1': 0.333, 'x2': 0.0}
        """
        # Get feature counts from any supported model representation
        counts = count_feature_occurrences(tree)
        
        # Initialize all features with 0
        importance = {name: 0.0 for name in self.feature_names}
        
        # Update with actual counts
        for feature, count in counts.items():
            if feature in importance:
                importance[feature] = float(count)
        
        if normalize:
            importance = normalize_importance_scores(importance)
        
        return importance
    
    def depth_weighted_importance(self, tree: Any, normalize: bool = True) -> Dict[str, float]:
        """
        Calculate feature importance weighted by tree depth.
        Features closer to the root have higher importance.
        
        Parameters
        ----------
        tree : Tree
            A Tree object with repr_ attribute.
        normalize : bool, optional
            If True, normalize scores to sum to 1.0. Default is True.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to depth-weighted importance scores.
        
        Notes
        -----
        Weight calculation: weight = 1 / (depth + 1)
        Root has depth 0, so weight = 1.0
        Children have depth 1, so weight = 0.5
        Grandchildren have depth 2, so weight = 0.333
        
        Examples
        --------
        >>> extractor = FeatureImportanceExtractor(n_features=3)
        >>> tree = Tree(('add', 'x0', ('mul', 'x1', 'x2')))
        >>> # x0 at depth 1 (weight=0.5), x1 and x2 at depth 2 (weight=0.333)
        >>> extractor.depth_weighted_importance(tree)
        {'x0': 0.375, 'x1': 0.3125, 'x2': 0.3125}
        """
        def get_feature_depths(node: Any, depth: int = 0):
            """Recursively get features with their depths."""
            if _has_collection(node):
                result: List[Tuple[str, int]] = []
                for sub_tree in getattr(node, "collection"):
                    result.extend(get_feature_depths(sub_tree, depth))
                return result

            node = _unwrap_tree_repr(node)

            if isinstance(node, str):
                if node.startswith('x'):
                    return [(node, depth)]
                return []

            # It's a tuple/list (function/operator and arguments)
            if isinstance(node, tuple):
                children = node[1:]  # Skip function name
            elif isinstance(node, list):
                children = node[1:] if len(node) > 0 else []  # Skip operator/callable
            else:
                return []

            result: List[Tuple[str, int]] = []
            for item in children:
                result.extend(get_feature_depths(item, depth + 1))
            return result
        
        # Get all features with their depths
        feature_depths = get_feature_depths(tree)
        
        # Calculate weighted scores: weight = 1 / (depth + 1)
        importance = {name: 0.0 for name in self.feature_names}
        
        for feature, depth in feature_depths:
            if feature in importance:
                weight = 1.0 / (depth + 1)
                importance[feature] += weight
        
        if normalize:
            importance = normalize_importance_scores(importance)
        
        return importance
    
    def permutation_importance(
        self,
        tree: Any,
        X: torch.Tensor,
        y: torch.Tensor,
        fitness_function,
        n_repeats: int = 10,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate feature importance using permutation method.
        
        Measures how much the fitness degrades when each feature is randomly shuffled.
        More important features cause larger fitness degradation when removed.
        
        Parameters
        ----------
        tree : Tree
            A Tree object with repr_ attribute and apply_tree method.
        X : torch.Tensor
            Input data (shape: [n_samples, n_features]).
        y : torch.Tensor
            Target values (shape: [n_samples]).
        fitness_function : callable
            Function to compute fitness/error. Should accept (y_true, y_pred).
        n_repeats : int, optional
            Number of times to shuffle each feature. Default is 10.
        normalize : bool, optional
            If True, normalize scores to sum to 1.0. Default is True.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to permutation importance scores.
        
        Notes
        -----
        - Higher scores indicate more important features
        - Computational cost: O(n_features * n_repeats * tree_evaluation_cost)
        - Accounts for feature interactions
        - Most accurate but slowest method
        
        Examples
        --------
        >>> extractor = FeatureImportanceExtractor(n_features=3)
        >>> X = torch.randn(100, 3)
        >>> y = torch.randn(100)
        >>> tree = Tree(('add', 'x0', 'x1'))
        >>> importance = extractor.permutation_importance(
        ...     tree, X, y, rmse, n_repeats=10
        ... )
        """
        def predict(model: Any, X_input: torch.Tensor) -> torch.Tensor:
            if hasattr(model, "apply_tree"):
                return model.apply_tree(X_input)
            if hasattr(model, "predict"):
                return model.predict(X_input)
            raise TypeError(
                "Unsupported model for permutation importance: expected .apply_tree() or .predict()."
            )

        # Get baseline fitness (original data)
        baseline_pred = predict(tree, X)
        baseline_fitness = float(fitness_function(y, baseline_pred))
        
        importance = {name: 0.0 for name in self.feature_names}
        
        # For each feature
        for i, feature_name in enumerate(self.feature_names):
            fitness_deltas = []
            
            # Repeat shuffling n_repeats times
            for _ in range(n_repeats):
                # Create a copy of X and shuffle this feature
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.shape[0])
                X_permuted[:, i] = X[perm_indices, i]
                
                # Evaluate with permuted feature
                permuted_pred = predict(tree, X_permuted)
                permuted_fitness = float(fitness_function(y, permuted_pred))
                
                # Calculate fitness increase (degradation)
                # For minimization objectives, higher is worse
                fitness_delta = permuted_fitness - baseline_fitness
                fitness_deltas.append(max(0.0, fitness_delta))  # Only count degradation
            
            # Average importance across repeats
            importance[feature_name] = sum(fitness_deltas) / len(fitness_deltas)
        
        if normalize:
            importance = normalize_importance_scores(importance)
        
        return importance
    
    def get_top_features(
        self,
        importance: Dict[str, float],
        n: int = 10,
        feature_name_mapping: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Get the top N most important features sorted by importance score.
        
        Parameters
        ----------
        importance : dict
            Dictionary of feature names to importance scores.
        n : int, optional
            Number of top features to return. Default is 10.
        feature_name_mapping : dict, optional
            Mapping from generic feature names (x0, x1) to actual names.
        
        Returns
        -------
        list
            List of tuples (feature_id, score, actual_name) sorted by importance.
        
        Examples
        --------
        >>> extractor = FeatureImportanceExtractor(n_features=5)
        >>> importance = {'x0': 0.5, 'x1': 0.3, 'x2': 0.15, 'x3': 0.05, 'x4': 0.0}
        >>> extractor.get_top_features(importance, n=3)
        [('x0', 0.5, None), ('x1', 0.3, None), ('x2', 0.15, None)]
        """
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for feat, score in sorted_items[:n]:
            if score > 0:  # Only include features with non-zero importance
                actual_name = None
                if feature_name_mapping and feat in feature_name_mapping:
                    actual_name = feature_name_mapping[feat]
                result.append((feat, score, actual_name))
        
        return result
    
    def summary(
        self,
        tree,
        X: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        fitness_function=None,
        feature_name_mapping: Optional[Dict[str, str]] = None,
        include_permutation: bool = True,
        n_top: int = 10
    ) -> Dict[str, any]:
        """
        Generate a comprehensive feature importance summary.
        
        Parameters
        ----------
        tree : Tree
            The GP tree to analyze.
        X : torch.Tensor, optional
            Input data (required for permutation importance).
        y : torch.Tensor, optional
            Target values (required for permutation importance).
        fitness_function : callable, optional
            Fitness function (required for permutation importance).
        feature_name_mapping : dict, optional
            Mapping from generic to actual feature names.
        include_permutation : bool, optional
            Whether to calculate permutation importance. Default is True.
        n_top : int, optional
            Number of top features to include. Default is 10.
        
        Returns
        -------
        dict
            Summary containing all importance metrics and analysis.
        """
        summary = {
            'n_features': self.n_features,
            'feature_names': self.feature_names,
        }
        
        # Frequency-based
        freq_imp = self.frequency_importance(tree, normalize=True)
        summary['frequency'] = {
            'scores': freq_imp,
            'top_features': self.get_top_features(freq_imp, n_top, feature_name_mapping)
        }
        
        # Depth-weighted
        depth_imp = self.depth_weighted_importance(tree, normalize=True)
        summary['depth_weighted'] = {
            'scores': depth_imp,
            'top_features': self.get_top_features(depth_imp, n_top, feature_name_mapping)
        }
        
        # Permutation-based (if data provided)
        if include_permutation and X is not None and y is not None and fitness_function is not None:
            perm_imp = self.permutation_importance(tree, X, y, fitness_function, n_repeats=10, normalize=True)
            summary['permutation'] = {
                'scores': perm_imp,
                'top_features': self.get_top_features(perm_imp, n_top, feature_name_mapping)
            }
        
        # Statistics
        features_used = sum(1 for s in freq_imp.values() if s > 0)
        summary['statistics'] = {
            'features_used': features_used,
            'feature_utilization': features_used / self.n_features * 100
        }
        
        return summary
    
    def __repr__(self) -> str:
        return f"FeatureImportanceExtractor(n_features={self.n_features})"
