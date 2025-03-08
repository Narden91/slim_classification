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
Utility functions for classification tasks using Genetic Programming.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class ClassificationTreeWrapper:
    """
    Wrapper class to handle multiple trees for multiclass classification
    or a single tree for binary classification.

    This class provides a unified interface for working with classification models
    regardless of whether they are binary or multiclass classifiers.

    Attributes
    ----------
    trees : list
        List containing the trees (for multiclass) or a single tree (for binary).
    n_classes : int
        Number of classes in the classification problem.
    class_labels : list or None
        Optional list of class labels for better interpretability.
    """

    def __init__(self, trees, n_classes, class_labels=None):
        """
        Initialize the ClassificationTreeWrapper.

        Parameters
        ----------
        trees : list or object
            List of trees for multiclass or single tree for binary classification.
        n_classes : int
            Number of classes in the classification problem.
        class_labels : list, optional
            Optional list of class labels (e.g., for categorical classes).
        """
        self.trees = trees if isinstance(trees, list) else [trees]
        self.n_classes = n_classes
        self.class_labels = class_labels

    def predict_raw(self, X):
        """
        Generate raw predictions (before sigmoid/softmax).

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Raw model outputs.
        """
        if self.n_classes == 2:
            # Binary classification: return single output
            return self.trees[0].predict(X)
        else:
            # Multiclass: return predictions from all trees
            class_outputs = [tree.predict(X) for tree in self.trees]
            return torch.stack(class_outputs, dim=1)

    def predict_proba(self, X):
        """
        Get probability predictions.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Probability predictions for each class.
        """
        raw_preds = self.predict_raw(X)

        if self.n_classes == 2:
            # Binary case: apply sigmoid
            probs = torch.sigmoid(raw_preds)
            return torch.stack([1 - probs, probs], dim=1)
        else:
            # Multiclass case: apply softmax
            return torch.softmax(raw_preds, dim=1)

    def predict(self, X, threshold=0.5):
        """
        Make class predictions.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        threshold : float, optional
            Threshold for binary classification (default is 0.5).

        Returns
        -------
        torch.Tensor
            Predicted class indices.
        """
        probs = self.predict_proba(X)

        if self.n_classes == 2:
            return (probs[:, 1] > threshold).long()
        else:
            return torch.argmax(probs, dim=1)

    def print_tree_representation(self):
        """
        Print all trees in the model for interpretability.
        """
        if self.n_classes == 2:
            print("Binary Classification Tree:")
            self.trees[0].print_tree_representation()
        else:
            for i, tree in enumerate(self.trees):
                class_label = self.class_labels[i] if self.class_labels else i
                print(f"Tree for class {class_label}:")
                tree.print_tree_representation()

    def get_tree_representation(self, indent=""):
        """
        Returns the tree representation as a string with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        str
            Returns the tree representation with the chosen indentation.
        """
        # For binary classification, get representation from the first tree
        if self.n_classes == 2:
            return self.trees[0].get_tree_representation(indent)
        else:
            # For multiclass, combine representations
            representations = []
            for i, tree in enumerate(self.trees):
                class_label = self.class_labels[i] if self.class_labels else i
                representations.append(f"Tree for class {class_label}:\n")
                representations.append(tree.get_tree_representation(indent))
            return "".join(representations)


def evaluate_classification_model(model, X, y, threshold=0.5, class_labels=None):
    """
    Comprehensive evaluation function for classification models.

    Parameters
    ----------
    model : ClassificationTreeWrapper or similar
        The classification model to evaluate.
    X : torch.Tensor
        Input features.
    y : torch.Tensor
        True class labels.
    threshold : float, optional
        Decision threshold for binary classification (default is 0.5).
    class_labels : list, optional
        Optional list of class labels for better reporting.

    Returns
    -------
    dict
        Dictionary containing various classification metrics.
    """
    # Determine number of classes
    n_classes = len(torch.unique(y)) if class_labels is None else len(class_labels)

    # Make predictions
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        preds = model.predict(X, threshold=threshold)
    else:
        # Basic compatibility with models that only have predict
        preds = model.predict(X)
        probs = None

    # Convert to numpy for sklearn metrics
    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds

    # Calculate common metrics
    metrics = {'accuracy': accuracy_score(y_np, preds_np)}

    if n_classes == 2:
        # Binary classification specific metrics
        metrics.update({
            'precision': precision_score(y_np, preds_np, zero_division=0),
            'recall': recall_score(y_np, preds_np, zero_division=0),
            'f1': f1_score(y_np, preds_np, zero_division=0),
            'confusion_matrix': confusion_matrix(y_np, preds_np)
        })

        # Add AUC if probabilities are available
        if probs is not None:
            probs_np = probs[:, 1].cpu().numpy() if isinstance(probs, torch.Tensor) else probs[:, 1]
            metrics['auc'] = roc_auc_score(y_np, probs_np)
    else:
        # Multiclass classification specific metrics
        metrics.update({
            'precision_macro': precision_score(y_np, preds_np, average='macro', zero_division=0),
            'recall_macro': recall_score(y_np, preds_np, average='macro', zero_division=0),
            'f1_macro': f1_score(y_np, preds_np, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_np, preds_np, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_np, preds_np, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_np, preds_np, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_np, preds_np),
            'classification_report': classification_report(
                y_np, preds_np, target_names=class_labels, zero_division=0
            )
        })

    return metrics


def classification_metrics(y_true, y_pred, n_classes=2, class_labels=None, prefix='val_'):
    """
    Compute classification metrics from true and predicted labels.

    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        True class labels.
    y_pred : torch.Tensor or np.ndarray
        Predicted class labels.
    n_classes : int, optional
        Number of classes (default is 2 for binary classification).
    class_labels : list, optional
        Optional list of class labels for better reporting.
    prefix : str, optional
        Prefix to add to metric names (default is 'val_').

    Returns
    -------
    dict
        Dictionary of classification metrics.
    """
    # Convert to numpy if tensors
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    metrics = {
        f'{prefix}accuracy': accuracy_score(y_true_np, y_pred_np)
    }

    if n_classes == 2:
        # Binary classification metrics
        metrics.update({
            f'{prefix}precision': precision_score(y_true_np, y_pred_np, zero_division=0),
            f'{prefix}recall': recall_score(y_true_np, y_pred_np, zero_division=0),
            f'{prefix}f1': f1_score(y_true_np, y_pred_np, zero_division=0)
        })
    else:
        # Multiclass classification metrics
        metrics.update({
            f'{prefix}precision_macro': precision_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            f'{prefix}recall_macro': recall_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            f'{prefix}f1_macro': f1_score(y_true_np, y_pred_np, average='macro', zero_division=0),
            f'{prefix}precision_weighted': precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0),
            f'{prefix}recall_weighted': recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0),
            f'{prefix}f1_weighted': f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
        })

    return metrics


def classification_accuracy_fitness(y_true, y_pred):
    """
    Fitness function based on classification accuracy.

    This function is designed for GP minimization, so it returns negative accuracy
    (since higher accuracy is better, but GP minimizes the fitness).

    Parameters
    ----------
    y_true : torch.Tensor
        True class labels.
    y_pred : torch.Tensor
        Predicted logits.

    Returns
    -------
    float
        Negative accuracy (for minimization).
    """
    with torch.no_grad():
        if len(y_pred.shape) > 1:  # Multiclass case
            predictions = torch.argmax(y_pred, dim=1)
        else:  # Binary case
            predictions = (torch.sigmoid(y_pred) > 0.5).long()

        accuracy = (predictions == y_true).float().mean()
        # For minimization problems, return negative accuracy
        return -accuracy


def f1_score_fitness(y_true, y_pred):
    """
    Fitness function based on F1 score.

    This function is designed for GP minimization, so it returns negative F1 score
    (since higher F1 is better, but GP minimizes the fitness).

    Parameters
    ----------
    y_true : torch.Tensor
        True class labels.
    y_pred : torch.Tensor
        Predicted logits.

    Returns
    -------
    float
        Negative F1 score (for minimization).
    """
    with torch.no_grad():
        if len(y_pred.shape) > 1:  # Multiclass case
            predictions = torch.argmax(y_pred, dim=1)
        else:  # Binary case
            predictions = (torch.sigmoid(y_pred) > 0.5).long()

        # Calculate precision and recall
        tp = torch.sum((predictions == 1) & (y_true == 1)).float()
        fp = torch.sum((predictions == 1) & (y_true == 0)).float()
        fn = torch.sum((predictions == 0) & (y_true == 1)).float()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        # For minimization, return negative F1
        return -f1


def binary_cross_entropy_with_logits(y_true, y_pred):
    """
    Custom binary cross entropy loss for classification.

    This function handles type conversion and ensures compatibility with the GP framework.

    Parameters
    ----------
    y_true : torch.Tensor
        True class labels.
    y_pred : torch.Tensor
        Predicted logits.

    Returns
    -------
    float
        Binary cross entropy loss.
    """
    # Ensure y_true is float type
    y_true = y_true.float() if isinstance(y_true, torch.Tensor) else torch.tensor(y_true, dtype=torch.float32)

    # Compute the loss using torch's functional API
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction='mean'
    )

    return loss


def create_balanced_data(X, y, n_classes=None, balance_strategy='undersample'):
    """
    Create a balanced dataset for classification.

    Parameters
    ----------
    X : torch.Tensor
        Input features.
    y : torch.Tensor
        Class labels.
    n_classes : int, optional
        Number of classes. If None, inferred from y.
    balance_strategy : str, optional
        Strategy for balancing: 'undersample' or 'oversample'.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Balanced X and y tensors.
    """
    if n_classes is None:
        n_classes = len(torch.unique(y))

    # Group indices by class
    indices_by_class = [torch.where(y == c)[0] for c in range(n_classes)]
    class_sizes = [len(indices) for indices in indices_by_class]

    if balance_strategy == 'undersample':
        # Undersample to the smallest class size
        target_size = min(class_sizes)
        balanced_indices = []

        for class_indices in indices_by_class:
            if len(class_indices) > target_size:
                # Randomly select 'target_size' samples
                selected = torch.randperm(len(class_indices))[:target_size]
                balanced_indices.append(class_indices[selected])
            else:
                balanced_indices.append(class_indices)

    elif balance_strategy == 'oversample':
        # Oversample to the largest class size
        target_size = max(class_sizes)
        balanced_indices = []

        for class_indices in indices_by_class:
            if len(class_indices) < target_size:
                # Sample with replacement to match target_size
                replacement_size = target_size - len(class_indices)
                replacement_indices = class_indices[torch.randint(len(class_indices), (replacement_size,))]
                balanced_indices.append(torch.cat([class_indices, replacement_indices]))
            else:
                balanced_indices.append(class_indices)

    # Combine all indices and shuffle
    all_indices = torch.cat(balanced_indices)
    shuffled_indices = all_indices[torch.randperm(len(all_indices))]

    return X[shuffled_indices], y[shuffled_indices]


def calculate_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies.

    Parameters
    ----------
    y : torch.Tensor
        Class labels.

    Returns
    -------
    torch.Tensor
        Weights for each class.
    """
    class_counts = torch.bincount(y)
    n_samples = len(y)
    n_classes = len(class_counts)

    # Weight is inversely proportional to class frequency
    weights = n_samples / (n_classes * class_counts.float())
    return weights


def convert_to_one_vs_rest(y, target_class):
    """
    Convert multiclass labels to binary labels for one-vs-rest classification.

    Parameters
    ----------
    y : torch.Tensor
        Original multiclass labels.
    target_class : int
        The positive class index.

    Returns
    -------
    torch.Tensor
        Binary labels (1 for target_class, 0 for all others) as float tensor.
    """
    # Always return float type for binary cross entropy compatibility
    return (y == target_class).float()