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
Multiclass Classification implementations for SLIM_GSGP.

This module provides specialized implementations for multiclass classification
tasks using Genetic Programming through various strategies like One-vs-Rest,
One-vs-One, and direct multiclass approaches.
"""

import itertools
import torch
import warnings
from concurrent.futures import ProcessPoolExecutor
from slim_gsgp.main_gp import gp
from slim_gsgp.main_gsgp import gsgp
from slim_gsgp.main_slim import slim
from slim_gsgp.classifiers.classification_utils import (
    ClassificationTreeWrapper,
    convert_to_one_vs_rest,
    create_balanced_data
)
from slim_gsgp.tree_visualizer import visualize_gp_tree


class MulticlassClassifier:
    """
    Multiclass classifier implementation using Genetic Programming.

    This class provides a unified interface for multiclass classification,
    regardless of the underlying strategy (One-vs-Rest, One-vs-One, etc.).

    Attributes
    ----------
    model : ClassificationTreeWrapper
        The wrapped model used for classification.
    n_classes : int
        Number of classes.
    strategy : str
        The multiclass strategy used ('ovr', 'ovo', or 'direct').
    class_labels : list or None
        Optional class labels for interpretability.
    """

    def __init__(self, model, n_classes, strategy='ovr', class_labels=None):
        """
        Initialize a multiclass classifier.

        Parameters
        ----------
        model : list or object
            The GP model(s) to wrap.
        n_classes : int
            Number of classes.
        strategy : str, optional
            The multiclass strategy ('ovr', 'ovo', or 'direct').
        class_labels : list, optional
            Optional class labels for interpretability.
        """
        self.model = ClassificationTreeWrapper(model, n_classes=n_classes, class_labels=class_labels)
        self.n_classes = n_classes
        self.strategy = strategy
        self.class_labels = class_labels

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Class probabilities (shape: [n_samples, n_classes]).
        """
        return self.model.predict_proba(X)

    def predict(self, X, threshold=None):
        """
        Predict class labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        threshold : float, optional
            Threshold for binary classification (ignored for multiclass).

        Returns
        -------
        torch.Tensor
            Predicted class labels (0 to n_classes-1).
        """
        # For multiclass, threshold is only relevant for binary submodels
        if self.strategy == 'ovr' or self.strategy == 'ovo':
            return self.model.predict(X, threshold=threshold)
        else:
            return self.model.predict(X)

    def print_tree_representation(self):
        """
        Print the tree representation for interpretability.
        """
        self.model.print_tree_representation()

    def visualize_tree(self, filename='multiclass_model', format='png', dataset=None, algorithm=None):
        """
        Create visual representations of the model's trees.

        Parameters:
        -----------
        filename : str
            Output filename (without extension)
        format : str
            Output format ('png', 'svg', 'pdf', etc.)
        dataset : str, optional
            Dataset name for directory organization
        algorithm : str, optional
            Algorithm name for directory organization

        Returns:
        --------
        list
            Paths to the generated visualization files
        """
        visualization_paths = []

        if self.strategy == 'ovr':
            # For one-vs-rest, create a visualization for each class
            for i, tree in enumerate(self.model.trees):
                class_name = self.class_labels[i] if self.class_labels else f"class_{i}"
                class_filename = f"{filename}_{class_name}"
                tree_str = tree.get_tree_representation()
                path = visualize_gp_tree(tree_str, class_filename, format, dataset, algorithm)
                visualization_paths.append(path)

        elif self.strategy == 'ovo':
            # For one-vs-one, we need special handling for the wrapper
            if hasattr(self.model.trees[0], 'pair_models'):
                # Access the pair models from the wrapper
                for i, ((class1, class2), model) in enumerate(self.model.trees[0].pair_models):
                    class1_name = self.class_labels[class1] if self.class_labels else f"class_{class1}"
                    class2_name = self.class_labels[class2] if self.class_labels else f"class_{class2}"
                    class_filename = f"{filename}_{class1_name}_vs_{class2_name}"
                    tree_str = model.get_tree_representation()
                    path = visualize_gp_tree(tree_str, class_filename, format, dataset, algorithm)
                    visualization_paths.append(path)
        else:
            # For direct strategy, visualize all trees
            for i, tree in enumerate(self.model.trees):
                tree_filename = f"{filename}_tree_{i}"
                tree_str = tree.get_tree_representation()
                path = visualize_gp_tree(tree_str, tree_filename, format, dataset, algorithm)
                visualization_paths.append(path)

        return visualization_paths


def train_multiclass_classifier(
        X_train, y_train, X_val=None, y_val=None,
        strategy='ovr', algo_type='gp',
        fitness_function='binary_cross_entropy',
        balance_data=False,
        n_classes=None,
        class_labels=None,
        parallel=False,
        **kwargs
):
    """
    Train a multiclass classifier using Genetic Programming.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels (0 to n_classes-1).
    X_val : torch.Tensor, optional
        Validation features.
    y_val : torch.Tensor, optional
        Validation labels.
    strategy : str, optional
        Multiclass strategy: 'ovr' (One-vs-Rest), 'ovo' (One-vs-One), or 'direct'.
    algo_type : str, optional
        Algorithm type: 'gp', 'gsgp', or 'slim' (default is 'gp').
    fitness_function : str, optional
        Fitness function for training.
    balance_data : bool or str, optional
        Whether to balance training data. If string, specifies strategy.
    n_classes : int, optional
        Number of classes. If None, inferred from y_train.
    class_labels : list, optional
        Optional class labels for interpretability.
    parallel : bool, optional
        Whether to train models in parallel (for ovr or ovo strategies).
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    MulticlassClassifier
        Trained multiclass classifier.
    """
    # Determine number of classes if not provided
    if n_classes is None:
        n_classes = len(torch.unique(y_train))

    # Select the appropriate training function based on strategy
    if strategy.lower() == 'ovr':
        # One-vs-Rest strategy
        return train_one_vs_rest_classifier(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            algo_type=algo_type, fitness_function=fitness_function,
            balance_data=balance_data, n_classes=n_classes,
            class_labels=class_labels, parallel=parallel, **kwargs
        )
    elif strategy.lower() == 'ovo':
        # One-vs-One strategy
        return train_one_vs_one_classifier(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            algo_type=algo_type, fitness_function=fitness_function,
            balance_data=balance_data, n_classes=n_classes,
            class_labels=class_labels, parallel=parallel, **kwargs
        )
    elif strategy.lower() == 'direct':
        # Direct multiclass strategy (not fully implemented)
        warnings.warn(
            "Direct multiclass strategy is experimental and may not work well. "
            "Consider using 'ovr' or 'ovo' strategies instead."
        )

        # For direct strategy, we need to use a multi-output fitness function
        # like cross-entropy instead of binary cross-entropy
        if fitness_function == 'binary_cross_entropy':
            fitness_function = 'cross_entropy'

        # This is a placeholder for future implementation
        # Currently, just use one-vs-rest as fallback
        return train_one_vs_rest_classifier(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            algo_type=algo_type, fitness_function=fitness_function,
            balance_data=balance_data, n_classes=n_classes,
            class_labels=class_labels, parallel=parallel, **kwargs
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'ovr', 'ovo', or 'direct'.")


def train_one_vs_rest_classifier(
        X_train, y_train, X_val=None, y_val=None,
        algo_type='gp', fitness_function='binary_cross_entropy',
        balance_data=False, n_classes=None, class_labels=None,
        parallel=False, **kwargs
):
    """
    Train a One-vs-Rest multiclass classifier.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels (0 to n_classes-1).
    X_val : torch.Tensor, optional
        Validation features.
    y_val : torch.Tensor, optional
        Validation labels.
    algo_type : str, optional
        Algorithm type: 'gp', 'gsgp', or 'slim'.
    fitness_function : str, optional
        Fitness function for training.
    balance_data : bool or str, optional
        Whether to balance training data. If string, specifies strategy.
    n_classes : int, optional
        Number of classes. If None, inferred from y_train.
    class_labels : list, optional
        Optional class labels for interpretability.
    parallel : bool, optional
        Whether to train models in parallel.
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    MulticlassClassifier
        Trained One-vs-Rest multiclass classifier.
    """
    # Determine number of classes if not provided
    if n_classes is None:
        n_classes = len(torch.unique(y_train))

    # Define function to train a single binary classifier for one class
    def train_for_class(class_idx):
        # Convert to binary problem (target class vs rest)
        # Result will already be float type from convert_to_one_vs_rest
        y_train_binary = convert_to_one_vs_rest(y_train, class_idx)

        if X_val is not None and y_val is not None:
            y_val_binary = convert_to_one_vs_rest(y_val, class_idx)
        else:
            y_val_binary = None

        # Balance data if requested
        if balance_data:
            strategy = 'undersample' if balance_data is True else balance_data
            X_train_balanced, y_train_binary_balanced = create_balanced_data(
                X_train, y_train_binary, n_classes=2, balance_strategy=strategy
            )
        else:
            X_train_balanced, y_train_binary_balanced = X_train, y_train_binary

        # Set dataset name for logging
        dataset_name = kwargs.get('dataset_name', 'multiclass')
        class_name = class_labels[class_idx] if class_labels else f"class_{class_idx}"
        kwargs['dataset_name'] = f"{dataset_name}_{class_name}"

        # Train the binary classifier
        if algo_type.lower() == 'gp':
            model = gp(
                X_train=X_train_balanced,
                y_train=y_train_binary_balanced,
                X_test=X_val,
                y_test=y_val_binary,
                fitness_function=fitness_function,
                **kwargs
            )
        elif algo_type.lower() == 'gsgp':
            model = gsgp(
                X_train=X_train_balanced,
                y_train=y_train_binary_balanced,
                X_test=X_val,
                y_test=y_val_binary,
                fitness_function=fitness_function,
                **kwargs
            )
        elif algo_type.lower() == 'slim':
            model = slim(
                X_train=X_train_balanced,
                y_train=y_train_binary_balanced,
                X_test=X_val,
                y_test=y_val_binary,
                fitness_function=fitness_function,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}. Use 'gp', 'gsgp', or 'slim'.")

        return model

    # Train binary classifiers for each class
    if parallel and n_classes > 2:
        print(f"Training {n_classes} binary classifiers in parallel...")
        with ProcessPoolExecutor() as executor:
            models = list(executor.map(train_for_class, range(n_classes)))
    else:
        models = []
        for class_idx in range(n_classes):
            print(f"Training binary classifier for class {class_idx}...")
            models.append(train_for_class(class_idx))

    # Return a MulticlassClassifier with all trained models
    return MulticlassClassifier(
        model=models,
        n_classes=n_classes,
        strategy='ovr',
        class_labels=class_labels
    )


def train_one_vs_one_classifier(
        X_train, y_train, X_val=None, y_val=None,
        algo_type='gp', fitness_function='binary_cross_entropy',
        balance_data=False, n_classes=None, class_labels=None,
        parallel=False, **kwargs
):
    """
    Train a One-vs-One multiclass classifier.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels (0 to n_classes-1).
    X_val : torch.Tensor, optional
        Validation features.
    y_val : torch.Tensor, optional
        Validation labels.
    algo_type : str, optional
        Algorithm type: 'gp', 'gsgp', or 'slim'.
    fitness_function : str, optional
        Fitness function for training.
    balance_data : bool or str, optional
        Whether to balance training data. If string, specifies strategy.
    n_classes : int, optional
        Number of classes. If None, inferred from y_train.
    class_labels : list, optional
        Optional class labels for interpretability.
    parallel : bool, optional
        Whether to train models in parallel.
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    MulticlassClassifier
        Trained One-vs-One multiclass classifier.
    """
    # Determine number of classes if not provided
    if n_classes is None:
        n_classes = len(torch.unique(y_train))

    # For One-vs-One, we need to train a binary classifier for each pair of classes
    class_pairs = list(itertools.combinations(range(n_classes), 2))
    num_pairs = len(class_pairs)

    # Display warning for large number of pairs
    if num_pairs > 10:
        warnings.warn(
            f"One-vs-One strategy requires training {num_pairs} classifiers. "
            "This might be computationally expensive. Consider using 'ovr' strategy instead."
        )

    # Function to train a classifier for a single pair of classes
    def train_for_pair(pair_idx):
        class1, class2 = class_pairs[pair_idx]

        # Extract samples belonging to the current pair of classes
        mask_train = (y_train == class1) | (y_train == class2)
        X_pair_train = X_train[mask_train]
        y_pair_train = y_train[mask_train]

        # Remap labels to 0 and 1 for binary classification
        y_pair_train = (y_pair_train == class2).float()

        if X_val is not None and y_val is not None:
            mask_val = (y_val == class1) | (y_val == class2)
            X_pair_val = X_val[mask_val]
            y_pair_val = (y_val[mask_val] == class2).float()
        else:
            X_pair_val, y_pair_val = None, None

        # Balance data if requested
        if balance_data:
            strategy = 'undersample' if balance_data is True else balance_data
            X_pair_train, y_pair_train = create_balanced_data(
                X_pair_train, y_pair_train, n_classes=2, balance_strategy=strategy
            )

        # Set dataset name for logging
        dataset_name = kwargs.get('dataset_name', 'multiclass')
        class1_name = class_labels[class1] if class_labels else f"class_{class1}"
        class2_name = class_labels[class2] if class_labels else f"class_{class2}"
        kwargs['dataset_name'] = f"{dataset_name}_{class1_name}_vs_{class2_name}"

        # Train the binary classifier
        if algo_type.lower() == 'gp':
            model = gp(
                X_train=X_pair_train,
                y_train=y_pair_train,
                X_test=X_pair_val,
                y_test=y_pair_val,
                fitness_function=fitness_function,
                **kwargs
            )
        elif algo_type.lower() == 'gsgp':
            model = gsgp(
                X_train=X_pair_train,
                y_train=y_pair_train,
                X_test=X_pair_val,
                y_test=y_pair_val,
                fitness_function=fitness_function,
                **kwargs
            )
        elif algo_type.lower() == 'slim':
            model = slim(
                X_train=X_pair_train,
                y_train=y_pair_train,
                X_test=X_pair_val,
                y_test=y_pair_val,
                fitness_function=fitness_function,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}. Use 'gp', 'gsgp', or 'slim'.")

        return ((class1, class2), model)

    # Train binary classifiers for each pair of classes
    if parallel and num_pairs > 1:
        print(f"Training {num_pairs} binary classifiers in parallel...")
        with ProcessPoolExecutor() as executor:
            pair_models = list(executor.map(train_for_pair, range(num_pairs)))
    else:
        pair_models = []
        for pair_idx in range(num_pairs):
            class1, class2 = class_pairs[pair_idx]
            print(f"Training binary classifier for classes {class1} vs {class2}...")
            pair_models.append(train_for_pair(pair_idx))

    # Create a custom wrapper for One-vs-One prediction
    class OneVsOneWrapper:
        def __init__(self, pair_models, n_classes):
            self.pair_models = pair_models
            self.n_classes = n_classes

        def predict(self, X):
            # Count votes for each class
            votes = torch.zeros((len(X), n_classes))

            for (class1, class2), model in self.pair_models:
                # Get predictions for this pair
                probs = torch.sigmoid(model.predict(X))

                # Class1 gets votes where probability < 0.5, Class2 where probability >= 0.5
                votes[:, class1] += (probs < 0.5).float()
                votes[:, class2] += (probs >= 0.5).float()

            # Return class with most votes
            return torch.argmax(votes, dim=1)

        def predict_proba(self, X):
            # This is a simplified implementation of probability estimation
            # A more sophisticated approach would use proper calibration
            votes = torch.zeros((len(X), n_classes))
            counts = torch.zeros((len(X), n_classes))

            for (class1, class2), model in self.pair_models:
                # Get probabilities for this pair
                probs = torch.sigmoid(model.predict(X))

                # Add probabilistic votes
                votes[:, class1] += 1 - probs
                votes[:, class2] += probs

                # Count number of classifiers involving each class
                counts[:, class1] += 1
                counts[:, class2] += 1

            # Normalize votes by counts to get approximate probabilities
            # Add small epsilon to avoid division by zero
            return votes / (counts + 1e-10)

        def print_tree_representation(self):
            for i, ((class1, class2), model) in enumerate(self.pair_models):
                print(f"Model {i + 1}: Class {class1} vs Class {class2}")
                model.print_tree_representation()
                print("\n")

        def get_tree_representation(self, indent=""):
            """Return a string representation of all binary classifiers in the wrapper."""
            representations = []
            for i, ((class1, class2), model) in enumerate(self.pair_models):
                representations.append(f"{indent}Model {i + 1}: Class {class1} vs Class {class2}\n")
                if hasattr(model, 'get_tree_representation'):
                    representations.append(model.get_tree_representation(indent + "  "))
                representations.append("\n")
            return "".join(representations)

    # Create the wrapper
    ovo_wrapper = OneVsOneWrapper([pm for pm in pair_models], n_classes)

    # Return a MulticlassClassifier with the OneVsOneWrapper
    return MulticlassClassifier(
        model=ovo_wrapper,
        n_classes=n_classes,
        strategy='ovo',
        class_labels=class_labels
    )