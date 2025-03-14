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
Binary Classification implementations for SLIM_GSGP.

This module provides specialized implementations for binary classification
tasks using Genetic Programming.
"""

import torch
import warnings
from slim_gsgp.main_gp import gp
from slim_gsgp.main_gsgp import gsgp
from slim_gsgp.main_slim import slim
from slim_gsgp.classifiers.classification_utils import (
    ClassificationTreeWrapper,
    create_balanced_data,
    calculate_class_weights
)
from slim_gsgp.tree_visualizer import visualize_gp_tree


class BinaryClassifier:
    """
    Binary classifier implementation using Genetic Programming.

    This class wraps the standard GP implementation with additional functionality
    specific to binary classification tasks.

    Attributes
    ----------
    model : ClassificationTreeWrapper
        The wrapped GP model used for classification.
    fitness_function : str
        The fitness function used for training.
    threshold : float
        The classification threshold (default is 0.5).
    """

    def __init__(self, model, fitness_function='binary_cross_entropy', threshold=0.5):
        """
        Initialize a binary classifier.

        Parameters
        ----------
        model : object
            The GP model to be wrapped.
        fitness_function : str
            The fitness function used for training.
        threshold : float, optional
            The classification threshold (default is 0.5).
        """
        self.model = ClassificationTreeWrapper(model, n_classes=2)
        self.fitness_function = fitness_function
        self.threshold = threshold

    def predict(self, X, threshold=None):
        """
        Predict class labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        threshold : float, optional
            Threshold for binary classification. If None, use self.threshold.

        Returns
        -------
        torch.Tensor
            Predicted class labels (0 or 1).
        """
        if threshold is None:
            threshold = self.threshold
        return self.model.predict(X, threshold=threshold)

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Predicted class labels (0 or 1).
        """
        return self.model.predict(X, threshold=self.threshold)

    def print_tree_representation(self):
        """
        Print the tree representation for interpretability.
        """
        self.model.print_tree_representation()

    def visualize_tree(self, filename='gp_tree', format='png', dataset=None, algorithm=None):
        """Create a visual representation of the model's tree.

        Parameters:
        -----------
        filename : str
            Base filename or full path for the output file
        format : str
            Output format ('png', 'svg', etc.)
        dataset : str, optional
            Dataset name for directory organization
        algorithm : str, optional
            Algorithm name for directory organization

        Returns:
        --------
        str
            Path to the visualization file
        """
        # Get the tree structure directly
        tree_structure = self.model.get_tree_structure()

        # If dataset and algorithm are provided, use them for proper directory structure
        if dataset and algorithm:
            return visualize_gp_tree(tree_structure, filename, format, dataset, algorithm)
        else:
            # Otherwise just use the filename as is
            return visualize_gp_tree(tree_structure, filename, format)


def train_binary_classifier(
        X_train, y_train, X_val=None, y_val=None,
        algo_type='gp',
        fitness_function='binary_cross_entropy',
        balance_data=False,
        class_weight=False,
        **kwargs
):
    """
    Train a binary classifier using Genetic Programming.

    Parameters
    ----------
    X_train : torch.Tensor
        Training features.
    y_train : torch.Tensor
        Training labels (0 or 1).
    X_val : torch.Tensor, optional
        Validation features.
    y_val : torch.Tensor, optional
        Validation labels.
    algo_type : str, optional
        Algorithm type: 'gp', 'gsgp', or 'slim' (default is 'gp').
    fitness_function : str, optional
        Fitness function for training (default is 'binary_cross_entropy').
    balance_data : bool or str, optional
        Whether to balance training data. If string, specifies strategy ('undersample' or 'oversample').
    class_weight : bool, optional
        Whether to use class weights for imbalanced data.
    **kwargs
        Additional arguments passed to the underlying algorithm.

    Returns
    -------
    BinaryClassifier
        Trained binary classifier.
    """
    # Ensure binary labels
    if len(torch.unique(y_train)) > 2:
        raise ValueError("Training labels must be binary (0 or 1).")

    # Convert targets to float for binary cross entropy
    y_train = y_train.float()
    if y_val is not None:
        y_val = y_val.float()

    # Handle class imbalance if requested
    if balance_data:
        strategy = 'undersample' if balance_data is True else balance_data
        X_train, y_train = create_balanced_data(
            X_train, y_train, n_classes=2, balance_strategy=strategy
        )

    # Use class weights if requested (currently just a placeholder for future extension)
    if class_weight:
        weights = calculate_class_weights(y_train)
        print(f"Using class weights: {weights.tolist()}")
        # Note: The current implementation doesn't directly support class weights
        # This would need to be implemented in the fitness functions

    # Train the model using the specified algorithm
    if algo_type.lower() == 'gp':
        model = gp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            fitness_function=fitness_function,
            **kwargs
        )
    elif algo_type.lower() == 'gsgp':
        # Remove max_depth from kwargs if it exists
        gsgp_kwargs = {k: v for k, v in kwargs.items() if k != 'max_depth'}
        gsgp_kwargs['reconstruct'] = True
        model = gsgp(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            fitness_function=fitness_function,
            **gsgp_kwargs
        )
    elif algo_type.lower() == 'slim':
        model = slim(
            X_train=X_train,
            y_train=y_train,
            X_test=X_val,
            y_test=y_val,
            fitness_function=fitness_function,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algo_type}. Use 'gp', 'gsgp', or 'slim'.")

    # Wrap the model in a BinaryClassifier
    return BinaryClassifier(
        model=model,
        fitness_function=fitness_function
    )
