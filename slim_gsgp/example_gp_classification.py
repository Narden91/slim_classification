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
Enhanced example script for classification using the SLIM-GSGP framework.

This script demonstrates how to use the specialized classification modules
for both binary and multiclass classification problems.
"""

import argparse
import time
import torch
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler

from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.classifiers.binary_classifiers import train_binary_classifier
from slim_gsgp.classifiers.multiclass_classifiers import train_multiclass_classifier
from slim_gsgp.classifiers.classification_utils import (
    evaluate_classification_model,
    binary_cross_entropy_with_logits
)


def load_dataset(dataset_name):
    """
    Load and preprocess a dataset for classification.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load ('breast_cancer', 'iris', 'digits', or 'wine').

    Returns
    -------
    tuple
        (X, y, n_classes, class_labels) where X and y are the features and labels,
        n_classes is the number of classes, and class_labels are the class names.
    """
    # Select and load the dataset
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = data.data
        y = data.target
        class_labels = data.target_names.tolist()
    elif dataset_name == 'iris':
        data = load_iris()
        X = data.data
        y = data.target
        class_labels = data.target_names.tolist()
    elif dataset_name == 'digits':
        data = load_digits()
        X = data.data
        y = data.target
        class_labels = [str(i) for i in range(10)]  # Digit names (0-9)
    elif dataset_name == 'wine':
        data = load_wine()
        X = data.data
        y = data.target
        class_labels = ["Class " + str(i) for i in range(len(data.target_names))]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Perform feature scaling (standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Determine number of classes
    n_classes = len(torch.unique(y))

    # For both binary and multiclass problems, we want long integers for y
    # The classification modules will handle conversion to float when needed
    y = y.long()

    return X, y, n_classes, class_labels


def main():
    """
    Main function to run the classification example.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SLIM-GSGP Classification Example')
    parser.add_argument('--dataset', type=str, default='breast_cancer',
                        choices=['breast_cancer', 'iris', 'digits', 'wine'],
                        help='Dataset to use for classification')
    parser.add_argument('--algo', type=str, default='gp',
                        choices=['gp', 'gsgp', 'slim'],
                        help='GP algorithm to use')
    parser.add_argument('--strategy', type=str, default='ovr',
                        choices=['ovr', 'ovo'],
                        help='Multiclass strategy (One-vs-Rest or One-vs-One)')
    parser.add_argument('--balance', action='store_true',
                        help='Whether to balance the dataset')
    parser.add_argument('--pop-size', type=int, default=50,
                        help='Population size for GP')
    parser.add_argument('--n-iter', type=int, default=20,
                        help='Number of iterations for GP')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel training for multiclass')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print(f"Running classification example with:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Balance data: {args.balance}")
    print(f"  Population size: {args.pop_size}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Parallel: {args.parallel}")
    print()

    # Load the dataset
    print(f"Loading dataset: {args.dataset}")
    X, y, n_classes, class_labels = load_dataset(args.dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    print()

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=args.seed)

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Common parameters for GP
    gp_params = {
        'pop_size': args.pop_size,
        'n_iter': args.n_iter,
        'max_depth': 8,
        'seed': args.seed,
        'dataset_name': args.dataset,
        'fitness_function': 'binary_cross_entropy'  # Use custom function
    }

    # Train the classifier
    start_time = time.time()

    if n_classes == 2:
        # Binary classification
        print("Training binary classifier...")
        model = train_binary_classifier(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            algo_type=args.algo,
            balance_data=args.balance,
            **gp_params
        )
    else:
        # Multiclass classification
        print(f"Training multiclass classifier using {args.strategy} strategy...")
        model = train_multiclass_classifier(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy=args.strategy,
            algo_type=args.algo,
            balance_data=args.balance,
            n_classes=n_classes,
            class_labels=class_labels,
            parallel=args.parallel,
            **gp_params
        )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print()

    # Evaluate on test set
    print("Evaluating on test set:")
    metrics = evaluate_classification_model(
        model=model,
        X=X_test,
        y=y_test,
        class_labels=class_labels
    )

    # Print metrics
    for name, value in metrics.items():
        if name not in ['confusion_matrix', 'classification_report']:
            print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    if n_classes > 2 and 'classification_report' in metrics:
        print("\nClassification Report:")
        print(metrics['classification_report'])

    # Print tree representation
    print("\nModel Tree Representation:")
    model.print_tree_representation()

    visualization_path = model.visualize_tree(filename='breast_cancer_model')
    print(f"Tree visualization saved to: {visualization_path}")


if __name__ == "__main__":
    main()