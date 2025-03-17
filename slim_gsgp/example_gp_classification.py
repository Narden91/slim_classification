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
Example script for classification using the SLIM-GSGP framework.

This script demonstrates how to use the specialized binary classification module
with integration of tree visualization.
"""
import time
import os
import uuid

import torch

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset
from slim_gsgp.utils.binary_classification import (
    train_binary_classifier,
    register_binary_fitness_functions,
    BinaryClassifier
)
from slim_gsgp.tree_visualizer import visualize_gp_tree, visualize_classification_model


def main():
    """
    Main function to run the binary classification example.
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Define parameters directly
    dataset = 'breast_cancer'  # Options: 'breast_cancer', 'iris', 'digits', 'wine'
    algo = 'gp'  # Options: 'gp', 'gsgp', 'slim'
    pop_size = 50  # Integer
    n_iter = 20  # Integer
    seed = 42  # Integer
    use_sigmoid = True  # Whether to use sigmoid activation
    sigmoid_scale = 1.0  # Scaling factor for sigmoid
    fitness_function = 'binary_rmse'  # 'binary_rmse', 'binary_mse', or 'binary_mae'

    # Register binary fitness functions
    register_binary_fitness_functions()

    # Set random seed
    torch.manual_seed(seed)

    print(f"Running binary classification example with:")
    print(f"  Dataset: {dataset}")
    print(f"  Algorithm: {algo}")
    print(f"  Population size: {pop_size}")
    print(f"  Iterations: {n_iter}")
    print(f"  Fitness function: {fitness_function}")
    print(f"  Use sigmoid: {use_sigmoid}")
    print(f"  Sigmoid scale: {sigmoid_scale}")
    print()

    # Load the dataset
    print(f"Loading dataset: {dataset}")
    X, y, n_classes, class_labels = load_classification_dataset(dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    print()

    # Check if dataset is binary
    if n_classes != 2:
        raise ValueError(f"This example is for binary classification only. Dataset {dataset} has {n_classes} classes.")

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=seed)

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Set algorithm-specific parameters
    algo_params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'seed': seed,
        'dataset_name': dataset,
        'max_depth': 8,
    }

    # Create log directory if needed
    log_dir = os.path.join(root_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if algo == 'gsgp':
        # For GSGP, ensure reconstruct=True to enable prediction
        algo_params['reconstruct'] = True
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(log_dir, f"gsgp_{seed}.csv")
    elif algo == 'slim':
        # For SLIM, set appropriate version
        algo_params['slim_version'] = 'SLIM+ABS'
        algo_params['p_inflate'] = 0.5
        algo_params['ms_lower'] = 0
        algo_params['ms_upper'] = 1
        algo_params['log_path'] = os.path.join(log_dir, f"slim_{seed}.csv")

    # Train the classifier
    start_time = time.time()

    print("Training binary classifier...")
    model = train_binary_classifier(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        algorithm=algo,
        use_sigmoid=use_sigmoid,
        sigmoid_scale=sigmoid_scale,
        fitness_function=fitness_function,
        **algo_params
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print()

    # Evaluate on test set
    print("Evaluating on test set:")
    metrics = model.evaluate(X_test, y_test)

    # Print metrics
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Save metrics
    params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'max_depth': 8,
        'training_time': training_time,
        'num_features': X.shape[1],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'use_sigmoid': use_sigmoid,
        'sigmoid_scale': sigmoid_scale,
        'fitness_function': fitness_function
    }

    # Create proper directory using our utility
    vis_dir = create_result_directory(
        root_dir=root_dir,
        dataset=dataset,
        algorithm=algo,
        result_type="visualizations"
    )

    # print("\nTree text representation:")
    # model.print_tree_representation()

    try:
        vis_base = os.path.join(vis_dir, f"classification")
        paths = visualize_classification_model(
            model,
            vis_base,
            dataset=dataset,
            algorithm=algo
        )
        if paths:
            print(f"Classification visualizations saved: {len(paths)} files")
    except Exception as e:
        print(f"Classification visualizer failed: {str(e)}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()