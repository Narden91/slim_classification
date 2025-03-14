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

This script demonstrates how to use the specialized classification modules
for both binary and multiclass classification problems.
"""
import time
import os
import torch

from slim_gsgp.utils.utils import train_test_split, create_result_directory
from slim_gsgp.datasets.data_loader import load_classification_dataset
from slim_gsgp.classifiers.binary_classifiers import train_binary_classifier
from slim_gsgp.classifiers.multiclass_classifiers import train_multiclass_classifier
from slim_gsgp.tree_visualizer import visualize_classification_model
from slim_gsgp.classifiers.classification_utils import evaluate_classification_model
from slim_gsgp.utils.metrics_utils import save_metrics


def main():
    """
    Main function to run the classification example.
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # Define parameters directly
    dataset = 'breast_cancer'  # Options: 'breast_cancer', 'iris', 'digits', 'wine'
    algo = 'gsgp'  # Options: 'gp', 'gsgp', 'slim'
    strategy = 'ovr'  # Options: 'ovr', 'ovo'
    balance = True  # True or False
    pop_size = 50  # Integer
    n_iter = 2  # Integer
    seed = 42  # Integer
    parallel = True  # True or False

    # Set random seed
    torch.manual_seed(seed)

    print(f"Running classification example with:")
    print(f"  Dataset: {dataset}")
    print(f"  Algorithm: {algo}")
    print(f"  Strategy: {strategy}")
    print(f"  Balance data: {balance}")
    print(f"  Population size: {pop_size}")
    print(f"  Iterations: {n_iter}")
    print(f"  Parallel: {parallel}")
    print()

    # Load the dataset
    print(f"Loading dataset: {dataset}")
    X, y, n_classes, class_labels = load_classification_dataset(dataset)
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    print()

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, p_test=0.3, seed=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, p_test=0.5, seed=seed)

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Common parameters for GP
    gp_params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'max_depth': 8,
        'seed': seed,
        'dataset_name': dataset,
        'fitness_function': 'binary_cross_entropy'
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
            algo_type=algo,
            balance_data=balance,
            **gp_params
        )
        # Binary classification has no strategy
        strategy = None
    else:
        # Multiclass classification
        print(f"Training multiclass classifier using {strategy} strategy...")
        model = train_multiclass_classifier(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy=strategy,
            algo_type=algo,
            balance_data=balance,
            n_classes=n_classes,
            class_labels=class_labels,
            parallel=parallel,
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

    # Save metrics
    params = {
        'pop_size': pop_size,
        'n_iter': n_iter,
        'max_depth': 8,
        'parallel': parallel,
        'training_time': training_time,
        'num_features': X.shape[1],
        'train_size': len(X_train),
        'test_size': len(X_test)
    }

    metrics_path = save_metrics(
        metrics=metrics,
        dataset=dataset,
        algorithm=algo,
        strategy=strategy,
        balance=balance,
        params=params,
        root_dir=root_dir
    )

    print(f"\nMetrics saved to: {metrics_path}")

    # Visualize the model trees
    print("\nVisualizing model trees...")
    base_filename = f"{dataset}_tree"

    try:
        # Pass dataset, algorithm and strategy to organize directories
        visualization_paths = visualize_classification_model(
            model,
            base_filename,
            dataset=dataset,
            algorithm=algo,
            strategy=strategy
        )

        if visualization_paths:
            print(f"Successfully saved {len(visualization_paths)} tree visualizations")
            print(
                f"Visualizations stored in results/{dataset}/{algo}{strategy if strategy else ''}/visualizations/ directory")
        else:
            print("No visualizations were created")
    except Exception as e:
        print(f"Enhanced visualization failed: {str(e)}")
        print("Falling back to model's built-in visualization...")

        # Fall back to the model's built-in visualization but use our directory structure
        try:
            # Get project root directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

            # Create proper directory using our utility
            from slim_gsgp.utils.utils import create_result_directory
            vis_dir = create_result_directory(
                root_dir=root_dir,
                dataset=dataset,
                algorithm=algo,
                result_type="visualizations",
                strategy=strategy
            )

            # Use full path with proper directory
            vis_filename = os.path.join(vis_dir, base_filename)

            # Call model's visualization with our path
            visualization_path = model.visualize_tree(filename=vis_filename, dataset=dataset, algorithm=algo)
            print(f"Built-in visualization saved to: {visualization_path}")
        except Exception as e:
            print(f"Built-in visualization also failed: {str(e)}")
            print("Unable to visualize the model trees")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
