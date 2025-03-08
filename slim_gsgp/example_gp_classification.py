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
from slim_gsgp.main_gp import gp  # Import GP from SLIM
from slim_gsgp.datasets.data_loader import load_breast_cancer  # Load dataset
from slim_gsgp.evaluators.fitness_functions import binary_cross_entropy  # Import BCE fitness function
from slim_gsgp.utils.utils import train_test_split  # Train-test split function
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)


def main():
    # Load dataset
    # X, y = load_ppb(X_y=True)
    X, y = load_breast_cancer(X_y=True)

    # Convert to PyTorch tensors if not already
    X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
    y = torch.tensor(y, dtype=torch.long) if not isinstance(y, torch.Tensor) else y

    # Detect number of classes
    n_classes = len(torch.unique(y))
    print(f"Detected {n_classes} classes -> {'Binary' if n_classes == 2 else 'Multiclass'} classification")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.2)

    # Further split test set into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.3)

    # Choose fitness function based on classification type
    fitness_function = "binary_cross_entropy" if n_classes == 2 else "cross_entropy"

    # Train the model using genetic programming (GP)
    final_tree = gp(X_train=X_train, y_train=y_train,
                    X_test=X_val, y_test=y_val,
                    dataset_name='breast_cancer',
                    pop_size=100, n_iter=10,
                    max_depth=None, fitness_function=fitness_function)

    # Display the best evolved tree
    final_tree.print_tree_representation()

    # Get raw predictions on the test set
    raw_predictions = final_tree.predict(X_test)

    # Ensure predictions are in tensor form
    raw_predictions_tensor = (raw_predictions if isinstance(raw_predictions, torch.Tensor)
                              else torch.tensor(raw_predictions, dtype=torch.float32))

    # Process predictions and calculate metrics based on classification type
    if n_classes == 2:
        # Binary Classification
        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(raw_predictions_tensor)
        predictions = (probabilities > 0.5).long()  # Binary predictions (0 or 1)

        # Convert to NumPy for sklearn compatibility
        y_true = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        y_pred = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        y_prob = probabilities.numpy() if isinstance(probabilities, torch.Tensor) else probabilities

        # Calculate binary metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Display results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.4f}")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print(f"ROC AUC: {roc_auc * 100:.2f}%")
        print("Confusion Matrix:")
        print(conf_matrix)

    else:
        # Multiclass Classification
        softmax = torch.nn.Softmax(dim=1)  # Apply softmax across class dimension
        probabilities = softmax(raw_predictions_tensor)
        predictions = torch.argmax(probabilities, dim=1)  # Predicted class indices

        # Convert to NumPy for sklearn compatibility
        y_true = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        y_pred = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions

        # Calculate multiclass metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Display results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision (Macro): {precision_macro * 100:.2f}%")
        print(f"Recall (Macro): {recall_macro * 100:.2f}%")
        print(f"F1 Score (Macro): {f1_macro * 100:.2f}%")
        print(f"Precision (Micro): {precision_micro * 100:.2f}%")
        print(f"Recall (Micro): {recall_micro * 100:.2f}%")
        print(f"F1 Score (Micro): {f1_micro * 100:.2f}%")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
