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
Test script for binary classification functionality across all GP variants.
"""

import unittest
import torch
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler

# Import binary classification utilities
from slim_gsgp.classification import (
    apply_sigmoid,
    modified_sigmoid,
    binary_threshold_transform,
    binary_sign_transform,
    create_binary_fitness_function,
    register_classification_fitness_functions,
    train_binary_classifier,
    BinaryClassifier
)

# Import fitness functions
from slim_gsgp.evaluators.fitness_functions import rmse


class TestBinaryClassification(unittest.TestCase):
    """Test cases for binary classification utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create necessary directories for tests
        import os
        log_dir = os.path.join(os.getcwd(), "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Load and prepare data
        data = load_breast_cancer()
        X, y = data.data, data.target

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # Create small dataset for quick testing
        X_small = X[:100]
        y_small = y[:100]

        # Split data
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X_small, y_small, test_size=0.3, random_state=42
        )

        cls.X_train = X_train
        cls.y_train = y_train
        cls.X_test = X_test
        cls.y_test = y_test

        # Register fitness functions
        register_classification_fitness_functions()

    def test_apply_sigmoid(self):
        """Test apply_sigmoid function."""
        # Test with various inputs
        test_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = apply_sigmoid(test_tensor, scaling_factor=1.0)

        # Check output range
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))

        # Check specific values
        self.assertAlmostEqual(result[2].item(), 0.5, places=6)  # sigmoid(0) = 0.5
        self.assertTrue(result[3].item() > 0.7)  # sigmoid(1) should be > 0.7
        self.assertTrue(result[1].item() < 0.3)  # sigmoid(-1) should be < 0.3
    
    def test_modified_sigmoid(self):
        """Test modified_sigmoid function (deprecated but kept for compatibility)."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sigmoid = modified_sigmoid(scaling_factor=1.0)
            
            # Check that deprecation warning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[-1].message).lower())

        # Test with various inputs
        test_tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = sigmoid(test_tensor)

        # Check output range
        self.assertTrue(torch.all(result >= 0) and torch.all(result <= 1))

        # Check specific values
        self.assertAlmostEqual(result[2].item(), 0.5, places=6)  # sigmoid(0) = 0.5
        self.assertTrue(result[3].item() > 0.7)  # sigmoid(1) should be > 0.7
        self.assertTrue(result[1].item() < 0.3)  # sigmoid(-1) should be < 0.3

    def test_binary_transforms(self):
        """Test binary transformation functions."""
        # Test sign transform
        test_tensor = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
        sign_result = binary_sign_transform(test_tensor)
        expected_sign = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(sign_result == expected_sign))

        # Test threshold transform
        test_probs = torch.tensor([0.1, 0.4, 0.5, 0.6, 0.9])
        threshold_result = binary_threshold_transform(test_probs, threshold=0.5)
        expected_threshold = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(threshold_result == expected_threshold))

    def test_create_binary_fitness_function(self):
        """Test creation of binary fitness function."""
        # Create binary RMSE
        binary_rmse = create_binary_fitness_function(rmse)

        # Test with some data
        y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
        y_pred = torch.tensor([-2.0, 0.5, -1.0, 3.0])

        # Calculate fitness
        fitness = binary_rmse(y_true, y_pred)

        # Check that it returns a value
        self.assertIsInstance(fitness.item(), float)

    def _test_algorithm(self, algorithm):
        """Helper to test a specific algorithm."""
        print(f"\nTesting {algorithm.upper()} binary classification...")

        # Set minimal training parameters
        params = {
            'pop_size': 10,
            'n_iter': 2,
            'seed': 42
        }

        # Add algorithm-specific parameters
        if algorithm == 'gsgp':
            params['reconstruct'] = True
            params['ms_lower'] = 0
            params['ms_upper'] = 1
        elif algorithm == 'slim':
            params['slim_version'] = 'SLIM+ABS'
            params['p_inflate'] = 0.5
            params['ms_lower'] = 0
            params['ms_upper'] = 1

        # Train classifier
        try:
            classifier = train_binary_classifier(
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_test,
                y_val=self.y_test,
                algorithm=algorithm,
                fitness_function='binary_rmse',
                **params
            )

            # Verify prediction
            predictions = classifier.predict(self.X_test)
            accuracy = (predictions == self.y_test).float().mean().item()

            print(f"  Test accuracy: {accuracy:.4f}")
            self.assertTrue(accuracy > 0.0)

            return True
        except Exception as e:
            print(f"  Error testing {algorithm}: {str(e)}")
            return False

    def test_gp(self):
        """Test GP binary classification."""
        self.assertTrue(self._test_algorithm('gp'))

    def test_gsgp(self):
        """Test GSGP binary classification."""
        self.assertTrue(self._test_algorithm('gsgp'))

    def test_slim(self):
        """Test SLIM binary classification."""
        self.assertTrue(self._test_algorithm('slim'))

    def test_classifier_methods(self):
        """Test BinaryClassifier methods."""

        # Create a mock model with a predict method
        class MockModel:
            def predict(self, X):
                return torch.randn(len(X))

            def print_tree_representation(self):
                print("Mock tree representation")

        # Create classifier
        classifier = BinaryClassifier(
            model=MockModel(),
            threshold=0.5,
            use_sigmoid=True
        )

        # Test predict_proba
        probs = classifier.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), 2))
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))

        # Test predict
        preds = classifier.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.X_test),))
        self.assertTrue(torch.all((preds == 0) | (preds == 1)))

        # Test evaluate
        metrics = classifier.evaluate(self.X_test, self.y_test)
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(metric_name, metrics)


if __name__ == '__main__':
    unittest.main()