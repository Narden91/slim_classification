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
Test cases for validators and exceptions in classification module.
"""

import unittest
import torch

from slim_gsgp.classification.validators import (
    validate_binary_labels,
    validate_threshold,
    validate_tensor_shape,
    validate_matching_shapes,
    validate_scaling_factor,
)
from slim_gsgp.classification.exceptions import (
    ClassificationError,
    InvalidLabelError,
    AlgorithmNotFoundError,
    InvalidThresholdError,
    InvalidShapeError,
    FitnessRegistrationError,
)
from slim_gsgp.classification.config import ClassifierConfig, TrainingConfig


class TestValidators(unittest.TestCase):
    """Test cases for validation functions."""
    
    def test_validate_binary_labels_valid(self):
        """Test validate_binary_labels with valid binary labels."""
        valid_labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        
        # Should not raise
        validate_binary_labels(valid_labels)
    
    def test_validate_binary_labels_invalid(self):
        """Test validate_binary_labels with non-binary labels."""
        invalid_labels = torch.tensor([0.0, 1.0, 2.0, 0.0])
        
        with self.assertRaises(InvalidLabelError) as context:
            validate_binary_labels(invalid_labels)
        
        self.assertIn('binary', str(context.exception).lower())
    
    def test_validate_binary_labels_negative(self):
        """Test validate_binary_labels with negative values."""
        negative_labels = torch.tensor([0.0, 1.0, -1.0])
        
        with self.assertRaises(InvalidLabelError):
            validate_binary_labels(negative_labels)
    
    def test_validate_binary_labels_only_zeros(self):
        """Test validate_binary_labels with only zeros."""
        zeros = torch.zeros(10)
        
        # Should not raise - all zeros is valid binary
        validate_binary_labels(zeros)
    
    def test_validate_binary_labels_only_ones(self):
        """Test validate_binary_labels with only ones."""
        ones = torch.ones(10)
        
        # Should not raise - all ones is valid binary
        validate_binary_labels(ones)
    
    def test_validate_threshold_valid(self):
        """Test validate_threshold with valid values."""
        # Should not raise
        validate_threshold(0.5)
        validate_threshold(0.1)
        validate_threshold(0.9)
        validate_threshold(0.001)
        validate_threshold(0.999)
    
    def test_validate_threshold_out_of_range(self):
        """Test validate_threshold with out of range values."""
        with self.assertRaises(InvalidThresholdError):
            validate_threshold(0.0)
        
        with self.assertRaises(InvalidThresholdError):
            validate_threshold(1.0)
        
        with self.assertRaises(InvalidThresholdError):
            validate_threshold(-0.5)
        
        with self.assertRaises(InvalidThresholdError):
            validate_threshold(1.5)
    
    def test_validate_tensor_shape_valid(self):
        """Test validate_tensor_shape with valid tensors."""
        # 2D tensor
        tensor_2d = torch.randn(10, 5)
        validate_tensor_shape(tensor_2d)
        
        # 1D tensor
        tensor_1d = torch.randn(10)
        validate_tensor_shape(tensor_1d)
    
    def test_validate_tensor_shape_empty(self):
        """Test validate_tensor_shape with empty tensor."""
        empty_tensor = torch.tensor([])
        
        with self.assertRaises(InvalidShapeError):
            validate_tensor_shape(empty_tensor)
    
    def test_validate_tensor_shape_expected_features(self):
        """Test validate_tensor_shape with expected features."""
        tensor = torch.randn(10, 5)
        
        # Should not raise
        validate_tensor_shape(tensor, expected_features=5)
        
        # Should raise
        with self.assertRaises(InvalidShapeError):
            validate_tensor_shape(tensor, expected_features=3)
    
    def test_validate_matching_shapes_valid(self):
        """Test validate_matching_shapes with valid matching shapes."""
        X = torch.randn(10, 5)
        y = torch.randn(10)
        
        # Should not raise
        validate_matching_shapes(X, y, "X", "y")
    
    def test_validate_matching_shapes_invalid(self):
        """Test validate_matching_shapes with mismatched shapes."""
        X = torch.randn(10, 5)
        y = torch.randn(20)
        
        with self.assertRaises(InvalidShapeError) as context:
            validate_matching_shapes(X, y, "X", "y")
        
        self.assertIn('10', str(context.exception))
        self.assertIn('20', str(context.exception))
    
    def test_validate_scaling_factor_valid(self):
        """Test validate_scaling_factor with valid values."""
        # Should not raise
        validate_scaling_factor(1.0)
        validate_scaling_factor(0.5)
        validate_scaling_factor(2.0)
        validate_scaling_factor(0.001)
    
    def test_validate_scaling_factor_invalid(self):
        """Test validate_scaling_factor with invalid values."""
        with self.assertRaises(ValueError):
            validate_scaling_factor(0.0)
        
        with self.assertRaises(ValueError):
            validate_scaling_factor(-1.0)


class TestExceptions(unittest.TestCase):
    """Test cases for custom exceptions."""
    
    def test_classification_error_hierarchy(self):
        """Test that all exceptions inherit from ClassificationError."""
        self.assertTrue(issubclass(InvalidLabelError, ClassificationError))
        self.assertTrue(issubclass(AlgorithmNotFoundError, ClassificationError))
        self.assertTrue(issubclass(InvalidThresholdError, ClassificationError))
        self.assertTrue(issubclass(InvalidShapeError, ClassificationError))
        self.assertTrue(issubclass(FitnessRegistrationError, ClassificationError))
    
    def test_classification_error_is_exception(self):
        """Test that ClassificationError inherits from Exception."""
        self.assertTrue(issubclass(ClassificationError, Exception))
    
    def test_exception_messages(self):
        """Test that exceptions can carry messages."""
        msg = "Test error message"
        
        exc = ClassificationError(msg)
        self.assertEqual(str(exc), msg)
        
        exc = InvalidLabelError(msg)
        self.assertEqual(str(exc), msg)
        
        exc = AlgorithmNotFoundError(msg)
        self.assertEqual(str(exc), msg)
    
    def test_exception_raising(self):
        """Test that exceptions can be raised and caught."""
        with self.assertRaises(ClassificationError):
            raise InvalidLabelError("test")
        
        with self.assertRaises(ClassificationError):
            raise AlgorithmNotFoundError("test")
        
        with self.assertRaises(ClassificationError):
            raise InvalidThresholdError("test")


class TestConfig(unittest.TestCase):
    """Test cases for configuration dataclasses."""
    
    def test_classifier_config_defaults(self):
        """Test ClassifierConfig default values."""
        config = ClassifierConfig()
        
        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(config.use_sigmoid, True)
        self.assertEqual(config.sigmoid_scale, 1.0)
    
    def test_classifier_config_custom(self):
        """Test ClassifierConfig with custom values."""
        config = ClassifierConfig(
            threshold=0.7,
            use_sigmoid=False,
            sigmoid_scale=2.0
        )
        
        self.assertEqual(config.threshold, 0.7)
        self.assertEqual(config.use_sigmoid, False)
        self.assertEqual(config.sigmoid_scale, 2.0)
    
    def test_classifier_config_immutable(self):
        """Test ClassifierConfig is frozen."""
        config = ClassifierConfig()
        
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.threshold = 0.8
    
    def test_classifier_config_invalid_threshold(self):
        """Test ClassifierConfig validates threshold."""
        with self.assertRaises(InvalidThresholdError):
            ClassifierConfig(threshold=0.0)
        
        with self.assertRaises(InvalidThresholdError):
            ClassifierConfig(threshold=1.0)
    
    def test_classifier_config_invalid_scale(self):
        """Test ClassifierConfig validates scale."""
        with self.assertRaises(ValueError):
            ClassifierConfig(sigmoid_scale=0.0)
        
        with self.assertRaises(ValueError):
            ClassifierConfig(sigmoid_scale=-1.0)
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        self.assertEqual(config.algorithm, 'gp')
        self.assertEqual(config.fitness_function, 'binary_rmse')
        self.assertEqual(config.pop_size, 100)
        self.assertEqual(config.n_iter, 100)
    
    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            algorithm='slim',
            fitness_function='binary_mae',
            pop_size=50,
            n_iter=200,
            seed=42,
            verbose=0
        )
        
        self.assertEqual(config.algorithm, 'slim')
        self.assertEqual(config.pop_size, 50)
        self.assertEqual(config.n_iter, 200)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.verbose, 0)
    
    def test_training_config_immutable(self):
        """Test TrainingConfig is frozen."""
        config = TrainingConfig()
        
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.pop_size = 200
    
    def test_training_config_invalid_algorithm(self):
        """Test TrainingConfig validates algorithm."""
        with self.assertRaises(ValueError):
            TrainingConfig(algorithm='invalid_algo')
    
    def test_training_config_invalid_pop_size(self):
        """Test TrainingConfig validates pop_size."""
        with self.assertRaises(ValueError):
            TrainingConfig(pop_size=0)
        
        with self.assertRaises(ValueError):
            TrainingConfig(pop_size=-10)
    
    def test_training_config_invalid_n_iter(self):
        """Test TrainingConfig validates n_iter."""
        with self.assertRaises(ValueError):
            TrainingConfig(n_iter=0)


class TestMetrics(unittest.TestCase):
    """Test cases for metrics calculation."""
    
    def test_calculate_binary_metrics(self):
        """Test calculate_binary_metrics returns expected keys."""
        from slim_gsgp.classification import calculate_binary_metrics
        
        y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
        y_pred = torch.tensor([0.0, 1.0, 0.0, 0.0])
        
        metrics = calculate_binary_metrics(y_true, y_pred)
        
        # Check expected keys
        expected_keys = [
            'accuracy', 'precision', 'recall', 'f1',
            'specificity', 'true_positives', 'true_negatives',
            'false_positives', 'false_negatives'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_calculate_binary_metrics_perfect(self):
        """Test metrics for perfect predictions."""
        from slim_gsgp.classification import calculate_binary_metrics
        
        y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
        y_pred = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        metrics = calculate_binary_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
