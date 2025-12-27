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
Test cases for Factory and Strategy patterns in classification module.
"""

import unittest
import torch

from slim_gsgp.classification import (
    # Factory Pattern
    AlgorithmFactory,
    get_default_factory,
    create_algorithm,
    register_algorithm,
    # Strategy Pattern
    PredictionStrategy,
    SigmoidStrategy,
    SignBasedStrategy,
    SoftmaxStrategy,
    PredictionContext,
    get_strategy,
    register_strategy,
    # Exceptions
    AlgorithmNotFoundError,
    InvalidThresholdError,
    # Other
    BinaryClassifier,
    register_classification_fitness_functions,
)


class TestAlgorithmFactory(unittest.TestCase):
    """Test cases for AlgorithmFactory."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        import os
        log_dir = os.path.join(os.getcwd(), "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        cls.X_train = torch.randn(50, 5)
        cls.y_train = torch.randint(0, 2, (50,)).float()
        register_classification_fitness_functions()
    
    def test_factory_initialization(self):
        """Test factory initializes with built-in algorithms."""
        factory = AlgorithmFactory()
        algorithms = factory.available_algorithms()
        
        self.assertIn('gp', algorithms)
        self.assertIn('gsgp', algorithms)
        self.assertIn('slim', algorithms)
    
    def test_get_default_factory(self):
        """Test singleton-like behavior of default factory."""
        factory1 = get_default_factory()
        factory2 = get_default_factory()
        
        # Should return the same instance
        self.assertIs(factory1, factory2)
    
    def test_factory_create_gp(self):
        """Test creating GP model through factory."""
        factory = AlgorithmFactory()
        
        model = factory.create(
            'gp',
            self.X_train,
            self.y_train,
            pop_size=5,
            n_iter=1,
            seed=42,
            fitness_function='binary_rmse'
        )
        
        # Model should be able to predict
        predictions = model.predict(self.X_train)
        self.assertEqual(predictions.shape, (50,))
    
    def test_factory_unknown_algorithm(self):
        """Test factory raises error for unknown algorithm."""
        factory = AlgorithmFactory()
        
        with self.assertRaises(AlgorithmNotFoundError) as context:
            factory.create('unknown', self.X_train, self.y_train)
        
        self.assertIn('unknown', str(context.exception).lower())
    
    def test_register_custom_algorithm(self):
        """Test registering a custom algorithm."""
        factory = AlgorithmFactory()
        
        # Create a mock algorithm
        def mock_algorithm(X_train, y_train, X_test=None, y_test=None, **kwargs):
            class MockModel:
                def predict(self, X):
                    return torch.zeros(X.shape[0])
            return MockModel()
        
        # Register it
        factory.register('mock_algo', mock_algorithm, instance_only=True)
        
        # Verify it's available
        self.assertIn('mock_algo', factory.available_algorithms())
        
        # Create a model
        model = factory.create('mock_algo', self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        self.assertTrue(torch.all(predictions == 0))
    
    def test_create_algorithm_convenience(self):
        """Test create_algorithm convenience function."""
        model = create_algorithm(
            'gp',
            self.X_train,
            self.y_train,
            pop_size=5,
            n_iter=1,
            seed=42,
            fitness_function='binary_rmse'
        )
        
        predictions = model.predict(self.X_train)
        self.assertEqual(predictions.shape, (50,))
    
    def test_factory_default_kwargs(self):
        """Test that default kwargs are applied."""
        factory = AlgorithmFactory()
        
        # GSGP should have reconstruct=True by default
        defaults = factory.get_default_kwargs('gsgp')
        self.assertEqual(defaults.get('reconstruct'), True)


class TestPredictionStrategies(unittest.TestCase):
    """Test cases for prediction strategies."""
    
    def setUp(self):
        """Set up test data."""
        self.raw_outputs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    def test_sigmoid_strategy_predict(self):
        """Test SigmoidStrategy prediction."""
        strategy = SigmoidStrategy(scale=1.0, threshold=0.5)
        
        predictions = strategy.predict(self.raw_outputs)
        
        # Should return binary values
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))
        
        # Negative should be 0, positive should be 1 (approximately)
        self.assertEqual(predictions[0].item(), 0.0)  # -2.0 -> 0
        self.assertEqual(predictions[4].item(), 1.0)  # 2.0 -> 1
    
    def test_sigmoid_strategy_proba(self):
        """Test SigmoidStrategy probability output."""
        strategy = SigmoidStrategy(scale=1.0, threshold=0.5)
        
        probs = strategy.predict_proba(self.raw_outputs)
        
        # Should have shape (n_samples, 2)
        self.assertEqual(probs.shape, (5, 2))
        
        # Probabilities should sum to 1
        sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones(5)))
        
        # P(class=1) for 0.0 input should be ~0.5
        self.assertAlmostEqual(probs[2, 1].item(), 0.5, places=3)
    
    def test_sigmoid_strategy_scaled(self):
        """Test SigmoidStrategy with different scales."""
        strategy_low = SigmoidStrategy(scale=0.5, threshold=0.5)
        strategy_high = SigmoidStrategy(scale=2.0, threshold=0.5)
        
        test_input = torch.tensor([1.0])
        
        probs_low = strategy_low.predict_proba(test_input)
        probs_high = strategy_high.predict_proba(test_input)
        
        # Higher scale should give more extreme probabilities
        self.assertTrue(probs_high[0, 1] > probs_low[0, 1])
    
    def test_sigmoid_strategy_validation(self):
        """Test SigmoidStrategy validates parameters."""
        # Invalid threshold
        with self.assertRaises(Exception):
            SigmoidStrategy(scale=1.0, threshold=1.5)
        
        # Invalid scale
        with self.assertRaises(Exception):
            SigmoidStrategy(scale=-1.0, threshold=0.5)
    
    def test_sign_based_strategy(self):
        """Test SignBasedStrategy."""
        strategy = SignBasedStrategy()
        
        predictions = strategy.predict(self.raw_outputs)
        
        # Negative -> 0, non-negative -> 1
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertTrue(torch.all(predictions == expected))
    
    def test_sign_based_strategy_proba(self):
        """Test SignBasedStrategy probability output."""
        strategy = SignBasedStrategy()
        
        probs = strategy.predict_proba(self.raw_outputs)
        
        # Should be hard probabilities (0 or 1)
        self.assertTrue(torch.all((probs == 0) | (probs == 1)))
    
    def test_softmax_strategy(self):
        """Test SoftmaxStrategy."""
        strategy = SoftmaxStrategy(temperature=1.0)
        
        predictions = strategy.predict(self.raw_outputs)
        
        self.assertTrue(torch.all((predictions == 0) | (predictions == 1)))
    
    def test_softmax_strategy_temperature(self):
        """Test SoftmaxStrategy with different temperatures."""
        strategy_cold = SoftmaxStrategy(temperature=0.5)
        strategy_hot = SoftmaxStrategy(temperature=2.0)
        
        test_input = torch.tensor([0.5])
        
        probs_cold = strategy_cold.predict_proba(test_input)
        probs_hot = strategy_hot.predict_proba(test_input)
        
        # Lower temperature = sharper distribution
        # Higher temperature = smoother distribution
        self.assertTrue(probs_cold[0, 1] > probs_hot[0, 1])
    
    def test_softmax_strategy_validation(self):
        """Test SoftmaxStrategy validates temperature."""
        with self.assertRaises(ValueError):
            SoftmaxStrategy(temperature=0)
        
        with self.assertRaises(ValueError):
            SoftmaxStrategy(temperature=-1)
    
    def test_get_strategy(self):
        """Test get_strategy factory function."""
        sigmoid = get_strategy('sigmoid', scale=1.0, threshold=0.5)
        self.assertIsInstance(sigmoid, SigmoidStrategy)
        
        sign = get_strategy('sign')
        self.assertIsInstance(sign, SignBasedStrategy)
        
        softmax = get_strategy('softmax', temperature=1.0)
        self.assertIsInstance(softmax, SoftmaxStrategy)
    
    def test_get_strategy_unknown(self):
        """Test get_strategy raises error for unknown strategy."""
        with self.assertRaises(ValueError) as context:
            get_strategy('unknown')
        
        self.assertIn('unknown', str(context.exception).lower())
    
    def test_prediction_context(self):
        """Test PredictionContext."""
        strategy = SigmoidStrategy(scale=1.0, threshold=0.5)
        context = PredictionContext(strategy)
        
        predictions = context.predict(self.raw_outputs)
        self.assertEqual(predictions.shape, (5,))
        
        probs = context.predict_proba(self.raw_outputs)
        self.assertEqual(probs.shape, (5, 2))
    
    def test_prediction_context_strategy_switching(self):
        """Test PredictionContext can switch strategies."""
        context = PredictionContext(SigmoidStrategy(scale=1.0, threshold=0.5))
        
        # Get predictions with sigmoid
        preds1 = context.predict(self.raw_outputs)
        
        # Switch to sign-based
        context.strategy = SignBasedStrategy()
        preds2 = context.predict(self.raw_outputs)
        
        # Results should be different
        self.assertFalse(torch.all(preds1 == preds2))
    
    def test_strategy_name_property(self):
        """Test strategy name property."""
        self.assertIn('sigmoid', SigmoidStrategy(1.0, 0.5).name.lower())
        self.assertIn('sign', SignBasedStrategy().name.lower())
        self.assertIn('softmax', SoftmaxStrategy(1.0).name.lower())


class TestBinaryClassifierWithStrategy(unittest.TestCase):
    """Test BinaryClassifier with Strategy Pattern integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        import os
        log_dir = os.path.join(os.getcwd(), "log")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        cls.X_train = torch.randn(50, 5)
        cls.y_train = torch.randint(0, 2, (50,)).float()
        register_classification_fitness_functions()
    
    def test_classifier_with_strategy(self):
        """Test BinaryClassifier accepts strategy parameter."""
        # Create a mock model
        class MockModel:
            def predict(self, X):
                return torch.randn(X.shape[0])
        
        strategy = SigmoidStrategy(scale=2.0, threshold=0.6)
        classifier = BinaryClassifier(MockModel(), strategy=strategy)
        
        self.assertEqual(classifier.threshold, 0.6)
        self.assertEqual(classifier.sigmoid_scale, 2.0)
        self.assertTrue(classifier.use_sigmoid)
    
    def test_classifier_strategy_override(self):
        """Test strategy overrides other parameters."""
        class MockModel:
            def predict(self, X):
                return torch.randn(X.shape[0])
        
        # Strategy should override threshold and scale
        strategy = SigmoidStrategy(scale=3.0, threshold=0.7)
        classifier = BinaryClassifier(
            MockModel(),
            threshold=0.5,  # Should be overridden
            sigmoid_scale=1.0,  # Should be overridden
            strategy=strategy
        )
        
        self.assertEqual(classifier.threshold, 0.7)
        self.assertEqual(classifier.sigmoid_scale, 3.0)
    
    def test_classifier_strategy_property(self):
        """Test classifier strategy property."""
        class MockModel:
            def predict(self, X):
                return torch.randn(X.shape[0])
        
        classifier = BinaryClassifier(MockModel())
        
        # Should have a strategy
        self.assertIsInstance(classifier.strategy, PredictionStrategy)
        
        # Can change strategy
        new_strategy = SignBasedStrategy()
        classifier.strategy = new_strategy
        self.assertIsInstance(classifier.strategy, SignBasedStrategy)
    
    def test_classifier_predictions_use_strategy(self):
        """Test that classifier predictions use the strategy."""
        class MockModel:
            def predict(self, X):
                return torch.tensor([1.0, -1.0, 0.0])
        
        # With sigmoid strategy
        classifier_sig = BinaryClassifier(
            MockModel(),
            strategy=SigmoidStrategy(scale=1.0, threshold=0.5)
        )
        
        # With sign strategy
        classifier_sign = BinaryClassifier(
            MockModel(),
            strategy=SignBasedStrategy()
        )
        
        X_test = torch.randn(3, 5)
        
        preds_sig = classifier_sig.predict(X_test)
        preds_sign = classifier_sign.predict(X_test)
        
        # Sign-based: [1, 0, 1]
        expected_sign = torch.tensor([1.0, 0.0, 1.0])
        self.assertTrue(torch.all(preds_sign == expected_sign))


class TestRegisterStrategy(unittest.TestCase):
    """Test custom strategy registration."""
    
    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        from slim_gsgp.classification.strategies import _STRATEGY_REGISTRY
        
        class CustomStrategy(PredictionStrategy):
            def predict(self, raw_outputs):
                return (raw_outputs > 1.0).float()
            
            def predict_proba(self, raw_outputs):
                probs = (raw_outputs > 1.0).float()
                return torch.stack([1 - probs, probs], dim=1)
            
            @property
            def name(self):
                return "custom"
        
        register_strategy('custom', CustomStrategy)
        
        # Should be retrievable
        self.assertIn('custom', _STRATEGY_REGISTRY)
        
        strategy = get_strategy('custom')
        self.assertIsInstance(strategy, CustomStrategy)
        
        # Clean up
        del _STRATEGY_REGISTRY['custom']


if __name__ == '__main__':
    unittest.main()
