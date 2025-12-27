#!/usr/bin/env python3
"""
Phase 1 Enhancements Test - Tests new features added in Phase 1.
"""
import torch
import sys
import os

# Ensure local project imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from slim_gsgp.classification import (
    train_binary_classifier,
    BinaryClassifier,
    ClassifierConfig,
    BinaryMetrics,
    calculate_binary_metrics,
)
from slim_gsgp.classification.strategies import SigmoidStrategy, SignBasedStrategy


def test_binary_metrics_dataclass():
    """Test BinaryMetrics dataclass features."""
    print("=" * 70)
    print("TEST 1: BinaryMetrics Dataclass")
    print("=" * 70)
    
    y_true = torch.tensor([0., 1., 1., 0., 1., 0., 1., 0.])
    y_pred = torch.tensor([0., 1., 0., 0., 1., 1., 1., 0.])
    
    metrics = calculate_binary_metrics(y_true, y_pred)
    
    # Test attribute access
    print(f"✓ Attribute access: metrics.accuracy = {metrics.accuracy:.4f}")
    assert hasattr(metrics, 'accuracy')
    
    # Test dict-like access (backward compatibility)
    print(f"✓ Dict-like access: metrics['accuracy'] = {metrics['accuracy']:.4f}")
    assert metrics['accuracy'] == metrics.accuracy
    
    # Test 'in' operator
    print(f"✓ Contains operator: 'accuracy' in metrics = {'accuracy' in metrics}")
    assert 'accuracy' in metrics
    assert 'nonexistent' not in metrics
    
    # Test get method
    print(f"✓ Get method: metrics.get('f1') = {metrics.get('f1'):.4f}")
    print(f"✓ Get with default: metrics.get('missing', 0.0) = {metrics.get('missing', 0.0)}")
    assert metrics.get('f1') == metrics.f1
    assert metrics.get('nonexistent', -1.0) == -1.0
    
    # Test iteration
    keys = list(metrics.keys())
    print(f"✓ Keys iteration: {len(keys)} keys")
    assert 'accuracy' in keys
    
    # Test to_dict
    metrics_dict = metrics.to_dict()
    print(f"✓ to_dict: type={type(metrics_dict).__name__}, keys={len(metrics_dict)}")
    assert isinstance(metrics_dict, dict)
    assert metrics_dict['accuracy'] == metrics.accuracy
    
    # Test repr
    repr_str = repr(metrics)
    print(f"✓ Repr: {repr_str[:60]}...")
    assert 'BinaryMetrics' in repr_str
    
    return True


def test_classifier_repr_eq():
    """Test BinaryClassifier __repr__ and __eq__ methods."""
    print("\n" + "=" * 70)
    print("TEST 2: BinaryClassifier __repr__ and __eq__")
    print("=" * 70)
    
    # Create synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(50, 3)
    y_train = (X_train[:, 0] > 0).float()
    
    # Train classifier
    classifier1 = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=5,
        n_iter=2,
        threshold=0.5,
        sigmoid_scale=1.0,
        verbose=0
    )
    
    # Test repr
    repr_str = repr(classifier1)
    print(f"✓ Repr: {repr_str}")
    assert 'BinaryClassifier' in repr_str
    assert 'threshold=0.5' in repr_str
    
    # Train another classifier with same config
    classifier2 = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=5,
        n_iter=2,
        threshold=0.5,
        sigmoid_scale=1.0,
        verbose=0
    )
    
    # Test equality (configuration-based, not model-based)
    print(f"✓ Equality: classifier1 == classifier2 = {classifier1 == classifier2}")
    assert classifier1 == classifier2
    
    # Test inequality with different config
    classifier3 = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=5,
        n_iter=2,
        threshold=0.6,  # Different
        sigmoid_scale=1.0,
        verbose=0
    )
    
    print(f"✓ Inequality: classifier1 == classifier3 = {classifier1 == classifier3}")
    assert classifier1 != classifier3
    
    return True


def test_classifier_from_config():
    """Test BinaryClassifier.from_config classmethod."""
    print("\n" + "=" * 70)
    print("TEST 3: BinaryClassifier.from_config")
    print("=" * 70)
    
    # Create config
    config = ClassifierConfig(threshold=0.7, sigmoid_scale=2.0, use_sigmoid=True)
    print(f"✓ Config: threshold={config.threshold}, scale={config.sigmoid_scale}")
    
    # Train model
    torch.manual_seed(42)
    X_train = torch.randn(50, 3)
    y_train = (X_train[:, 0] > 0).float()
    
    classifier = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=5,
        n_iter=2,
        verbose=0
    )
    
    # Create new classifier from config
    new_classifier = BinaryClassifier.from_config(classifier.model, config)
    print(f"✓ from_config: threshold={new_classifier.threshold}, scale={new_classifier.sigmoid_scale}")
    
    assert new_classifier.threshold == config.threshold
    assert new_classifier.sigmoid_scale == config.sigmoid_scale
    assert new_classifier.use_sigmoid == config.use_sigmoid
    
    return True


def test_classifier_with_strategy():
    """Test BinaryClassifier.with_strategy method."""
    print("\n" + "=" * 70)
    print("TEST 4: BinaryClassifier.with_strategy")
    print("=" * 70)
    
    # Train classifier
    torch.manual_seed(42)
    X_train = torch.randn(50, 3)
    y_train = (X_train[:, 0] > 0).float()
    X_test = torch.randn(20, 3)
    
    classifier = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=5,
        n_iter=2,
        verbose=0
    )
    
    original_strategy = classifier.strategy.name
    print(f"✓ Original strategy: {original_strategy}")
    
    # Create new classifier with different strategy
    new_strategy = SignBasedStrategy()
    new_classifier = classifier.with_strategy(new_strategy)
    
    print(f"✓ New strategy: {new_classifier.strategy.name}")
    assert new_classifier.strategy.name == 'sign-based'
    assert classifier.strategy.name == original_strategy  # Original unchanged
    
    # Test predictions differ
    pred_original = classifier.predict(X_test)
    pred_new = new_classifier.predict(X_test)
    
    print(f"✓ Predictions differ: {not torch.equal(pred_original, pred_new)}")
    # They may be the same by chance, but strategies are different
    assert classifier.strategy.name != new_classifier.strategy.name
    
    return True


def test_validators_return_values():
    """Test validators return validated values."""
    print("\n" + "=" * 70)
    print("TEST 5: Validators Return Values")
    print("=" * 70)
    
    from slim_gsgp.classification.validators import (
        validate_threshold,
        validate_scaling_factor,
        validate_binary_labels,
        validate_tensor_shape,
        validate_matching_shapes,
    )
    
    # Test validate_threshold returns value
    result = validate_threshold(0.5)
    print(f"✓ validate_threshold(0.5) returns: {result}")
    assert result == 0.5
    assert isinstance(result, float)
    
    # Test validate_scaling_factor returns value
    result = validate_scaling_factor(2.0)
    print(f"✓ validate_scaling_factor(2.0) returns: {result}")
    assert result == 2.0
    assert isinstance(result, float)
    
    # Test validate_binary_labels returns tensor
    labels = torch.tensor([0., 1., 1., 0.])
    result = validate_binary_labels(labels)
    print(f"✓ validate_binary_labels returns tensor: {result.shape}")
    assert torch.equal(result, labels)
    
    # Test validate_tensor_shape returns tensor
    X = torch.randn(10, 5)
    result = validate_tensor_shape(X)
    print(f"✓ validate_tensor_shape returns tensor: {result.shape}")
    assert torch.equal(result, X)
    
    # Test validate_matching_shapes returns tuple
    X = torch.randn(10, 5)
    y = torch.randn(10)
    X_val, y_val = validate_matching_shapes(X, y)
    print(f"✓ validate_matching_shapes returns tuple: ({X_val.shape}, {y_val.shape})")
    assert torch.equal(X_val, X)
    assert torch.equal(y_val, y)
    
    return True


def test_factory_thread_safety():
    """Test factory thread-safety enhancements."""
    print("\n" + "=" * 70)
    print("TEST 6: Factory Thread-Safety")
    print("=" * 70)
    
    from slim_gsgp.classification.factories import AlgorithmFactory, get_default_factory
    
    # Get factory
    factory = get_default_factory()
    print(f"✓ Factory obtained: {type(factory).__name__}")
    
    # Test reset_factory method exists
    assert hasattr(AlgorithmFactory, 'reset_factory')
    print("✓ reset_factory method exists")
    
    # Test _lock attribute exists
    assert hasattr(AlgorithmFactory, '_lock')
    print("✓ Thread lock exists")
    
    # Verify algorithms still available after implementation
    algorithms = factory.available_algorithms()
    print(f"✓ Available algorithms: {algorithms}")
    assert 'gp' in algorithms
    assert 'gsgp' in algorithms
    assert 'slim' in algorithms
    
    return True


def main():
    """Run all Phase 1 enhancement tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 ENHANCEMENT TESTS - NEW FEATURES")
    print("=" * 70 + "\n")
    
    tests = [
        ("BinaryMetrics Dataclass", test_binary_metrics_dataclass),
        ("Classifier __repr__ and __eq__", test_classifier_repr_eq),
        ("Classifier from_config", test_classifier_from_config),
        ("Classifier with_strategy", test_classifier_with_strategy),
        ("Validators Return Values", test_validators_return_values),
        ("Factory Thread-Safety", test_factory_thread_safety),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
