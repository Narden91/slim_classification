#!/usr/bin/env python3
"""
Phase 1 Baseline Test - Tests current classification behavior before refactoring.
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
    calculate_binary_metrics,
    register_classification_fitness_functions,
    apply_sigmoid,
)
from slim_gsgp.classification.factories import get_default_factory, AlgorithmFactory
from slim_gsgp.classification.validators import (
    validate_binary_labels,
    validate_threshold,
    validate_scaling_factor,
)


def test_basic_classification():
    """Test basic classification workflow with SLIM."""
    print("=" * 70)
    print("TEST 1: Basic SLIM Classification")
    print("=" * 70)
    
    # Register fitness functions
    register_classification_fitness_functions()
    
    # Create synthetic binary dataset
    torch.manual_seed(42)
    X_train = torch.randn(100, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).float()
    X_test = torch.randn(30, 5)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).float()
    
    print(f"Training set: {X_train.shape}, Labels: {y_train.unique().tolist()}")
    print(f"Test set: {X_test.shape}, Labels: {y_test.unique().tolist()}")
    
    # Train classifier
    classifier = train_binary_classifier(
        X_train, y_train, X_test, y_test,
        algorithm='slim',
        slim_version='SLIM+ABS',
        pop_size=10,
        n_iter=5,
        fitness_function='binary_rmse',
        verbose=0
    )
    
    print(f"✓ Classifier trained: {type(classifier).__name__}")
    print(f"  - threshold: {classifier.threshold}")
    print(f"  - use_sigmoid: {classifier.use_sigmoid}")
    print(f"  - sigmoid_scale: {classifier.sigmoid_scale}")
    
    # Test predictions
    predictions = classifier.predict(X_test)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"  - Unique values: {predictions.unique().tolist()}")
    
    # Test probabilities
    probas = classifier.predict_proba(X_test)
    print(f"✓ Probabilities shape: {probas.shape}")
    print(f"  - Column 0 range: [{probas[:, 0].min():.4f}, {probas[:, 0].max():.4f}]")
    print(f"  - Column 1 range: [{probas[:, 1].min():.4f}, {probas[:, 1].max():.4f}]")
    
    # Test evaluation
    metrics = classifier.evaluate(X_test, y_test)
    print(f"✓ Metrics calculated: {type(metrics).__name__}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1: {metrics['f1']:.4f}")
    
    return True


def test_classifier_config():
    """Test ClassifierConfig usage."""
    print("\n" + "=" * 70)
    print("TEST 2: ClassifierConfig Usage")
    print("=" * 70)
    
    # Create config
    config = ClassifierConfig(threshold=0.6, sigmoid_scale=2.0, use_sigmoid=True)
    print(f"✓ Config created: threshold={config.threshold}, scale={config.sigmoid_scale}")
    
    # Create synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(50, 3)
    y_train = (X_train[:, 0] > 0).float()
    
    # Train with minimal params
    classifier = train_binary_classifier(
        X_train, y_train,
        algorithm='gp',
        pop_size=10,
        n_iter=3,
        threshold=config.threshold,
        sigmoid_scale=config.sigmoid_scale,
        verbose=0
    )
    
    print(f"✓ Classifier matches config: threshold={classifier.threshold}")
    assert classifier.threshold == config.threshold
    assert classifier.sigmoid_scale == config.sigmoid_scale
    
    return True


def test_validators():
    """Test validator functions."""
    print("\n" + "=" * 70)
    print("TEST 3: Validator Functions")
    print("=" * 70)
    
    # Test valid inputs
    try:
        validate_threshold(0.5)
        print("✓ validate_threshold(0.5) passed")
    except Exception as e:
        print(f"✗ validate_threshold failed: {e}")
        return False
    
    try:
        validate_scaling_factor(1.0)
        print("✓ validate_scaling_factor(1.0) passed")
    except Exception as e:
        print(f"✗ validate_scaling_factor failed: {e}")
        return False
    
    # Test invalid threshold
    try:
        validate_threshold(1.5)
        print("✗ validate_threshold(1.5) should have raised error")
        return False
    except Exception:
        print("✓ validate_threshold(1.5) correctly raised error")
    
    # Test invalid scaling factor
    try:
        validate_scaling_factor(-1.0)
        print("✗ validate_scaling_factor(-1.0) should have raised error")
        return False
    except Exception:
        print("✓ validate_scaling_factor(-1.0) correctly raised error")
    
    # Test binary labels
    labels = torch.tensor([0., 1., 1., 0.])
    try:
        validate_binary_labels(labels)
        print("✓ validate_binary_labels with binary labels passed")
    except Exception as e:
        print(f"✗ validate_binary_labels failed: {e}")
        return False
    
    return True


def test_factory_pattern():
    """Test algorithm factory."""
    print("\n" + "=" * 70)
    print("TEST 4: Algorithm Factory Pattern")
    print("=" * 70)
    
    factory = get_default_factory()
    print(f"✓ Default factory obtained: {type(factory).__name__}")
    
    algorithms = factory.available_algorithms()
    print(f"✓ Available algorithms: {algorithms}")
    assert 'gp' in algorithms
    assert 'gsgp' in algorithms
    assert 'slim' in algorithms
    
    # Test factory creation
    torch.manual_seed(42)
    X = torch.randn(30, 3)
    y = torch.randn(30)
    
    model = factory.create('gp', X, y, pop_size=5, n_iter=2, verbose=0)
    print(f"✓ Factory created GP model: {type(model).__name__}")
    
    return True


def test_metrics_structure():
    """Test metrics calculation and structure."""
    print("\n" + "=" * 70)
    print("TEST 5: Metrics Structure")
    print("=" * 70)
    
    y_true = torch.tensor([0., 1., 1., 0., 1., 0., 1., 0.])
    y_pred = torch.tensor([0., 1., 0., 0., 1., 1., 1., 0.])
    
    metrics = calculate_binary_metrics(y_true, y_pred)
    print(f"✓ Metrics calculated: {type(metrics).__name__}")
    
    # Check required keys
    required_keys = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
        print(f"  - {key}: {metrics[key]:.4f}")
    
    # Check confusion matrix components
    cm_keys = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']
    for key in cm_keys:
        assert key in metrics, f"Missing key: {key}"
        print(f"  - {key}: {metrics[key]}")
    
    return True


def test_apply_sigmoid():
    """Test sigmoid utility function."""
    print("\n" + "=" * 70)
    print("TEST 6: Sigmoid Utility")
    print("=" * 70)
    
    x = torch.tensor([-2., -1., 0., 1., 2.])
    result = apply_sigmoid(x, scaling_factor=1.0)
    
    print(f"✓ Sigmoid applied: input shape={x.shape}, output shape={result.shape}")
    print(f"  - Input: {x.tolist()}")
    print(f"  - Output: {[f'{v:.4f}' for v in result.tolist()]}")
    
    # Check bounds
    assert torch.all(result >= 0) and torch.all(result <= 1)
    print("✓ All values in valid range [0, 1]")
    
    # Check sigmoid(0) = 0.5
    assert abs(result[2].item() - 0.5) < 0.001
    print("✓ sigmoid(0) ≈ 0.5")
    
    return True


def main():
    """Run all baseline tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 BASELINE TESTS - CURRENT IMPLEMENTATION")
    print("=" * 70 + "\n")
    
    tests = [
        ("Basic Classification", test_basic_classification),
        ("ClassifierConfig", test_classifier_config),
        ("Validators", test_validators),
        ("Factory Pattern", test_factory_pattern),
        ("Metrics Structure", test_metrics_structure),
        ("Apply Sigmoid", test_apply_sigmoid),
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
