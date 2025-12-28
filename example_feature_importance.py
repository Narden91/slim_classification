"""
Real-World Feature Importance Example with Breast Cancer Dataset.

Demonstrates feature importance extraction on a trained GP model for binary classification.
"""
import sys
sys.path.append('.')

import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from slim_gsgp.algorithms.GP.representations.tree import Tree
from slim_gsgp.initializers.initializers import rhh
from slim_gsgp.evaluators.fitness_functions import binary_cross_entropy
from slim_gsgp.explainability.feature_importance import FeatureImportanceExtractor
from slim_gsgp.utils.utils import get_terminals, protected_div
from slim_gsgp.selection.selection_algorithms import tournament_selection_min


def create_operators():
    """Create GP operators."""
    FUNCTIONS = {
        'add': {'function': lambda x, y: x + y, 'arity': 2},
        'sub': {'function': lambda x, y: x - y, 'arity': 2},
        'mul': {'function': lambda x, y: x * y, 'arity': 2},
        'div': {'function': protected_div, 'arity': 2},
    }
    return FUNCTIONS


def main():
    print("=" * 80)
    print("FEATURE IMPORTANCE EXTRACTION - REAL-WORLD EXAMPLE")
    print("Dataset: Breast Cancer Wisconsin (Binary Classification)")
    print("=" * 80)
    print()
    
    # Load and prepare data
    print("Step 1: Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"  Dataset shape: {X.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Classes: {data.target_names}")
    print(f"  First 5 features: {list(feature_names[:5])}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to torch
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()
    
    # Setup GP
    print("Step 2: Initializing GP population (without constants for simplicity)...")
    terminals = get_terminals(X_train)
    functions = create_operators()
    
    Tree.TERMINALS = terminals
    Tree.FUNCTIONS = functions
    Tree.CONSTANTS = {}
    
    # Use grow() initializer instead of rhh() to avoid needing constants
    from slim_gsgp.initializers.initializers import grow
    
    population_tuples = grow(
        init_pop_size=20,
        init_depth=6,
        FUNCTIONS=functions,
        TERMINALS=terminals,
        CONSTANTS={},
        p_c=0.0  # Set to 0 to avoid using constants
    )
    
    # Convert tuples to Tree objects
    population = [Tree(tree_tuple) for tree_tuple in population_tuples]
    
    print(f"  Population size: {len(population)}")
    print(f"  Max depth: 6")
    print()
    
    # Evaluate population
    print("Step 3: Evaluating population...")
    for tree in population:
        tree.evaluate(binary_cross_entropy, X_train, y_train, testing=False)
    
    # Get best individual
    best_tree = min(population, key=lambda t: t.fitness)
    print(f"  Best training fitness: {best_tree.fitness:.4f}")
    
    # Test performance
    best_tree.evaluate(binary_cross_entropy, X_test, y_test, testing=True)
    print(f"  Best test fitness: {best_tree.test_fitness:.4f}")
    print()
    
    # Calculate feature importance
    print("Step 4: Extracting feature importance...")
    print()
    
    # Map generic feature names (x0, x1, ...) to actual feature names
    extractor = FeatureImportanceExtractor(
        n_features=X_train.shape[1],
        feature_names=[f"x{i}" for i in range(X_train.shape[1])]
    )
    
    print("  Method 1: Frequency-based importance (fast)...")
    freq_importance = extractor.frequency_importance(best_tree, normalize=True)
    
    print("  Method 2: Depth-weighted importance (fast)...")
    depth_importance = extractor.depth_weighted_importance(best_tree, normalize=True)
    
    print("  Method 3: Permutation importance (accurate but slower)...")
    perm_importance = extractor.permutation_importance(
        best_tree, X_test, y_test, binary_cross_entropy, n_repeats=5, normalize=True
    )
    
    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE RESULTS")
    print("=" * 80)
    print()
    
    # Get top 10 features from each method
    def get_top_n(importance_dict, n=10):
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    # Frequency-based
    print("Top 10 Features - Frequency-Based:")
    print(f"  {'Feature ID':<15} {'Score':<10} {'Actual Feature Name':<30}")
    print("  " + "-" * 55)
    for feat, score in get_top_n(freq_importance, 10):
        if score > 0:
            idx = int(feat[1:])
            actual_name = feature_names[idx]
            print(f"  {feat:<15} {score:<10.4f} {actual_name:<30}")
    print()
    
    # Depth-weighted
    print("Top 10 Features - Depth-Weighted:")
    print(f"  {'Feature ID':<15} {'Score':<10} {'Actual Feature Name':<30}")
    print("  " + "-" * 55)
    for feat, score in get_top_n(depth_importance, 10):
        if score > 0:
            idx = int(feat[1:])
            actual_name = feature_names[idx]
            print(f"  {feat:<15} {score:<10.4f} {actual_name:<30}")
    print()
    
    # Permutation-based
    print("Top 10 Features - Permutation-Based (Most Reliable):")
    print(f"  {'Feature ID':<15} {'Score':<10} {'Actual Feature Name':<30}")
    print("  " + "-" * 55)
    for feat, score in get_top_n(perm_importance, 10):
        if score > 0:
            idx = int(feat[1:])
            actual_name = feature_names[idx]
            print(f"  {feat:<15} {score:<10.4f} {actual_name:<30}")
    print()
    
    # Count features used in tree
    features_in_tree = set(feat for feat, score in freq_importance.items() if score > 0)
    print("=" * 80)
    print(f"Summary:")
    print(f"  Total features in dataset: {X_train.shape[1]}")
    print(f"  Features used in best tree: {len(features_in_tree)}")
    print(f"  Feature utilization: {len(features_in_tree) / X_train.shape[1] * 100:.1f}%")
    print("=" * 80)
    print()
    
    print("✓ Feature importance extraction completed successfully!")
    print()
    print("Interpretation Guide:")
    print("  - Frequency: How often a feature appears in the tree")
    print("  - Depth-weighted: Features closer to root have higher importance")
    print("  - Permutation: How much accuracy degrades when feature is shuffled")
    print("    (Most reliable for understanding actual predictive importance)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
