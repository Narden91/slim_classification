import argparse
import os
import sys
import shutil
import importlib.util
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.getcwd())

from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from slim_gsgp.utils.utils import get_terminals, protected_div
from slim_gsgp.evaluators.fitness_functions import binary_cross_entropy, rmse
from slim_gsgp.explainability.feature_importance import FeatureImportanceExtractor
from slim_gsgp.explainability.tree_exporter import TreeExporter
from slim_gsgp.main_slim import slim


def resolve_export_formats(raw_formats: str):
    allowed = {"text", "html", "svg", "pdf"}
    requested = [fmt.strip().lower() for fmt in raw_formats.split(",") if fmt.strip()]
    if not requested:
        requested = ["text"]

    has_plotly = importlib.util.find_spec("plotly") is not None
    has_kaleido = importlib.util.find_spec("kaleido") is not None
    has_chrome = any(
        shutil.which(cmd)
        for cmd in ("google-chrome", "chrome", "chromium", "chromium-browser")
    )

    selected = []
    for fmt in requested:
        if fmt not in allowed:
            print(f"Skipping unsupported export format: {fmt}")
            continue
        if fmt in {"html", "svg", "pdf"} and not has_plotly:
            if fmt == "svg":
                selected.append(fmt)
                continue
            print(f"Skipping {fmt}: plotly is not installed")
            continue
        if fmt == "pdf" and (not has_kaleido or not has_chrome):
            print(f"Skipping {fmt}: requires kaleido and a Chrome/Chromium binary")
            continue
        selected.append(fmt)

    if not selected:
        selected = ["text"]
        print("No visual export backend available. Falling back to text export only.")

    return selected

def load_dataset(dataset_name, benchmark_dir="slim_gsgp/datasets/benchmark"):
    """
    Load dataset from benchmark directory or standard data loader.
    """
    # 1. Try benchmark directory (CSVs)
    csv_path = os.path.join(benchmark_dir, f"{dataset_name}.csv")
    txt_path = os.path.join(benchmark_dir, f"{dataset_name}.txt")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    elif os.path.exists(txt_path):
        df = pd.read_csv(txt_path, sep=" ", header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y

    # 2. Try standard data loader functions (fallback)
    from slim_gsgp.datasets import data_loader
    func_name = f"load_{dataset_name}"
    if hasattr(data_loader, func_name):
        return getattr(data_loader, func_name)(X_y=True)
        
    raise FileNotFoundError(f"Dataset {dataset_name} not found in {benchmark_dir} and no loader found in data_loader.py")

def main():
    parser = argparse.ArgumentParser(description="Run SLIM Explainability Experiment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--max_depth", type=int, required=True, help="Max depth for SLIM")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="logs_explainability", help="Output directory")
    
    # Standard SLIM args
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--n_iter", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--slim_version", type=str, default="SLIM+SIG2", help="SLIM version")
    parser.add_argument("--p_inflate", type=float, default=0.7, help="Probability of inflate mutation")
    parser.add_argument("--export_formats", type=str, default="html,svg,text", help="Comma-separated formats to export (e.g. text,html,svg,pdf)")
    
    args = parser.parse_args()
    
    print(f"Starting Explainability Experiment: Dataset={args.dataset}, MaxDepth={args.max_depth}, Seed={args.seed}")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load Data
    try:
        X, y = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
        
    unique_targets = np.unique(y)
    is_classification = len(unique_targets) <= 2
    
    if is_classification:
        print("Detected task: Binary Classification")
        y[y == unique_targets[0]] = 0
        y[y == unique_targets[1]] = 1
        fitness_function = "binary_cross_entropy"
    else:
        print("Detected task: Regression")
        fitness_function = "rmse"

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y if is_classification else None
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # Adjust init_depth if max_depth is small
    init_depth = 6
    if args.max_depth is not None and args.max_depth < 10:
        init_depth = max(2, args.max_depth - 2)
        print(f"Adjusted init_depth to {init_depth} for max_depth={args.max_depth}")

    slim_version_safe = args.slim_version.replace("*", "MUL")
    output_dir = os.path.join("results", args.dataset, slim_version_safe, "explainability")
    os.makedirs(output_dir, exist_ok=True)

    export_formats = resolve_export_formats(args.export_formats)
    print(f"Resolved export formats: {','.join(export_formats)}")

    log_path = os.path.join(
        output_dir,
        f"slim_internal_depth_{args.max_depth}_seed_{args.seed}.csv"
    )

    final_tree = slim(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        dataset_name=args.dataset,
        slim_version=args.slim_version,
        max_depth=args.max_depth,
        init_depth=init_depth,  
        pop_size=args.pop_size,
        n_iter=args.n_iter,
        p_inflate=args.p_inflate,
        seed=args.seed,
        reconstruct=True,  # Crucial for feature extraction and export
        n_jobs=1,
        verbose=0,
        log_level=0,
        log_path=log_path,
        fitness_function=fitness_function
    )
    
    # Ensure final output dict is used if SLIM returns one, else assume it's the model
    if isinstance(final_tree, tuple):
        final_tree = final_tree[0]

    if hasattr(final_tree, "model"):
        final_tree = final_tree.model  # Sometimes `slim` returns a result dictionary or wrapper object

    print("Extracting feature importance and exporting tree...")
    
    num_features = X_train.shape[1]
    feature_names = [f"x{i}" for i in range(num_features)]
    
    # Feature Importance
    extractor = FeatureImportanceExtractor(
        n_features=num_features,
        feature_names=feature_names
    )
    
    freq_imp = extractor.frequency_importance(final_tree, normalize=False)
    
    features_output_file = os.path.join(output_dir, f"features_depth_{args.max_depth}_seed_{args.seed}.csv")
    
    rows = []
    for feat_name, count in freq_imp.items():
        feat_idx = int(feat_name[1:])
        rows.append({
            "Dataset": args.dataset,
            "Max Depth": args.max_depth,
            "Seed": args.seed,
            "Feature Index": feat_idx,
            "Feature Name": feat_name,
            "Occurrences in Tree": count
        })
        
    df_res = pd.DataFrame(rows)
    df_res.to_csv(features_output_file, index=False)
    print(f"Feature importance saved to {features_output_file}")
    
    # Export Tree Artifacts (HTML/PDF/Text/SVG)
    exporter = TreeExporter()
    export_results = {}
    for fmt in export_formats:
        res = exporter.export(
            individual=final_tree,
            output_dir=output_dir,
            format=fmt,
            filename=f"final_tree_depth_{args.max_depth}_seed_{args.seed}",
            verbose=True
        )
        export_results.update(res)

    print("Explainability artifacts created:")
    for fmt, path in export_results.items():
        if path:
            print(f" - {fmt}: {path}")

if __name__ == "__main__":
    main()
