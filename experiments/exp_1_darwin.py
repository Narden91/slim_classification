
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
import time

# Add project root to path
sys.path.append(os.getcwd())

from slim_gsgp.datasets.data_loader import load_darwin
from slim_gsgp.main_gp import gp
from slim_gsgp.main_gsgp import gsgp
from slim_gsgp.main_slim import slim
from slim_gsgp.classification import register_classification_fitness_functions
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.evaluators.fitness_functions import rmse
from experiments.metrics import get_all_metrics
from experiments.model_utils import SLIM_VERSIONS, extract_model_size, extract_block_count

# Configuration for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(run_id, output_dir, pop_size=500, n_iter=2000, sigmoid_scale=1.0, fitness_function="binary_cross_entropy"):
    
    dataset_name = "darwin"
    algorithms = ["GP", "GSGP"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Register binary fitness functions for GP/GSGP/SLIM
    register_classification_fitness_functions()
    
    # Load Data
    X, y = load_darwin(X_y=True)
    
    # Split Data (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.3, seed=run_id)
    
    results = []

    for algorithm in algorithms:
        print(f"[{algorithm}] Running on {dataset_name} | Run: {run_id}")
        log_path = os.path.join(output_dir, f"{algorithm}_{dataset_name}_run_{run_id}.csv")
        
        start_time = time.time()
        final_model = None
        
        if algorithm == "GP":
            final_model = gp(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                dataset_name=dataset_name,
                pop_size=pop_size,
                n_iter=n_iter,
                p_xo=0.8,
                elitism=True,
                n_elites=1,
                init_depth=6,
                log_path=log_path,
                seed=run_id,
                verbose=1,
                n_jobs=1,
                fitness_function=fitness_function
            )
        elif algorithm == "GSGP":
            final_model = gsgp(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                dataset_name=dataset_name,
                pop_size=pop_size,
                n_iter=n_iter,
                p_xo=0.0,
                elitism=True,
                n_elites=1,
                init_depth=6,
                log_path=log_path,
                seed=run_id,
                verbose=1,
                n_jobs=1,
                reconstruct=True,
                fitness_function=fitness_function
            )
        end_time = time.time()
        train_time = end_time - start_time
        
        # Evaluation
        if final_model:
            preds = final_model.predict(X_test)
            test_rmse = rmse(y_test, preds).item()
            
            # Classification Metrics
            metrics = get_all_metrics(y_test, preds, sigmoid_scale=sigmoid_scale)
            
            tree_size = extract_model_size(final_model)
            blocks = extract_block_count(final_model)
            
            result_row = {
                "Algorithm": algorithm,
                "SLIM_Version": "",
                "Run_ID": run_id,
                "Train_Time": train_time,
                "Main_Metric_Name": metrics["main_metric_name"],
                "Main_Metric_Value": metrics["main_metric_value"],
                "Test_RMSE": test_rmse,
                "Tree_Size": tree_size,
                "Blocks": blocks,
                **metrics
            }
            results.append(result_row)
            print(f"[{algorithm}] Finished. Accuracy: {metrics['accuracy']:.4f}, Test RMSE: {test_rmse:.4f}")

    for slim_version in SLIM_VERSIONS:
        algorithm = "SLIM"
        version_safe = slim_version.replace("*", "MUL")
        run_tag = f"{algorithm}_{version_safe}"
        print(f"[{run_tag}] Running on {dataset_name} | Run: {run_id}")
        log_path = os.path.join(output_dir, f"{run_tag}_{dataset_name}_run_{run_id}.csv")

        start_time = time.time()
        final_model = slim(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            dataset_name=dataset_name,
            slim_version=slim_version,
            pop_size=pop_size,
            n_iter=n_iter,
            p_xo=0.0,
            elitism=True,
            n_elites=1,
            init_depth=6,
            log_path=log_path,
            seed=run_id,
            verbose=1,
            n_jobs=1,
            reconstruct=True,
            fitness_function=fitness_function
        )

        end_time = time.time()
        train_time = end_time - start_time

        if final_model:
            preds = final_model.predict(X_test)
            test_rmse = rmse(y_test, preds).item()
            metrics = get_all_metrics(y_test, preds, sigmoid_scale=sigmoid_scale)
            tree_size = extract_model_size(final_model)
            blocks = extract_block_count(final_model)

            result_row = {
                "Algorithm": algorithm,
                "SLIM_Version": slim_version,
                "Run_ID": run_id,
                "Train_Time": train_time,
                "Main_Metric_Name": metrics["main_metric_name"],
                "Main_Metric_Value": metrics["main_metric_value"],
                "Test_RMSE": test_rmse,
                "Tree_Size": tree_size,
                "Blocks": blocks,
                **metrics
            }
            results.append(result_row)
            print(f"[{run_tag}] Finished. Accuracy: {metrics['accuracy']:.4f}, Test RMSE: {test_rmse:.4f}")
            
    # Save results to a summary file
    summary_path = os.path.join(output_dir, f"summary_darwin_run_{run_id}.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: DARWIN Benchmark")
    parser.add_argument("--run_id", type=int, default=0, help="Seed/Run ID")
    parser.add_argument("--output_dir", type=str, default="experiment_results/exp1_darwin", help="Directory to save logs")
    parser.add_argument("--pop_size", type=int, default=500, help="Population Size")
    parser.add_argument("--n_iter", type=int, default=2000, help="Number of Generations")
    
    # SLURM Compatibility Arguments (Ignored or Validated)
    parser.add_argument("--dataset", type=str, help="Dataset name (ignored, hardcoded to darwin)")
    parser.add_argument("--algorithm", type=str, help="Algorithm name (ignored, runs all)")
    parser.add_argument("--slim-version", type=str, help="SLIM version (ignored)")
    parser.add_argument("--max-depth", type=str, help="Max depth (ignored)")
    parser.add_argument("--p-inflate", type=float, help="P inflate (ignored)")
    parser.add_argument("--sigmoid-scale", type=float, default=1.0, help="Sigmoid scale")
    parser.add_argument("--fitness-function", type=str, default="binary_cross_entropy", help="Fitness function")
    parser.add_argument("--seed", type=int, help="Seed (mapped to run_id if run_id is default)")
    
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run to check setup")

    args = parser.parse_args()
    
    # Map seed to run_id if provided and run_id is default
    if args.seed is not None and args.run_id == 0:
        args.run_id = args.seed
        
    if args.dry_run:
        print(f"[{os.path.basename(__file__)}] Performing DRY RUN...")
        try:
            # check data loading
            X, y = load_darwin(X_y=True)
            print(f"[OK] Data loaded successfully. Shape: {X.shape}")
            
            # check algorithm initialization
            print("[INFO] Checking algorithm initialization...")
            gp(X_train=X[:10], y_train=y[:10], X_test=X[:10], y_test=y[:10], 
               dataset_name='darwin', n_iter=1, pop_size=5, verbose=0, n_jobs=1)
            print("[OK] GP initialized")
            
            gsgp(X_train=X[:10], y_train=y[:10], X_test=X[:10], y_test=y[:10], 
                 dataset_name='darwin', n_iter=1, pop_size=5, verbose=0, n_jobs=1, reconstruct=True)
            print("[OK] GSGP initialized")
            
            for slim_version in SLIM_VERSIONS:
                slim(X_train=X[:10], y_train=y[:10], X_test=X[:10], y_test=y[:10],
                     dataset_name='darwin', slim_version=slim_version, n_iter=1, pop_size=5, verbose=0, n_jobs=1, reconstruct=True)
                print(f"[OK] SLIM initialized ({slim_version})")
            
            print("[SUCCESS] Dry run completed successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"[FAIL] Dry run failed: {e}")
            sys.exit(1)
    
    set_seed(args.run_id)
    run_experiment(
        args.run_id,
        args.output_dir,
        args.pop_size,
        args.n_iter,
        args.sigmoid_scale,
        args.fitness_function,
    )
