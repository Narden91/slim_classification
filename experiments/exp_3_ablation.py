
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
from slim_gsgp.main_slim import slim
from slim_gsgp.utils.utils import train_test_split
from slim_gsgp.evaluators.fitness_functions import rmse
from experiments.metrics import get_all_metrics

# Configuration for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(run_id, output_dir, pop_size=100, n_iter=100):
    
    dataset_name = "darwin"
    # Ablation: Varying p_inflate (Probability of Inflation Mutation)
    # Default is usually 0.2. We test 0.2, 0.5, 0.8.
    p_inflate_values = [0.2, 0.5, 0.8]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Data
    X, y = load_darwin(X_y=True)
    
    # Split Data (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.3, seed=run_id)
    
    results = []

    for p_inflate in p_inflate_values:
        variant_name = f"SLIM_p_inflate_{p_inflate}"
        print(f"[{variant_name}] Running on {dataset_name} | Run: {run_id}")
        log_path = os.path.join(output_dir, f"{variant_name}_{dataset_name}_run_{run_id}.csv")
        
        start_time = time.time()
        
        # Run SLIM with specific p_inflate
        final_model = slim(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            dataset_name=dataset_name,
            slim_version="SLIM+SIG2",
            pop_size=pop_size,
            n_iter=n_iter,
            p_xo=0.0,
            elitism=True,
            n_elites=1,
            init_depth=6,
            p_inflate=p_inflate, # Ablation Parameter
            log_path=log_path,
            seed=run_id,
            verbose=1,
            n_jobs=1,
            reconstruct=True
        )
            
        end_time = time.time()
        train_time = end_time - start_time
        
        # Evaluation
        if final_model:
            preds = final_model.predict(X_test)
            test_rmse = rmse(y_test, preds).item()
            
            # Classification Metrics
            metrics = get_all_metrics(y_test, preds)
            
            # Tree Size
            if hasattr(final_model, 'nodes_count'):
                tree_size = final_model.nodes_count
            elif hasattr(final_model, 'node_count'):
                tree_size = final_model.node_count
            else:
                 tree_size = 0
            
            result_row = {
                "Variant": variant_name,
                "P_Inflate": p_inflate,
                "Run_ID": run_id,
                "Train_Time": train_time,
                "Test_RMSE": test_rmse,
                "Tree_Size": tree_size,
                **metrics
            }
            results.append(result_row)
            print(f"[{variant_name}] Finished. Test RMSE: {test_rmse:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
    # Save results to a summary file
    summary_path = os.path.join(output_dir, f"summary_ablation_run_{run_id}.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"Results saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3: SLIM Ablation Study")
    parser.add_argument("--run_id", type=int, default=0, help="Seed/Run ID")
    parser.add_argument("--output_dir", type=str, default="experiment_results/exp3_ablation", help="Directory to save logs")
    parser.add_argument("--pop_size", type=int, default=100, help="Population Size")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of Generations")
    
    args = parser.parse_args()
    
    set_seed(args.run_id)
    run_experiment(args.run_id, args.output_dir, args.pop_size, args.n_iter)
