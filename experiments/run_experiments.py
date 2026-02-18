
import argparse
import os
import sys
import time

# Add project root to path
sys.path.append(os.getcwd())

from experiments import exp_1_darwin
from experiments import exp_2_hand_stat
from experiments import exp_3_ablation
from experiments import exp_4_depth_ablation

def run_all(n_runs=30, start_run=0, selected_experiments=None, n_jobs=1):
    
    experiments_map = {
        1: ("Experiment 1 (DARWIN)", exp_1_darwin),
        2: ("Experiment 2 (HAND_STAT)", exp_2_hand_stat),
        3: ("Experiment 3 (Ablation p_inflate)", exp_3_ablation),
        4: ("Experiment 4 (Ablation init_depth)", exp_4_depth_ablation)
    }
    
    if selected_experiments is None:
        selected_experiments = [1, 2, 3, 4]
        
    for run_id in range(start_run, start_run + n_runs):
        print(f"\n{'='*30}\nStarting Run ID: {run_id}\n{'='*30}")
        
        for exp_id in selected_experiments:
            if exp_id not in experiments_map:
                print(f"Warning: Experiment {exp_id} not found. Skipping.")
                continue
                
            exp_name, exp_module = experiments_map[exp_id]
            print(f"\n--- Running {exp_name} ---")
            
            # Define output directory based on experiment
            output_dir = os.path.join("experiment_results", f"exp{exp_id}_{exp_module.__name__.split('_')[-1]}")
            
            # Run the experiment
            # Note: pop_size and n_iter are defaulted in the specific scripts to 100/100
            # We let the scripts use their internal defaults unless overridden here (which we don't, to respect the "specified from code" request)
            # However, for testing purposes, we might want to override.
            # But for full reproduction, we pass only run_id and output_dir, letting defaults handle the rest.
            
            try:
                exp_module.run_experiment(run_id=run_id, output_dir=output_dir)
                print(f"✓ {exp_name} completed for Run {run_id}")
            except Exception as e:
                print(f"❌ {exp_name} FAILED for Run {run_id}")
                print(e)
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLIM Classification Experiments")
    parser.add_argument("--experiments", type=str, default="1,2,3,4", help="Comma-separated list of experiment IDs to run (e.g., '1,2')")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs to execute per experiment")
    parser.add_argument("--start_run", type=int, default=0, help="Starting Run ID")
    
    args = parser.parse_args()
    
    selected_exps = [int(x.strip()) for x in args.experiments.split(',') if x.strip()]
    
    run_all(n_runs=args.n_runs, start_run=args.start_run, selected_experiments=selected_exps)
