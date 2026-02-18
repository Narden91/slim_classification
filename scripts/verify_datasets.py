
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from slim_gsgp.datasets.data_loader import *

# List of datasets from dataset_info_published.txt
datasets = [
    "blood", "clima", "eeg", "fertility", "gina", "hill", "ilpd", "kc1", 
    "liver", "musk1", "ozone", "pc1", "pc3", "qsar", "retinopathy", 
    "scene", "spam", "spect"
]

# Map dataset names to loader functions
# Note: In data_loader.py, some functions are explicit (load_breast_cancer) 
# but many of these might rely on a generic loader or need specific handling.
# Let's check which loaders exist in data_loader.py for these specific names.
# Based on my view of data_loader.py, I saw load_efficiency_heating, load_forest_fires, etc.
# I did NOT see explicit loaders for "blood", "clima", "eeg" etc in the first 800 lines.
# I need to check if there's a generic loader or if I need to implement them.
# However, the user said "all the dataset that will be in the folder".
# Let's try to load them assuming they are in slim_gsgp/datasets/benchmark
# and use a generic loader if possible, or check if specific loaders exist.

# If specific loaders don't exist, we might need to use pandas directly to test.
import pandas as pd
import torch

def verify_datasets():
    print("Verifying datasets...")
    benchmark_dir = os.path.join("slim_gsgp", "datasets", "benchmark")
    
    # Check if benchmark dir exists
    if not os.path.exists(benchmark_dir):
        print(f"ERROR: Benchmark directory not found at {benchmark_dir}")
        return

    results = {}
    
    for ds in datasets:
        # Try finding CSV
        csv_path = os.path.join(benchmark_dir, f"{ds}.csv")
        txt_path = os.path.join(benchmark_dir, f"{ds}.txt")
        
        found = False
        path_used = None
        
        if os.path.exists(csv_path):
            found = True
            path_used = csv_path
        elif os.path.exists(txt_path):
            found = True
            path_used = txt_path
            
        if found:
            try:
                # Try loading
                if path_used.endswith(".csv"):
                    df = pd.read_csv(path_used)
                else:
                    df = pd.read_csv(path_used, sep=" ", header=None) # Assumption for txt
                
                print(f"[OK] {ds:<15} | Shape: {df.shape}")
                results[ds] = "OK"
            except Exception as e:
                print(f"[FAIL] {ds:<15} | Error loading: {e}")
                results[ds] = f"Error: {e}"
        else:
            print(f"[MISSING] {ds:<15} | Not found in {benchmark_dir}")
            results[ds] = "Missing"

    print("\nSummary:")
    passed = sum(1 for status in results.values() if status == "OK")
    print(f"Passed: {passed}/{len(datasets)}")

if __name__ == "__main__":
    verify_datasets()
