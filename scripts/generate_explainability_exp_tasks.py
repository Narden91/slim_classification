import pandas as pd
import os

def main():
    # 1. Define Design Space
    datasets = [
        "gina", "blood", "eeg", "scene", "fertility", "liver", "ozone", "pc1", 
        "pc3", "qsar", "retinopathy", "spam", "spect", "hill", "ilpd", "kc1", "musk1"
    ]
    
    slim_version = "SLIM+SIG2"
    seeds = range(42, 62)  # 20 seeds
    max_depths = [2, 5, 10, 15, 20]
    
    # 2. Generate Tasks
    tasks = []
    task_id = 0
    
    for dataset in datasets:
        for depth in max_depths:
            for i, seed in enumerate(seeds):
                run_number = i + 1
                tasks.append({
                    "task_id": task_id,
                    "dataset": dataset,
                    "slim_version": slim_version,
                    "seed": seed,
                    "run_number": run_number,
                    "max_depth": depth
                })
                task_id += 1
                
    # 3. Create DataFrame
    df = pd.DataFrame(tasks)
    
    # 4. Save
    output_dir = "config"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "task_list_explainability.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(tasks)} tasks.")
    print(f"Saved to {output_path}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
