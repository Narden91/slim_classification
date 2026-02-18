
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze Feature Importance Histograms")
    parser.add_argument("--results_dir", type=str, default="logs/feature_importance", help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save plots")
    args = parser.parse_args()

    # 1. Collect all CSV files
    print(f"Searching for results in {args.results_dir}...")
    # Pattern: logs/feature_importance/{dataset}/depth_{depth}/features_seed_{seed}.csv
    # We can just recursively search for .csv files
    files = glob.glob(os.path.join(args.results_dir, "**", "*.csv"), recursive=True)
    
    if not files:
        print("No result files found!")
        return

    print(f"Found {len(files)} files. Loading...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        print("No valid data loaded.")
        return
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Analysis per Dataset
    datasets = full_df['dataset'].unique()
    os.makedirs(args.output_dir, exist_ok=True)
    
    for dataset in datasets:
        print(f"Analyzing dataset: {dataset}")
        dataset_df = full_df[full_df['dataset'] == dataset]
        
        # --- Plot A: Total Occurrence Frequency ---
        # Sum of 'count' across all seeds for each feature/depth
        agg_occurrences = dataset_df.groupby(['max_depth', 'feature_index'])['count'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=agg_occurrences, x='feature_index', y='count', hue='max_depth', palette='viridis')
        
        plt.title(f"Feature Occurrence Count by Max Depth - {dataset}")
        plt.xlabel("Feature Index")
        plt.ylabel("Total Count (sum over all seeds)")
        plt.legend(title="Max Depth")
        plt.tight_layout()
        
        save_path = os.path.join(args.output_dir, f"{dataset}_total_occurrences.png")
        plt.savefig(save_path)
        plt.close()
        
        # --- Plot B: Selection Probability (Stability) ---
        # Fraction of runs where a feature appeared at all (count > 0)
        
        # Add binary column
        dataset_df['selected'] = (dataset_df['count'] > 0).astype(int)
        
        # Mean of 'selected' gives the probability (0 to 1)
        agg_prob = dataset_df.groupby(['max_depth', 'feature_index'])['selected'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=agg_prob, x='feature_index', y='selected', hue='max_depth', palette='viridis')
        
        plt.title(f"Feature Selection Probability by Max Depth - {dataset}")
        plt.xlabel("Feature Index")
        plt.ylabel("Probability of Selection (fraction of runs)")
        plt.ylim(0, 1.05)
        plt.legend(title="Max Depth")
        plt.tight_layout()
        
        save_path_prob = os.path.join(args.output_dir, f"{dataset}_selection_prob.png")
        plt.savefig(save_path_prob)
        plt.close()
        print(f"Saved plots for {dataset}")

        # --- Plot C: Sparsity (Avg Features Used) ---
        sparsity_data = []
        for depth in dataset_df['max_depth'].unique():
            depth_df = dataset_df[dataset_df['max_depth'] == depth]
            
            # Count unique features used per seed
            # We group by seed and sum 'selected' (number of unique features with count > 0)
            features_per_seed = depth_df.groupby('seed')['selected'].sum()
            
            avg_used = features_per_seed.mean()
            std_used = features_per_seed.std()
            
            sparsity_data.append({
                'max_depth': depth, 
                'avg_used_features': avg_used,
                'std_used_features': std_used
            })
            
        sparsity_df = pd.DataFrame(sparsity_data)
        
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=sparsity_df, x='max_depth', y='avg_used_features', marker='o')
        # Add error bars
        plt.errorbar(sparsity_df['max_depth'], sparsity_df['avg_used_features'], yerr=sparsity_df['std_used_features'], fmt='none', ecolor='gray', capsize=3)
        
        plt.title(f"Average Unique Features Used vs Max Depth - {dataset}")
        plt.xlabel("Max Depth")
        plt.ylabel("Avg Unique Features Selected")
        plt.grid(True)
        plt.tight_layout()
        
        save_path_sparsity = os.path.join(args.output_dir, f"{dataset}_sparsity.png")
        plt.savefig(save_path_sparsity)
        plt.close()

if __name__ == "__main__":
    main()
