#!/usr/bin/env python3
"""
Aggregate results from SLURM array job experiments.
Scans log directories and generates summary statistics.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def parse_log_file(log_path):
    """
    Extract key metrics from a log file.
    
    Returns dict with metrics or None if parsing fails.
    """
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract metrics (adjust patterns based on actual log format)
        metrics = {}
        
        # Look for common patterns
        if 'Experiment completed successfully' in content:
            metrics['status'] = 'completed'
        elif 'error' in content.lower() or 'exception' in content.lower():
            metrics['status'] = 'failed'
        else:
            metrics['status'] = 'unknown'
        
        # Extract test accuracy/error if available
        # Pattern examples - adjust based on actual output
        acc_match = re.search(r'Test Accuracy[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if acc_match:
            metrics['test_accuracy'] = float(acc_match.group(1))
        
        rmse_match = re.search(r'Test RMSE[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if rmse_match:
            metrics['test_rmse'] = float(rmse_match.group(1))
        
        # Extract runtime if logged
        time_match = re.search(r'Total time[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if time_match:
            metrics['runtime_seconds'] = float(time_match.group(1))
        
        return metrics
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def scan_logs(base_dir, datasets, slim_versions, algorithm='slim'):
    """
    Scan log directories and collect results.
    """
    results = []
    
    for dataset in datasets:
        for slim_version in slim_versions:
            # Sanitize SLIM version for filesystem (replace * with MUL)
            slim_version_safe = slim_version.replace('*', 'MUL')
            log_dir = Path(base_dir) / dataset / algorithm / slim_version_safe
            
            if not log_dir.exists():
                continue
            
            # Find all log files (including those with carriage returns or other issues)
            for log_file in log_dir.glob('*.log'):
                # Clean filename for pattern matching
                clean_name = log_file.name.replace('\r', '')
                
                # Extract run number and seed from filename
                match = re.match(r'run_(\d+)_seed_(\d+)\.log', clean_name)
                if match:
                    run_number = int(match.group(1))
                    seed = int(match.group(2))
                    
                    metrics = parse_log_file(log_file)
                    if metrics:
                        results.append({
                            'dataset': dataset,
                            'slim_version': slim_version,
                            'run_number': run_number,
                            'seed': seed,
                            'log_file': str(log_file),
                            **metrics
                        })
    
    return results


def generate_summary(results):
    """
    Generate summary statistics grouped by dataset and SLIM version.
    """
    summary = defaultdict(lambda: {
        'total': 0,
        'completed': 0,
        'failed': 0,
        'unknown': 0,
        'test_accuracy': [],
        'test_rmse': []
    })
    
    for result in results:
        key = (result['dataset'], result['slim_version'])
        summary[key]['total'] += 1
        summary[key][result['status']] += 1
        
        if 'test_accuracy' in result:
            summary[key]['test_accuracy'].append(result['test_accuracy'])
        if 'test_rmse' in result:
            summary[key]['test_rmse'].append(result['test_rmse'])
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Base directory containing logs')
    parser.add_argument('--output', type=str, default='results_summary.csv',
                        help='Output summary CSV file')
    parser.add_argument('--detailed-output', type=str, default='results_detailed.csv',
                        help='Detailed results CSV file')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["gina", "blood", "eeg", "scene", "fertility", "liver", 
                                "ozone", "pc1", "pc3", "qsar", "retinopathy", "spam", 
                                "spect", "hill", "ilpd", "kc1", "musk1", "clima"],
                        help='List of datasets')
    parser.add_argument('--slim-versions', type=str, nargs='+',
                        default=["SLIM+SIG2", "SLIM*SIG2", "SLIM+ABS", 
                                "SLIM*ABS", "SLIM+SIG1", "SLIM*SIG1"],
                        help='List of SLIM versions')
    parser.add_argument('--test', action='store_true',
                        help='Analyze test results')
    
    args = parser.parse_args()
    
    # Override for test mode
    if args.test:
        args.log_dir = 'logs_test'
        args.datasets = ["blood", "eeg"]
        args.slim_versions = ["SLIM+ABS", "SLIM*ABS"]
        args.output = 'results_summary_test.csv'
        args.detailed_output = 'results_detailed_test.csv'
        print("TEST MODE: Analyzing test results")
    
    print(f"Scanning logs in: {args.log_dir}")
    results = scan_logs(args.log_dir, args.datasets, args.slim_versions)
    
    print(f"Found {len(results)} experiment results")
    
    if not results:
        print("No results found. Check log directory and configuration.")
        return
    
    # Save detailed results
    if results:
        with open(args.detailed_output, 'w', newline='') as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Detailed results saved to: {args.detailed_output}")
    
    # Generate and save summary
    summary = generate_summary(results)
    
    summary_rows = []
    for (dataset, slim_version), stats in sorted(summary.items()):
        row = {
            'dataset': dataset,
            'slim_version': slim_version,
            'total_runs': stats['total'],
            'completed': stats['completed'],
            'failed': stats['failed'],
            'unknown': stats['unknown'],
            'completion_rate': f"{stats['completed']/stats['total']*100:.1f}%"
        }
        
        if stats['test_accuracy']:
            import statistics
            row['mean_accuracy'] = statistics.mean(stats['test_accuracy'])
            row['std_accuracy'] = statistics.stdev(stats['test_accuracy']) if len(stats['test_accuracy']) > 1 else 0
        
        if stats['test_rmse']:
            import statistics
            row['mean_rmse'] = statistics.mean(stats['test_rmse'])
            row['std_rmse'] = statistics.stdev(stats['test_rmse']) if len(stats['test_rmse']) > 1 else 0
        
        summary_rows.append(row)
    
    if summary_rows:
        with open(args.output, 'w', newline='') as f:
            fieldnames = list(summary_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary saved to: {args.output}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for row in summary_rows:
        print(f"\n{row['dataset']} - {row['slim_version']}")
        print(f"  Runs: {row['total_runs']} (Completed: {row['completed']}, Failed: {row['failed']})")
        print(f"  Completion Rate: {row['completion_rate']}")
    
    print("\n" + "="*80)
    print(f"Total experiments analyzed: {len(results)}")
    total_completed = sum(r['status'] == 'completed' for r in results)
    total_failed = sum(r['status'] == 'failed' for r in results)
    print(f"Overall completion rate: {total_completed/len(results)*100:.1f}%")
    print(f"Overall failure rate: {total_failed/len(results)*100:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
