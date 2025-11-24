#!/usr/bin/env python3
"""
Check GEPA optimization metrics from WandB
"""

import wandb

api = wandb.Api()
project_path = "xuanfeiren-university-of-wisconsin-madison/debug-DSPy"

print(f"Fetching runs from project: {project_path}\n")

try:
    runs = api.runs(project_path)
    print(f"Found {len(runs)} runs in the project:\n")
    
    for i, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Run #{i+1}: {run.name}")
        print(f"Run ID: {run.id}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        
        # Get summary metrics
        summary = run.summary
        
        print(f"\n📊 Summary Metrics:")
        print(f"  - Total Metric Calls: {summary.get('Update/total_metric_calls', 'N/A')}")
        print(f"  - Total Samples: {summary.get('Update/total_samples', 'N/A')}")
        print(f"  - Final Iteration: {summary.get('Update/iteration', 'N/A')}")
        print(f"  - Frontier Size: {summary.get('Update/frontier_size', 'N/A')}")
        print(f"  - Best Test Score: {summary.get('Test/test_score_empirical_mean', 'N/A')}")
        print(f"  - Tasks Solved: {summary.get('Diversity/num_tasks_solved', 'N/A')}")
        
        # Get full history
        history = run.history()
        
        if not history.empty:
            print(f"\n📈 History Info:")
            print(f"  - Total logged steps: {len(history)}")
            
            # Check for metric calls column
            if 'Update/total_metric_calls' in history.columns:
                total_calls = history['Update/total_metric_calls'].max()
                print(f"  - Maximum metric calls reached: {total_calls}")
            
            # Show available columns
            print(f"\n📋 Available metrics:")
            metric_cols = [col for col in history.columns if not col.startswith('_')]
            for col in sorted(metric_cols)[:20]:  # Show first 20
                print(f"    - {col}")
            if len(metric_cols) > 20:
                print(f"    ... and {len(metric_cols) - 20} more")
                
except Exception as e:
    print(f"Error: {e}")
    print("\nPossible reasons:")
    print("1. No runs have been logged to this project yet")
    print("2. WandB authentication issue")
    print("3. The run hasn't been synced to the cloud")

