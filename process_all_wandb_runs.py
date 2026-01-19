#!/usr/bin/env python3
"""
Process multiple wandb run types and generate CSV files.
Extracts data from wandb project "tau-bench-10-tasks-10-evals" 
for different algorithm configurations.
"""

import csv
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb package is required. Install with: pip install wandb")
    exit(1)

PROJECT_NAME = "tau-bench-10-tasks-10-evals"
METRIC_NAME = "Test/test_score_empirical_mean"
SAMPLES_METRIC = "Update/total_samples"

# Configuration: (run_name_pattern, output_dir)
CONFIGURATIONS = [
    ("PS-mean_exploration-Nov7", "my_data_over_multiple_x/vanilla_PS"),
    ("epsnet_0-PS_with_summarizer", "my_data_over_multiple_x/PS_summarizer"),
    ("epsnet_0.1_PS", "my_data_over_multiple_x/PS_epsnet"),
]


def fetch_runs_by_pattern(pattern):
    """
    Fetch runs from wandb matching the pattern.
    
    Args:
        pattern: Run name pattern to match
    
    Returns:
        List of matching wandb Run objects
    """
    print(f"\n{'='*60}")
    print(f"Fetching runs matching: '{pattern}'")
    print(f"{'='*60}")
    
    api = wandb.Api()
    
    # Get all runs from the project
    try:
        runs = api.runs(f"xuanfeiren-university-of-wisconsin-madison/{PROJECT_NAME}")
    except Exception as e:
        print(f"Error connecting to wandb: {e}")
        print("Make sure you're logged in with: wandb login")
        return []
    
    # Filter runs matching the pattern (exclude "-old" runs)
    matching_runs = []
    for run in runs:
        if pattern in run.name and "-old" not in run.name:
            matching_runs.append(run)
    
    # Sort by run name to ensure consistent ordering
    matching_runs.sort(key=lambda r: r.name)
    
    print(f"Found {len(matching_runs)} runs matching '{pattern}'")
    
    if len(matching_runs) > 0:
        print("Matching runs:")
        for r in matching_runs:
            print(f"  {r.name}")
    
    return matching_runs


def process_run_to_csv(run, run_index, output_dir):
    """
    Process a single wandb run and generate CSV file.
    
    Args:
        run: wandb Run object
        run_index: Index of the run (1-N)
        output_dir: Output directory for CSV files
    """
    print(f"\nProcessing run {run_index}: {run.name} (ID: {run.id})")
    
    # Get run history
    try:
        history = run.history()
    except Exception as e:
        print(f"  Error: Could not fetch history: {e}")
        return
    
    if history.empty:
        print(f"  Warning: Empty history for run {run.name}")
        return
    
    # Check available columns
    print(f"  Available columns: {list(history.columns)}")
    
    # Prepare CSV data
    csv_data = []
    
    # Track maximum score so far
    max_score_so_far = None
    
    # First, check if there's data at step 0
    step_0_num_samples = 0
    
    # Process each row in history to find step 0 and track max score
    for idx, row in history.iterrows():
        step = row.get('_step', idx)
        iteration = int(step)
        
        # Check for step 0 data
        if iteration == 0:
            step_0_num_samples = row.get(SAMPLES_METRIC)
            if pd.isna(step_0_num_samples):
                step_0_num_samples = 0
            else:
                step_0_num_samples = int(step_0_num_samples)
            
            # Check for score at step 0
            score_value = row.get(METRIC_NAME)
            if pd.notna(score_value):
                score_value = float(score_value)
                if max_score_so_far is None or score_value > max_score_so_far:
                    max_score_so_far = score_value
    
    # Add initial row (iteration 0)
    csv_data.append({
        "iteration": 0,
        "eval_step": 0,
        "prop_step": 0,
        "num_samples": step_0_num_samples,
        "num_proposals": 0,
        "score": max_score_so_far if max_score_so_far is not None else ""
    })
    
    # Process each row in history (skip step 0 since already added)
    for idx, row in history.iterrows():
        step = row.get('_step', idx)
        
        # iteration = step (start from 0)
        iteration = int(step)
        
        # Skip if iteration is 0 (already added)
        if iteration == 0:
            continue
        
        # Calculate derived values
        eval_step = 2 * iteration
        prop_step = iteration
        num_proposals = 5 * iteration
        
        # Get num_samples
        num_samples = row.get(SAMPLES_METRIC)
        if pd.isna(num_samples):
            num_samples = 0
        else:
            num_samples = int(num_samples)
        
        # Get score
        score_value = row.get(METRIC_NAME)
        if pd.notna(score_value):
            score_value = float(score_value)
            # Update maximum score so far
            if max_score_so_far is None or score_value > max_score_so_far:
                max_score_so_far = score_value
        
        # Use maximum score so far if we have one, otherwise keep empty
        if max_score_so_far is not None:
            score = max_score_so_far
        else:
            score = ""
        
        # Add row
        csv_data.append({
            "iteration": iteration,
            "eval_step": eval_step,
            "prop_step": prop_step,
            "num_samples": num_samples,
            "num_proposals": num_proposals,
            "score": score
        })
    
    # Write CSV file
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"data_{run_index}.csv")
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "eval_step", "prop_step", "num_samples", "num_proposals", "score"])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"  Generated: {csv_file}")
    print(f"  Total rows: {len(csv_data)}")
    print(f"  Max iteration: {csv_data[-1]['iteration'] if csv_data else 0}")


def process_configuration(pattern, output_dir):
    """
    Process all runs matching a pattern and generate CSV files.
    
    Args:
        pattern: Run name pattern to match
        output_dir: Output directory for CSV files
    """
    # Fetch runs
    runs = fetch_runs_by_pattern(pattern)
    
    if not runs:
        print(f"No runs found matching '{pattern}'. Skipping.")
        return 0
    
    # Process each run
    for idx, run in enumerate(runs, start=1):
        process_run_to_csv(run, idx, output_dir)
    
    print(f"\n=== Processing Complete for {pattern} ===")
    print(f"Generated {len(runs)} CSV files in {output_dir}/")
    
    return len(runs)


def main():
    """Main function to process all configurations."""
    print(f"Connecting to wandb project: {PROJECT_NAME}")
    
    total_files = 0
    
    # Process each configuration
    for pattern, output_dir in CONFIGURATIONS:
        count = process_configuration(pattern, output_dir)
        total_files += count
    
    print(f"\n{'='*60}")
    print(f"=== Overall Summary ===")
    print(f"Total CSV files generated: {total_files}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
