#!/usr/bin/env python3
"""
Process gepa results from dspy_results and generate CSV files.
Each run generates 2 CSVs (one for each score version).
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_eval_step_increment(samples_added: int) -> int:
    """
    Calculate eval_step increment based on samples_added.
    
    Args:
        samples_added: Number of samples added (capped at 20)
    
    Returns:
        Integer increment for eval_step
    """
    # Cap at 20
    samples_added = min(samples_added, 20)
    
    if samples_added < 10:
        increment = samples_added / 2
    else:  # samples_added >= 10
        increment = 1 + (samples_added - 10) / 2
    
    return int(increment)


def process_gepa_run(run_dir: str, output_dir: str):
    """
    Process a single gepa run and generate 2 CSV files.
    
    Args:
        run_dir: Path to gepa_Nov25_X directory
        output_dir: Path to output directory for CSVs
    """
    run_name = os.path.basename(run_dir)
    print(f"\nProcessing {run_name}...")
    
    # Paths
    snapshots_dir = os.path.join(run_dir, "pareto_snapshots")
    eval_results_dir = os.path.join(run_dir, "eval_results")
    
    # Check if directories exist
    if not os.path.exists(snapshots_dir):
        print(f"  Warning: {snapshots_dir} does not exist, skipping")
        return
    
    # Load eval results for both versions
    eval_results_most_frequent_path = os.path.join(eval_results_dir, "eval_results_most_frequent.json")
    eval_results_sample_by_freq_path = os.path.join(eval_results_dir, "eval_results_sample_by_freq.json")
    
    eval_scores_most_frequent = {}
    eval_scores_sample_by_freq = {}
    
    if os.path.exists(eval_results_most_frequent_path):
        eval_results_most_frequent = load_json(eval_results_most_frequent_path)
        for result in eval_results_most_frequent:
            iteration = result["iteration"]
            score = result["score"]
            eval_scores_most_frequent[iteration] = score
    
    if os.path.exists(eval_results_sample_by_freq_path):
        eval_results_sample_by_freq = load_json(eval_results_sample_by_freq_path)
        for result in eval_results_sample_by_freq:
            iteration = result["iteration"]
            score = result["score"]
            eval_scores_sample_by_freq[iteration] = score
    
    # Load all snapshots
    snapshot_files = sorted([
        os.path.join(snapshots_dir, f)
        for f in os.listdir(snapshots_dir)
        if f.startswith("snapshot_iter") and f.endswith(".json")
    ], key=lambda x: int(os.path.basename(x).replace("snapshot_iter", "").replace(".json", "")))
    
    if not snapshot_files:
        print(f"  Warning: No snapshot files found in {snapshots_dir}, skipping")
        return
    
    # Load snapshots and extract data
    snapshots_data = []
    for snapshot_file in snapshot_files:
        snapshot = load_json(snapshot_file)
        iteration = snapshot["iteration"]
        total_samples = snapshot["total_samples"]
        snapshots_data.append({
            "iteration": iteration,
            "total_samples": total_samples
        })
    
    # Sort by iteration to ensure correct order
    snapshots_data.sort(key=lambda x: x["iteration"])
    
    # Prepare CSV data
    csv_data_most_frequent = []
    csv_data_sample_by_freq = []
    
    # Add initial row (iteration 0, all columns 0)
    csv_data_most_frequent.append({
        "iteration": 0,
        "eval_step": 0,
        "prop_step": 0,
        "num_samples": 0,
        "num_proposals": 0,
        "score": ""
    })
    csv_data_sample_by_freq.append({
        "iteration": 0,
        "eval_step": 0,
        "prop_step": 0,
        "num_samples": 0,
        "num_proposals": 0,
        "score": ""
    })
    
    # Process each snapshot
    eval_step = 0
    prev_total_samples = None
    
    for snapshot in snapshots_data:
        logging_iteration = snapshot["iteration"]
        total_samples = snapshot["total_samples"]
        
        # CSV iteration = logging iteration + 1
        csv_iteration = logging_iteration + 1
        
        # Calculate samples_added
        if prev_total_samples is None:
            # First iteration: samples_added is total_samples (from 0)
            samples_added = total_samples
        else:
            samples_added = total_samples - prev_total_samples
        
        # Calculate eval_step increment (only for iterations after the first)
        if prev_total_samples is not None:
            eval_step_increment = calculate_eval_step_increment(samples_added)
            eval_step += eval_step_increment
        # For first iteration (csv_iteration == 1), eval_step stays 0
        
        # Get scores (empty if not available)
        score_most_frequent = eval_scores_most_frequent.get(logging_iteration, "")
        score_sample_by_freq = eval_scores_sample_by_freq.get(logging_iteration, "")
        
        # Add row to both CSVs
        # For gepa, num_proposals = prop_step
        csv_data_most_frequent.append({
            "iteration": csv_iteration,
            "eval_step": eval_step,
            "prop_step": csv_iteration,
            "num_samples": total_samples,
            "num_proposals": csv_iteration,
            "score": score_most_frequent
        })
        
        csv_data_sample_by_freq.append({
            "iteration": csv_iteration,
            "eval_step": eval_step,
            "prop_step": csv_iteration,
            "num_samples": total_samples,
            "num_proposals": csv_iteration,
            "score": score_sample_by_freq
        })
        
        prev_total_samples = total_samples
    
    # Forward fill scores
    # For iteration 0, use score from iteration 1
    # For other iterations, use previous iteration's score if current is empty
    if len(csv_data_most_frequent) > 1:
        # Get score from iteration 1 for iteration 0
        iter1_score_most_frequent = csv_data_most_frequent[1]["score"]
        iter1_score_sample_by_freq = csv_data_sample_by_freq[1]["score"]
        
        if iter1_score_most_frequent != "":
            csv_data_most_frequent[0]["score"] = iter1_score_most_frequent
        if iter1_score_sample_by_freq != "":
            csv_data_sample_by_freq[0]["score"] = iter1_score_sample_by_freq
        
        # Forward fill for remaining iterations
        for i in range(1, len(csv_data_most_frequent)):
            # Forward fill most_frequent
            if csv_data_most_frequent[i]["score"] == "":
                prev_score = csv_data_most_frequent[i-1]["score"]
                if prev_score != "":
                    csv_data_most_frequent[i]["score"] = prev_score
            
            # Forward fill sample_by_freq
            if csv_data_sample_by_freq[i]["score"] == "":
                prev_score = csv_data_sample_by_freq[i-1]["score"]
                if prev_score != "":
                    csv_data_sample_by_freq[i]["score"] = prev_score
    
    # Write CSV files
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract run number (e.g., "1" from "gepa_Nov25_1")
    run_num = run_name.split("_")[-1] if "_" in run_name else "1"
    
    csv_file_most_frequent = os.path.join(output_dir, f"data_{run_num}_most_frequent.csv")
    csv_file_sample_by_freq = os.path.join(output_dir, f"data_{run_num}_sample_by_freq.csv")
    
    # Write most_frequent CSV
    with open(csv_file_most_frequent, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "eval_step", "prop_step", "num_samples", "num_proposals", "score"])
        writer.writeheader()
        writer.writerows(csv_data_most_frequent)
    
    # Write sample_by_freq CSV
    with open(csv_file_sample_by_freq, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "eval_step", "prop_step", "num_samples", "num_proposals", "score"])
        writer.writeheader()
        writer.writerows(csv_data_sample_by_freq)
    
    print(f"  Generated: {csv_file_most_frequent}")
    print(f"  Generated: {csv_file_sample_by_freq}")
    print(f"  Total rows: {len(csv_data_most_frequent)}")


def main():
    """Process all gepa runs."""
    # Paths
    dspy_results_dir = "dspy_results"
    output_dir = "my_data_over_multiple_x/gepa"
    
    # Process runs 1-6
    for run_num in range(1, 7):
        run_dir = os.path.join(dspy_results_dir, f"gepa_Nov25_{run_num}")
        if os.path.exists(run_dir):
            process_gepa_run(run_dir, output_dir)
        else:
            print(f"\nWarning: {run_dir} does not exist, skipping")


if __name__ == "__main__":
    main()
