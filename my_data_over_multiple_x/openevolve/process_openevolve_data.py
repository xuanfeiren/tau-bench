#!/usr/bin/env python3
"""
Process OpenEvolve data from JSON files to CSV format.

Reads data_run1.json through data_run6.json from results/openevolve_new/
and creates data_1.csv through data_6.csv with columns:
iteration, eval_step, prop_step, num_samples, num_proposals, score
"""

import json
import csv
from pathlib import Path


def process_openevolve_run(json_file_path, output_csv_path, min_checkpoint=20, max_checkpoint=200):
    """
    Process a single OpenEvolve run JSON file and create CSV output.
    
    Args:
        json_file_path: Path to input JSON file (e.g., data_run1.json)
        output_csv_path: Path to output CSV file (e.g., data_1.csv)
        min_checkpoint: Minimum checkpoint to include (default: 20)
        max_checkpoint: Maximum checkpoint to include (default: 200)
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Filter checkpoints: numeric checkpoints between min and max (inclusive)
    checkpoints = []
    for item in data:
        checkpoint = item.get('checkpoint')
        # Skip non-numeric checkpoints (like "final")
        if isinstance(checkpoint, (int, float)) and min_checkpoint <= checkpoint <= max_checkpoint:
            checkpoints.append(item)
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: x['checkpoint'])
    
    # Prepare CSV rows
    rows = []
    max_score_so_far = None
    
    for item in checkpoints:
        checkpoint = item['checkpoint']
        test_score = item.get('test_score')
        
        # Skip if test_score is None or invalid
        if test_score is None:
            continue
        
        # Update cumulative maximum score
        if max_score_so_far is None or test_score > max_score_so_far:
            max_score_so_far = test_score
        
        # Create row
        iteration = checkpoint
        row = {
            'iteration': iteration,
            'eval_step': iteration,
            'prop_step': iteration,
            'num_samples': iteration * 10,
            'num_proposals': iteration,
            'score': max_score_so_far
        }
        rows.append(row)
    
    # Write CSV file
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='') as f:
        fieldnames = ['iteration', 'eval_step', 'prop_step', 'num_samples', 'num_proposals', 'score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Processed {json_file_path.name}: {len(rows)} checkpoints -> {output_csv_path.name}")
    return len(rows)


def main():
    # Paths
    tau_bench_root = Path(__file__).parent.parent.parent
    input_dir = tau_bench_root / "results" / "openevolve_new"
    output_dir = Path(__file__).parent
    
    # Process each run
    for run_num in range(1, 7):
        input_file = input_dir / f"data_run{run_num}.json"
        output_file = output_dir / f"data_{run_num}.csv"
        
        if not input_file.exists():
            print(f"⚠ Warning: {input_file} not found, skipping")
            continue
        
        try:
            process_openevolve_run(input_file, output_file)
        except Exception as e:
            print(f"✗ Error processing {input_file.name}: {e}")
    
    print("\n✓ All processing complete!")


if __name__ == "__main__":
    main()
