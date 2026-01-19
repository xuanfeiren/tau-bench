import json
import pandas as pd
from pathlib import Path
import re

def calculate_eval_steps(gap):
    """
    Calculate eval steps based on gap in total_samples.
    Each combination can have at most one 10-sample eval and at most two 2-sample evals.
    Example: gap = 14 → 1×10 + 2×2 = 14 → 3 eval steps (1 for the 10, 2 for the 2s)
    """
    if gap == 0:
        return 0
    
    total_steps = 0
    remaining = gap
    
    # Process in combinations: at most 1×10 + 2×2 = 14 per combination
    while remaining > 0:
        # Use one 10-sample eval if possible
        if remaining >= 10:
            remaining -= 10
            total_steps += 1
        # Use up to two 2-sample evals for the remainder
        twos = min(remaining // 2, 2)  # At most 2 twos per combination
        remaining -= twos * 2
        total_steps += twos
    
    return total_steps

def load_snapshots(snapshot_dir):
    """Load all snapshot JSON files and return sorted list."""
    snapshot_dir = Path(snapshot_dir)
    snapshots = []
    
    for json_file in sorted(snapshot_dir.glob("snapshot_iter*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            snapshots.append(data)
    
    return snapshots

def load_eval_results(eval_results_file):
    """Load eval results JSON file."""
    with open(eval_results_file, 'r') as f:
        return json.load(f)

def generate_csv_data(snapshots, eval_results, output_file):
    """
    Generate CSV data from snapshots and eval results.
    
    CSV iteration = JSON iteration + 1
    """
    rows = []
    
    # Create a mapping from JSON iteration to score
    score_map = {}
    for eval_result in eval_results:
        json_iter = eval_result['iteration']
        # Skip iteration -1 if present
        if json_iter < 0:
            continue
        score_map[json_iter] = eval_result['score']
    
    # Add iteration 0 row (everything 0, score will be set later)
    rows.append({
        'iteration': 0,
        'eval_step': 0,
        'prop_step': 0,
        'num_samples': 0,
        'num_proposals': 0,
        'score': None  # Start with empty score
    })
    
    # Process snapshots - first pass: set all scores to None (empty)
    prev_total_samples = 0
    prev_eval_step = 0
    
    for snapshot in snapshots:
        json_iter = snapshot['iteration']
        csv_iter = json_iter + 1  # CSV iteration = JSON iteration + 1
        
        total_samples = snapshot['total_samples']
        num_proposals = len(snapshot['program_library'])
        
        # Calculate eval_step based on gap in total_samples
        gap = total_samples - prev_total_samples
        eval_step_increment = calculate_eval_steps(gap)
        eval_step = prev_eval_step + eval_step_increment
        
        # Get score for this iteration (using JSON iteration as key)
        # Only set score if it exists in eval_results, otherwise None
        score = score_map.get(json_iter, None)
        
        rows.append({
            'iteration': csv_iter,
            'eval_step': eval_step,
            'prop_step': csv_iter,  # prop_step = iteration
            'num_samples': total_samples,
            'num_proposals': num_proposals,
            'score': score  # Will be None for most iterations
        })
        
        prev_total_samples = total_samples
        prev_eval_step = eval_step
    
    # Step 1: For iterations with scores, replace with cumulative max (highest score so far)
    # This ensures scores are non-decreasing
    best_score_so_far = None
    for row in rows:
        if row['score'] is not None:
            # Update best_score_seen_so_far if current score is higher
            if best_score_so_far is None or row['score'] > best_score_so_far:
                best_score_so_far = row['score']
            # Replace score with cumulative max
            row['score'] = best_score_so_far
    
    # Step 2: Forward fill missing scores using the previous score
    prev_score = None
    for row in rows:
        if row['score'] is not None:
            prev_score = row['score']
        else:
            # Forward fill empty scores with previous score
            if prev_score is not None:
                row['score'] = prev_score
    
    # Step 3: For iteration 0, use score from iteration 1 if available
    if rows[0]['score'] is None and len(rows) > 1:
        rows[0]['score'] = rows[1]['score']
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns
    df = df[['iteration', 'eval_step', 'prop_step', 'num_samples', 'num_proposals', 'score']]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {output_file} with {len(df)} rows")

if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent.parent
    dspy_results_dir = base_dir / "dspy_results"
    
    # Process runs 1-6
    for run_num in range(1, 7):
        run_name = f"gepa_Nov25_{run_num}"
        run_dir = dspy_results_dir / run_name
        
        if not run_dir.exists():
            print(f"\nWarning: {run_name} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {run_name}")
        print(f"{'='*60}")
        
        snapshot_dir = run_dir / "pareto_snapshots"
        eval_results_most_frequent = run_dir / "eval_results" / "eval_results_most_frequent.json"
        eval_results_sample_by_freq = run_dir / "eval_results" / "eval_results_sample_by_freq.json"
        
        # Check if required files exist
        if not snapshot_dir.exists():
            print(f"  Warning: {snapshot_dir} not found, skipping...")
            continue
        
        if not eval_results_most_frequent.exists():
            print(f"  Warning: {eval_results_most_frequent} not found, skipping most_frequent...")
        else:
            # Generate data for most_frequent
            output_dir = Path(__file__).parent / "gepa_most_frequent"
            output_dir.mkdir(exist_ok=True)
            
            print(f"  Loading snapshots from {run_name}...")
            snapshots = load_snapshots(snapshot_dir)
            print(f"  Loaded {len(snapshots)} snapshots")
            
            print(f"  Generating data_{run_num}.csv for most_frequent...")
            eval_results = load_eval_results(eval_results_most_frequent)
            output_file = output_dir / f"data_{run_num}.csv"
            generate_csv_data(snapshots, eval_results, output_file)
        
        if not eval_results_sample_by_freq.exists():
            print(f"  Warning: {eval_results_sample_by_freq} not found, skipping sample_by_freq...")
        else:
            # Generate data for sample_by_freq
            output_dir = Path(__file__).parent / "gepa_sample_by_freq"
            output_dir.mkdir(exist_ok=True)
            
            if 'snapshots' not in locals() or len(snapshots) == 0:
                print(f"  Loading snapshots from {run_name}...")
                snapshots = load_snapshots(snapshot_dir)
                print(f"  Loaded {len(snapshots)} snapshots")
            
            print(f"  Generating data_{run_num}.csv for sample_by_freq...")
            eval_results = load_eval_results(eval_results_sample_by_freq)
            output_file = output_dir / f"data_{run_num}.csv"
            generate_csv_data(snapshots, eval_results, output_file)
    
    print(f"\n{'='*60}")
    print("Done processing all runs!")
    print(f"{'='*60}")
