#!/usr/bin/env python3
"""
Evaluate GEPA

This script evaluates DSPy results. For example, we can evaluate dspy_results/gepa_Nov25.
We load snapshots from the pareto_snapshots directory. A frequency parameter controls how
often snapshots are loaded (e.g., every 10th snapshot).

For each snapshot, we extract:
    - iteration
    - total_samples
    - a program selected from pareto_front_by_task based on selection criterion:
        * "most_frequent": select the program that appears most frequently
        * "sample_by_freq": sample a program with probability proportional to frequency
    - the program string from program_library

We use evaluate_from_str.py to evaluate the program by passing only the instruction string
while keeping other parameters at their defaults.

Output files are saved under {results_dir}/eval_results/:
    1. eval_results_{select_criterion}.json: Each item contains iteration, total_samples,
       selected program index, and the evaluation score.
    2. scores.json: Each item contains program index and its evaluation score (shared cache).

Caching mechanism:
    When evaluating a new program, we first check scores.json for an existing score.
    If found, we use the cached score. Otherwise, we evaluate the program and save the
    score to scores.json, then add the result to eval_results.json.

Usage:
    python my_processing_agents/DSPy_GEPA_eval.py --dir dspy_results/gepa_Nov25_5 --frequency 30 --select_criterion most_frequent

    python my_processing_agents/DSPy_GEPA_eval.py --dir dspy_results/gepa_Nov25_5 --frequency 30 --select_criterion sample_by_freq
"""

import os
import json
import glob
import random
import argparse
from collections import Counter
from evaluate_from_str import evaluate_agent_from_str


def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath, data):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_program_counts(pareto_front_by_task):
    """
    Count occurrences of each program across all tasks.
    
    Args:
        pareto_front_by_task: dict mapping task_id -> list of program indices
    
    Returns:
        Counter with program index -> count
    """
    program_counts = Counter()
    for task_id, program_indices in pareto_front_by_task.items():
        for prog_idx in program_indices:
            program_counts[prog_idx] += 1
    return program_counts


def select_program(pareto_front_by_task, criterion="most_frequent"):
    """
    Select a program from the pareto front based on the specified criterion.
    
    Args:
        pareto_front_by_task: dict mapping task_id -> list of program indices
        criterion: selection method
            - "most_frequent": select the program that appears most frequently
            - "sample_by_freq": sample a program with probability proportional to frequency
    
    Returns:
        The selected program index
    """
    program_counts = get_program_counts(pareto_front_by_task)
    
    if not program_counts:
        return None
    
    if criterion == "most_frequent":
        # Return the most common program (smallest index breaks ties)
        most_common = program_counts.most_common()
        max_count = most_common[0][1]
        # Among programs with max count, choose the smallest index for consistency
        candidates = [prog_idx for prog_idx, count in most_common if count == max_count]
        return min(candidates)
    
    elif criterion == "sample_by_freq":
        # Sample a program with probability proportional to its frequency
        programs = list(program_counts.keys())
        counts = [program_counts[p] for p in programs]
        total = sum(counts)
        weights = [c / total for c in counts]
        return random.choices(programs, weights=weights, k=1)[0]
    
    else:
        raise ValueError(f"Unknown selection criterion: {criterion}")


def get_snapshot_files(pareto_dir, frequency=1):
    """
    Get snapshot files at the specified frequency.
    
    Args:
        pareto_dir: directory containing snapshot files
        frequency: load every Nth snapshot (1 = all, 10 = every 10th)
    
    Returns:
        List of snapshot file paths sorted by iteration
    """
    pattern = os.path.join(pareto_dir, "snapshot_iter*.json")
    all_files = sorted(glob.glob(pattern))
    
    if frequency <= 1:
        return all_files
    
    selected_files = []
    for filepath in all_files:
        # Extract iteration number from filename
        filename = os.path.basename(filepath)
        iter_num = int(filename.replace("snapshot_iter", "").replace(".json", ""))
        if iter_num % frequency == 0:
            selected_files.append(filepath)
    
    return selected_files


def evaluate_gepa_results(
    results_dir: str,
    frequency: int = 1,
    select_criterion: str = "most_frequent",
    num_test_samples: int = 10,
    num_threads: int = 20,
    num_eval_times: int = 10,
    model: str = 'gemini-2.0-flash',
    user_model: str = 'gemini-2.0-flash'
):
    """
    Evaluate GEPA results from the specified directory.
    
    Args:
        results_dir: Path to the GEPA results directory (e.g., 'dspy_results/gepa_Nov25')
        frequency: Load every Nth snapshot
        select_criterion: Program selection method ("most_frequent" or "sample_by_freq")
        num_test_samples: Number of test samples per evaluation
        num_threads: Number of threads for parallel processing
        num_eval_times: Number of evaluation runs per step
        model: Model to use for the agent
        user_model: Model to use for the user
    """
    # Set up paths
    pareto_dir = os.path.join(results_dir, "pareto_snapshots")
    eval_results_dir = os.path.join(results_dir, "eval_results")
    scores_path = os.path.join(eval_results_dir, "scores.json")
    eval_results_path = os.path.join(eval_results_dir, f"eval_results_{select_criterion}.json")
    
    # Load or create cache files
    if os.path.exists(scores_path):
        scores_cache = load_json(scores_path)
        print(f"Loaded {len(scores_cache)} cached scores from {scores_path}")
    else:
        scores_cache = {}
        print("Starting with empty scores cache")
    
    if os.path.exists(eval_results_path):
        eval_results = load_json(eval_results_path)
        print(f"Loaded {len(eval_results)} existing eval results from {eval_results_path}")
        # Create set of already evaluated iterations for quick lookup
        evaluated_iters = {r["iteration"] for r in eval_results}
    else:
        eval_results = []
        evaluated_iters = set()
        print("Starting with empty eval results")
    
    # Get snapshot files to process
    snapshot_files = get_snapshot_files(pareto_dir, frequency)
    print(f"Found {len(snapshot_files)} snapshots to process (frequency={frequency}, criterion={select_criterion})")
    
    # Initial evaluation with prog_0 (total_samples=0)
    if -1 not in evaluated_iters:
        # Load first snapshot to get prog_0
        first_snapshot_path = os.path.join(pareto_dir, "snapshot_iter0000.json")
        first_snapshot = load_json(first_snapshot_path)
        program_library = first_snapshot["program_library"]
        
        prog_idx_str = "0"
        if prog_idx_str not in program_library:
            raise ValueError("Initial program (prog_0) not found in program_library")
        
        program_str = program_library[prog_idx_str]["prog"]
        
        # Check cache for score
        if prog_idx_str in scores_cache:
            score = scores_cache[prog_idx_str]
            print(f"[Initial] total_samples=0, prog_idx=0: cached score = {score:.4f}")
        else:
            print(f"[Initial] total_samples=0, prog_idx=0: evaluating...")
            score = evaluate_agent_from_str(
                instruction_str=program_str,
                num_test_samples=num_test_samples,
                num_threads=num_threads,
                num_eval_times=num_eval_times,
                model=model,
                user_model=user_model
            )
            print(f"    Score: {score:.4f}")
            scores_cache[prog_idx_str] = score
        
        # Add initial result
        result = {
            "iteration": -1,
            "total_samples": 0,
            "program_index": 0,
            "score": score
        }
        eval_results.append(result)
        evaluated_iters.add(-1)
        
        # Save caches
        save_json(scores_path, scores_cache)
        save_json(eval_results_path, eval_results)
    else:
        print("[Initial] total_samples=0: already evaluated, skipping")
    
    # Process each snapshot
    for i, snapshot_path in enumerate(snapshot_files):
        snapshot = load_json(snapshot_path)
        iteration = snapshot["iteration"]
        total_samples = snapshot["total_samples"]
        
        # Skip if already evaluated
        if iteration in evaluated_iters:
            print(f"[{i+1}/{len(snapshot_files)}] Iteration {iteration}: already evaluated, skipping")
            continue
        
        # Select program based on criterion
        pareto_front_by_task = snapshot["pareto_front_by_task"]
        program_library = snapshot["program_library"]
        
        selected_prog_idx = select_program(pareto_front_by_task, criterion=select_criterion)
        if selected_prog_idx is None:
            raise ValueError(f"Iteration {iteration}: no programs found in pareto_front_by_task")
        
        # Get program string
        prog_idx_str = str(selected_prog_idx)
        if prog_idx_str not in program_library:
            raise ValueError(f"Iteration {iteration}: program {selected_prog_idx} not found in program_library")
        
        program_str = program_library[prog_idx_str]["prog"]
        
        # Check cache for score
        if prog_idx_str in scores_cache:
            score = scores_cache[prog_idx_str]
            print(f"[{i+1}/{len(snapshot_files)}] Iteration {iteration}, total_samples={total_samples}, "
                  f"prog_idx={selected_prog_idx}: cached score = {score:.4f}")
        else:
            # Evaluate the program
            print(f"[{i+1}/{len(snapshot_files)}] Iteration {iteration}, total_samples={total_samples}, "
                  f"prog_idx={selected_prog_idx}: evaluating...")
            
            score = evaluate_agent_from_str(
                instruction_str=program_str,
                num_test_samples=num_test_samples,
                num_threads=num_threads,
                num_eval_times=num_eval_times,
                model=model,
                user_model=user_model
            )
            
            print(f"    Score: {score:.4f}")
            
            # Update scores cache with newly evaluated score
            scores_cache[prog_idx_str] = score
        
        # Add to eval results
        result = {
            "iteration": iteration,
            "total_samples": total_samples,
            "program_index": selected_prog_idx,
            "score": score
        }
        eval_results.append(result)
        evaluated_iters.add(iteration)
        
        # Save both caches after each iteration for resume capability
        # (in case the code is interrupted, we still get partial results)
        save_json(scores_path, scores_cache)
        save_json(eval_results_path, eval_results)
    
    # Sort eval_results by iteration before final save
    eval_results = sorted(eval_results, key=lambda x: x["iteration"])
    save_json(eval_results_path, eval_results)
    
    print(f"\nEvaluation complete!")
    print(f"  Scores cache: {scores_path} ({len(scores_cache)} programs)")
    print(f"  Eval results: {eval_results_path} ({len(eval_results)} iterations)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate GEPA Results')
    
    parser.add_argument('--dir', type=str, required=True,
                       help='Path to GEPA results directory (e.g., dspy_results/gepa_Nov25)')
    parser.add_argument('--frequency', type=int, default=30,
                       help='Load every Nth snapshot (default: 30)')
    parser.add_argument('--select_criterion', type=str, default='most_frequent',
                       choices=['most_frequent', 'sample_by_freq'],
                       help='Program selection criterion: most_frequent or sample_by_freq')
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples per evaluation')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--num_eval_times', type=int, default=10,
                       help='Number of evaluation runs per step')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    
    args = parser.parse_args()
    
    evaluate_gepa_results(
        results_dir=args.dir,
        frequency=args.frequency,
        select_criterion=args.select_criterion,
        num_test_samples=args.num_test_samples,
        num_threads=args.num_threads,
        num_eval_times=args.num_eval_times,
        model=args.model,
        user_model=args.user_model
    )
