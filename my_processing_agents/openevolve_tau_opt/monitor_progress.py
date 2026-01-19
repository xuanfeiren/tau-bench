"""
Monitor and log OpenEvolve optimization progress.

This script parses OpenEvolve's evolution trace and database to extract
iteration-level statistics including cumulative samples and best scores.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import sqlite3

def parse_evolution_trace(trace_path: Path) -> List[Dict[str, Any]]:
    """Parse the JSONL evolution trace file."""
    iterations = []
    
    if not trace_path.exists():
        return iterations
    
    with open(trace_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    iterations.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return iterations

def parse_samples_counter(counter_path: Path) -> int:
    """Parse the cumulative samples counter."""
    if not counter_path.exists():
        return 0
    
    try:
        with open(counter_path, 'r') as f:
            data = json.load(f)
            return data.get('cumulative_samples', 0)
    except Exception:
        return 0

def query_database(db_path: Path) -> List[Dict[str, Any]]:
    """Query the OpenEvolve database for program statistics."""
    if not db_path.exists():
        return []
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get all programs with their scores
        cursor.execute("""
            SELECT id, iteration, fitness, code, features, metadata
            FROM programs
            ORDER BY iteration, fitness DESC
        """)
        
        programs = []
        for row in cursor.fetchall():
            programs.append({
                'id': row[0],
                'iteration': row[1],
                'fitness': row[2],
                'code': row[3],
                'features': row[4],
                'metadata': row[5]
            })
        
        conn.close()
        return programs
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def summarize_progress(output_dir: Path) -> Dict[str, Any]:
    """
    Summarize the optimization progress from all available sources.
    
    Returns a dictionary with iteration-level statistics.
    """
    summary = {
        'iterations': [],
        'total_samples': 0,
        'best_score': 0.0,
        'best_iteration': 0
    }
    
    # Parse evolution trace
    trace_path = output_dir / "evolution_trace.jsonl"
    if trace_path.exists():
        iterations = parse_evolution_trace(trace_path)
        summary['num_iterations'] = len(iterations)
    
    # Parse samples counter
    counter_path = output_dir / "samples_counter.json"
    summary['total_samples'] = parse_samples_counter(counter_path)
    
    # Parse database if available
    db_path = output_dir / "openevolve.db"
    programs = query_database(db_path)
    
    if programs:
        # Group by iteration
        iteration_stats = {}
        for prog in programs:
            iter_num = prog['iteration']
            if iter_num not in iteration_stats:
                iteration_stats[iter_num] = {
                    'iteration': iter_num,
                    'best_score': prog['fitness'],
                    'num_programs': 0
                }
            
            iteration_stats[iter_num]['num_programs'] += 1
            iteration_stats[iter_num]['best_score'] = max(
                iteration_stats[iter_num]['best_score'],
                prog['fitness']
            )
        
        summary['iterations'] = sorted(iteration_stats.values(), 
                                      key=lambda x: x['iteration'])
        
        # Find overall best
        best = max(programs, key=lambda x: x['fitness'])
        summary['best_score'] = best['fitness']
        summary['best_iteration'] = best['iteration']
    
    return summary

def generate_progress_report(output_dir: Path, report_path: Path = None):
    """Generate a comprehensive progress report."""
    summary = summarize_progress(output_dir)
    
    if report_path is None:
        report_path = output_dir / "progress_report.json"
    
    # Save JSON report
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print human-readable report
    print("=" * 80)
    print("OpenEvolve Optimization Progress Report")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total samples used: {summary['total_samples']}")
    print(f"Best score: {summary['best_score']:.4f}")
    print(f"Best iteration: {summary['best_iteration']}")
    print(f"Number of iterations: {len(summary['iterations'])}")
    
    if summary['iterations']:
        print("\nIteration Summary:")
        print("-" * 80)
        print(f"{'Iter':<8} {'Best Score':<15} {'# Programs':<15}")
        print("-" * 80)
        for iter_stat in summary['iterations'][-10:]:  # Show last 10
            print(f"{iter_stat['iteration']:<8} "
                  f"{iter_stat['best_score']:<15.4f} "
                  f"{iter_stat['num_programs']:<15}")
    
    print("=" * 80)
    print(f"Report saved to: {report_path}")
    
    return summary

def main():
    """Main function for standalone usage."""
    if len(sys.argv) < 2:
        print("Usage: python monitor_progress.py <output_dir>")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    generate_progress_report(output_dir)

if __name__ == "__main__":
    main()



