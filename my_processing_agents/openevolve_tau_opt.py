#!/usr/bin/env python3
"""
OpenEvolve-based optimization for tau-agent's additional_instructions parameter.

This script uses OpenEvolve to evolve the additional_instructions parameter
of the ToolCallingAgent_v2, tracking cumulative training samples and logging
the best parameters at each iteration.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import yaml

# Add tau-bench to path
tau_bench_root = Path(__file__).parent.parent
sys.path.insert(0, str(tau_bench_root))

from openevolve.api import run_evolution, EvolutionResult

def setup_logging_dir(output_dir):
    """Create output directory and initialize logging files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize optimization log file
    log_file = output_path / "optimization_log.jsonl"
    if not log_file.exists():
        log_file.write_text("")
    
    # Initialize samples counter
    counter_file = output_path / "samples_counter.json"
    if not counter_file.exists():
        with open(counter_file, 'w') as f:
            json.dump({'cumulative_samples': 0}, f)
    
    return output_path

def log_iteration(output_dir, iteration, cumulative_samples, best_score, best_instructions):
    """Log iteration results to JSONL file."""
    log_file = Path(output_dir) / "optimization_log.jsonl"
    
    log_entry = {
        "iteration": iteration,
        "cumulative_samples": cumulative_samples,
        "best_score": best_score,
        "best_instructions": best_instructions,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def load_config_with_overrides(config_path, args):
    """Load config and override with command line arguments."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.max_iterations is not None:
        config['max_iterations'] = args.max_iterations
    
    if args.parallel_evaluations is not None:
        config['evaluator']['parallel_evaluations'] = args.parallel_evaluations
    
    if args.num_workers is not None:
        config['num_workers'] = args.num_workers
    
    if args.model is not None:
        config['llm']['models'] = [{"name": args.model, "weight": 1.0}]
    
    return config

def create_custom_evaluator_wrapper(output_dir):
    """
    Create a wrapper evaluator that logs iteration results.
    This allows us to track and log each evaluation with cumulative samples.
    """
    evaluator_dir = Path(__file__).parent / "openevolve_tau_opt"
    evaluator_path = evaluator_dir / "evaluator.py"
    
    # We'll use the original evaluator directly
    # The logging will be handled by monitoring the samples counter file
    return str(evaluator_path)

def monitor_and_log_progress(output_dir):
    """
    Monitor the optimization progress and log best parameters.
    This function reads the OpenEvolve database/outputs to track progress.
    """
    # This will be called periodically or at the end
    # For now, we'll rely on OpenEvolve's built-in logging
    # and our evaluator's sample tracking
    pass

def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(
        description='Optimize tau-agent additional_instructions using OpenEvolve'
    )
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                       help='Number of training samples (tasks) to evaluate on')
    
    # Optimization parameters
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of optimization iterations')
    parser.add_argument('--parallel_evaluations', type=int, default=10,
                       help='Number of parallel evaluations within a single iteration (default: 10)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker processes for parallel iterations (default: 1)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='LLM model to use for evolution')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                       default='results/openevolve_new',
                       help='Output directory for results')
    parser.add_argument('--project_name', type=str, 
                       default='tau-bench-openevolve',
                       help='Project name for logging')
    parser.add_argument('--run_name', type=str, 
                       default=None,
                       help='Run name for logging (default: timestamp)')
    
    # Config file
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='Path to custom config file (overrides default)')
    
    args = parser.parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"openevolve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup paths
    script_dir = Path(__file__).parent
    openevolve_dir = script_dir / "openevolve_tau_opt"
    
    initial_program_path = openevolve_dir / "initial_program.py"
    evaluator_path = openevolve_dir / "evaluator.py"
    
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = openevolve_dir / "config.yaml"
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.run_name
    setup_logging_dir(output_dir)
    
    # Load and override config
    config = load_config_with_overrides(config_path, args)
    
    # Save the effective config
    effective_config_path = output_dir / "effective_config.yaml"
    with open(effective_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("=" * 80)
    print("OpenEvolve Tau Agent Optimization")
    print("=" * 80)
    print(f"Project: {args.project_name}")
    print(f"Run: {args.run_name}")
    print(f"Training samples: {args.num_train_samples}")
    print(f"Max iterations: {config['max_iterations']}")
    print(f"Parallel evaluations (within iteration): {config['evaluator']['parallel_evaluations']}")
    print(f"Number of workers (across iterations): {config.get('num_workers', 1)}")
    print(f"Model: {config['llm']['models'][0]['name']}")
    print(f"Output directory: {output_dir}")
    print(f"Initial program: {initial_program_path}")
    print(f"Evaluator: {evaluator_path}")
    print(f"Config: {config_path}")
    print("=" * 80)
    
    # Set environment variables for evaluator
    os.environ["NUM_TRAIN_SAMPLES"] = str(args.num_train_samples)
    os.environ["TAU_MODEL"] = args.model
    os.environ["SAMPLES_COUNTER_FILE"] = str(output_dir / "samples_counter.json")
    os.environ["PARALLEL_EVALUATIONS"] = str(config['evaluator']['parallel_evaluations'])
    
    # Save run metadata
    metadata = {
        "project_name": args.project_name,
        "run_name": args.run_name,
        "num_train_samples": args.num_train_samples,
        "max_iterations": config['max_iterations'],
        "parallel_evaluations": config['evaluator']['parallel_evaluations'],
        "num_workers": config.get('num_workers', 1),
        "model": config['llm']['models'][0]['name'],
        "start_time": datetime.now().isoformat(),
        "output_dir": str(output_dir),
    }
    
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run evolution
    print("\nStarting OpenEvolve optimization...\n")
    
    try:
        # Pass config file path directly, OpenEvolve will load it
        result: EvolutionResult = run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(effective_config_path),  # Pass path to config file
            output_dir=str(output_dir),
            cleanup=False  # Keep all files for analysis
        )
        
        # Save results
        print("\n" + "=" * 80)
        print("Optimization Complete!")
        print("=" * 80)
        print(f"Best score: {result.best_score:.4f}")
        print(f"Best code preview:")
        print("-" * 80)
        print(result.best_code[:500])
        if len(result.best_code) > 500:
            print("... (truncated)")
        print("-" * 80)
        
        # Load final cumulative samples
        samples_counter_file = output_dir / "samples_counter.json"
        if samples_counter_file.exists():
            with open(samples_counter_file, 'r') as f:
                final_samples = json.load(f).get('cumulative_samples', 0)
        else:
            final_samples = 0
        
        # Save final results
        final_results = {
            "best_score": result.best_score,
            "best_code": result.best_code,
            "cumulative_samples": final_samples,
            "end_time": datetime.now().isoformat(),
            "metrics": result.metrics if hasattr(result, 'metrics') else {}
        }
        
        with open(output_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save best program to a separate file for easy access
        with open(output_dir / "best_program.py", 'w') as f:
            f.write(result.best_code)
        
        print(f"\nTotal cumulative samples used: {final_samples}")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        
        # Update metadata with completion info
        metadata["end_time"] = datetime.now().isoformat()
        metadata["best_score"] = result.best_score
        metadata["cumulative_samples"] = final_samples
        metadata["status"] = "completed"
        
        with open(output_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        
        # Update metadata with error info
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        
        with open(output_dir / "run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()

