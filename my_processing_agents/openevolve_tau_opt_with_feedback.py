#!/usr/bin/env python3
"""
OpenEvolve-based optimization for tau-agent with RICH FEEDBACK.

This script is a wrapper around openevolve_tau_opt.py that automatically
uses the feedback-enabled evaluator and config.

Key differences from basic version:
- Uses evaluator_with_feedback.py (provides per-task feedback)
- Uses config_with_feedback.yaml (enables artifacts)
- Provides detailed failure analysis to guide evolution
"""

import os
import sys
import argparse
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
        import json
        with open(counter_file, 'w') as f:
            json.dump({'cumulative_samples': 0}, f)
    
    return output_path


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
    
    # IMPORTANT: Ensure artifacts are enabled
    if 'prompt' not in config:
        config['prompt'] = {}
    config['prompt']['include_artifacts'] = True
    config['prompt']['max_artifact_bytes'] = args.max_artifact_bytes
    
    return config


def main():
    """Main optimization function with feedback."""
    parser = argparse.ArgumentParser(
        description='Optimize tau-agent with OpenEvolve + Rich Feedback',
        epilog='This version provides per-task feedback to guide evolution more effectively.'
    )
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                       help='Number of training samples (tasks) to evaluate on')
    
    # Optimization parameters
    parser.add_argument('--max_iterations', type=int, default=200,
                       help='Maximum number of optimization iterations')
    parser.add_argument('--parallel_evaluations', type=int, default=10,
                       help='Number of parallel evaluations within iteration (default: 4)')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker processes for parallel iterations (default: 1)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='LLM model to use for evolution')
    
    # Feedback parameters
    parser.add_argument('--max_feedback_tasks', type=int, default=3,
                       help='Number of worst tasks to show detailed feedback for (default: 3)')
    parser.add_argument('--max_conversation_length', type=int, default=2000,
                       help='Maximum conversation length in feedback (chars, default: 2000)')
    parser.add_argument('--max_artifact_bytes', type=int, default=20480,
                       help='Maximum artifact size in bytes (default: 20KB)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, 
                       default='results/openevolve_feedback',
                       help='Output directory for results')
    parser.add_argument('--project_name', type=str, 
                       default='tau-bench-feedback',
                       help='Project name for logging')
    parser.add_argument('--run_name', type=str, 
                       default=None,
                       help='Run name for logging (default: timestamp)')
    
    # Config file (defaults to feedback config)
    parser.add_argument('--config', type=str, 
                       default=None,
                       help='Path to custom config file (default: config_with_feedback.yaml)')
    
    args = parser.parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup paths
    script_dir = Path(__file__).parent
    openevolve_dir = script_dir / "openevolve_tau_opt"
    
    initial_program_path = openevolve_dir / "initial_program.py"
    
    # USE FEEDBACK EVALUATOR
    evaluator_path = openevolve_dir / "evaluator_with_feedback.py"
    
    # USE FEEDBACK CONFIG by default
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = openevolve_dir / "config_with_feedback.yaml"
    
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
    print("OpenEvolve Tau Agent Optimization WITH RICH FEEDBACK")
    print("=" * 80)
    print(f"Project: {args.project_name}")
    print(f"Run: {args.run_name}")
    print(f"Training samples: {args.num_train_samples}")
    print(f"Max iterations: {config['max_iterations']}")
    print(f"Parallel evaluations: {config['evaluator']['parallel_evaluations']}")
    print(f"Number of workers: {config.get('num_workers', 1)}")
    print(f"Model: {config['llm']['models'][0]['name']}")
    print(f"Output directory: {output_dir}")
    print()
    print("FEEDBACK FEATURES ENABLED:")
    print(f"  - Per-task success/failure tracking")
    print(f"  - Detailed feedback for worst {args.max_feedback_tasks} tasks")
    print(f"  - Conversation history (up to {args.max_conversation_length} chars)")
    print(f"  - Failure pattern analysis")
    print(f"  - Actionable improvement hints")
    print(f"  - Artifact size limit: {args.max_artifact_bytes} bytes")
    print()
    print(f"Initial program: {initial_program_path}")
    print(f"Evaluator: {evaluator_path} (WITH FEEDBACK)")
    print(f"Config: {config_path}")
    print("=" * 80)
    
    # Set environment variables for evaluator
    os.environ["NUM_TRAIN_SAMPLES"] = str(args.num_train_samples)
    os.environ["TAU_MODEL"] = args.model
    os.environ["SAMPLES_COUNTER_FILE"] = str(output_dir / "samples_counter.json")
    os.environ["PARALLEL_EVALUATIONS"] = str(config['evaluator']['parallel_evaluations'])
    
    # Feedback-specific environment variables
    os.environ["MAX_FEEDBACK_TASKS"] = str(args.max_feedback_tasks)
    os.environ["MAX_CONVERSATION_LENGTH"] = str(args.max_conversation_length)
    
    # Save run metadata
    import json
    metadata = {
        "project_name": args.project_name,
        "run_name": args.run_name,
        "num_train_samples": args.num_train_samples,
        "max_iterations": config['max_iterations'],
        "parallel_evaluations": config['evaluator']['parallel_evaluations'],
        "num_workers": config.get('num_workers', 1),
        "model": config['llm']['models'][0]['name'],
        "feedback_enabled": True,
        "max_feedback_tasks": args.max_feedback_tasks,
        "max_conversation_length": args.max_conversation_length,
        "max_artifact_bytes": args.max_artifact_bytes,
        "start_time": datetime.now().isoformat(),
        "output_dir": str(output_dir),
    }
    
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Run evolution
    print("\nStarting OpenEvolve optimization with rich feedback...\n")
    
    try:
        # Pass config file path directly, OpenEvolve will load it
        result: EvolutionResult = run_evolution(
            initial_program=str(initial_program_path),
            evaluator=str(evaluator_path),
            config=str(effective_config_path),
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
            "metrics": result.metrics if hasattr(result, 'metrics') else {},
            "feedback_enabled": True,
        }
        
        with open(output_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Save best program to a separate file for easy access
        with open(output_dir / "best_program.py", 'w') as f:
            f.write(result.best_code)
        
        print(f"\nTotal cumulative samples used: {final_samples}")
        print(f"Results saved to: {output_dir}")
        print("\nTIP: Review the evolution trace to see how feedback guided improvements:")
        print(f"  cat {output_dir}/evolution_trace.jsonl | jq '.artifacts'")
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
