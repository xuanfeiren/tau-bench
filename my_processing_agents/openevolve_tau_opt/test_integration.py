"""
Integration smoke test for OpenEvolve tau-agent optimization.

This runs a minimal optimization (2 iterations, 2 tasks) to verify the complete pipeline works.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add tau-bench to path
tau_bench_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tau_bench_root))

def run_smoke_test():
    """Run a minimal optimization to test the pipeline."""
    print("=" * 80)
    print("OpenEvolve Tau-Agent Integration Smoke Test")
    print("=" * 80)
    print("\nThis will run a minimal optimization (2 iterations, 2 tasks)")
    print("to verify the complete pipeline works.\n")
    
    # Create a temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="openevolve_test_")
    print(f"Using temporary directory: {temp_dir}\n")
    
    try:
        # Set minimal parameters
        os.environ["NUM_TRAIN_SAMPLES"] = "2"
        os.environ["TAU_MODEL"] = "gemini-2.0-flash"
        os.environ["SAMPLES_COUNTER_FILE"] = str(Path(temp_dir) / "samples_counter.json")
        
        # Import after setting env vars
        from openevolve import run_evolution
        from openevolve.config import Config
        import yaml
        
        # Load base config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Override for minimal test
        config_dict['max_iterations'] = 2
        config_dict['evaluator']['parallel_evaluations'] = 1
        config_dict['database']['population_size'] = 5
        config_dict['database']['num_islands'] = 1
        
        # Create config object
        config = Config(**config_dict)
        
        # Setup paths
        base_dir = Path(__file__).parent
        initial_program = base_dir / "initial_program.py"
        evaluator = base_dir / "evaluator.py"
        
        print("Starting minimal optimization...")
        print(f"  Iterations: 2")
        print(f"  Tasks per eval: 2")
        print(f"  Expected samples: ~8-16 (depends on OpenEvolve's evaluation strategy)")
        print()
        
        # Run evolution
        result = run_evolution(
            initial_program=str(initial_program),
            evaluator=str(evaluator),
            config=config,
            output_dir=temp_dir,
            cleanup=False
        )
        
        print("\n" + "=" * 80)
        print("Smoke Test Results")
        print("=" * 80)
        print(f"Best score: {result.best_score:.4f}")
        print(f"Output directory: {temp_dir}")
        
        # Check that key files were created
        expected_files = [
            'samples_counter.json',
        ]
        
        print("\nChecking output files:")
        for filename in expected_files:
            filepath = Path(temp_dir) / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} missing")
        
        # Read and display sample count
        import json
        counter_file = Path(temp_dir) / "samples_counter.json"
        if counter_file.exists():
            with open(counter_file, 'r') as f:
                data = json.load(f)
                samples = data.get('cumulative_samples', 0)
                print(f"\nTotal samples used: {samples}")
        
        print("\n✓ Integration smoke test completed successfully!")
        print(f"\nTest output saved to: {temp_dir}")
        print("You can inspect the results or delete this directory.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Keep temp dir for debugging
        print(f"\nTest output saved to: {temp_dir}")
        print("Please inspect for debugging.")
        
        return False

def main():
    """Main test function."""
    success = run_smoke_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())



