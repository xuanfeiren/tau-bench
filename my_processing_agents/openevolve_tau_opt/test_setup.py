"""
Test script to validate the OpenEvolve tau-agent optimization setup.

This script performs basic validation checks without running a full optimization.
"""

import sys
from pathlib import Path

# Add tau-bench to path
tau_bench_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tau_bench_root))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import openevolve
        print("  ✓ OpenEvolve imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import OpenEvolve: {e}")
        return False
    
    try:
        from tau_bench.envs import get_env
        print("  ✓ Tau-bench imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import tau-bench: {e}")
        return False
    
    try:
        from agents.tool_calling_agent import ToolCallingAgent_v2
        print("  ✓ Agent imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import agent: {e}")
        return False
    
    return True

def test_files_exist():
    """Test that all required files exist."""
    print("\nChecking required files...")
    
    base_dir = Path(__file__).parent
    required_files = [
        'initial_program.py',
        'evaluator.py',
        'config.yaml',
        'monitor_progress.py',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            all_exist = False
    
    return all_exist

def test_initial_program():
    """Test that initial_program.py is valid."""
    print("\nValidating initial_program.py...")
    
    try:
        base_dir = Path(__file__).parent
        program_path = base_dir / 'initial_program.py'
        
        with open(program_path, 'r') as f:
            content = f.read()
        
        # Check for EVOLVE-BLOCK markers
        if 'EVOLVE-BLOCK-START' not in content:
            print("  ✗ Missing EVOLVE-BLOCK-START marker")
            return False
        
        if 'EVOLVE-BLOCK-END' not in content:
            print("  ✗ Missing EVOLVE-BLOCK-END marker")
            return False
        
        # Try to execute and extract variable
        namespace = {}
        exec(content, namespace)
        
        if 'additional_instructions' not in namespace:
            print("  ✗ No additional_instructions variable found")
            return False
        
        instructions = namespace['additional_instructions']
        print(f"  ✓ Found additional_instructions ({len(instructions)} chars)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error validating initial_program.py: {e}")
        return False

def test_evaluator():
    """Test that evaluator.py is importable and has required functions."""
    print("\nValidating evaluator.py...")
    
    try:
        base_dir = Path(__file__).parent
        sys.path.insert(0, str(base_dir))
        
        # Try importing the evaluator module
        import evaluator
        
        # Check for required functions
        required_functions = ['evaluate', 'evaluate_stage1', 'evaluate_stage2']
        all_present = True
        
        for func_name in required_functions:
            if hasattr(evaluator, func_name):
                print(f"  ✓ {func_name} function found")
            else:
                print(f"  ✗ {func_name} function not found")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ✗ Error importing evaluator: {e}")
        return False

def test_config():
    """Test that config.yaml is valid."""
    print("\nValidating config.yaml...")
    
    try:
        import yaml
        
        base_dir = Path(__file__).parent
        config_path = base_dir / 'config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required fields
        required_fields = ['max_iterations', 'llm', 'database', 'evaluator']
        all_present = True
        
        for field in required_fields:
            if field in config:
                print(f"  ✓ {field} configured")
            else:
                print(f"  ✗ {field} not found in config")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"  ✗ Error parsing config.yaml: {e}")
        return False

def test_environment_setup():
    """Test that environment is properly configured."""
    print("\nChecking environment...")
    
    import os
    
    # Check for API key
    if 'GEMINI_API_KEY' in os.environ:
        print("  ✓ GEMINI_API_KEY is set")
        has_key = True
    else:
        print("  ⚠ GEMINI_API_KEY not set (required for actual runs)")
        has_key = False
    
    return has_key

def main():
    """Run all validation tests."""
    print("=" * 80)
    print("OpenEvolve Tau-Agent Optimization Setup Validation")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Files", test_files_exist),
        ("Initial Program", test_initial_program),
        ("Evaluator", test_evaluator),
        ("Config", test_config),
        ("Environment", test_environment_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nUnexpected error in {test_name} test: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Setup is ready for optimization.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



