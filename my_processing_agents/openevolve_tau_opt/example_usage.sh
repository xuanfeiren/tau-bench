#!/bin/bash
# Example usage scripts for OpenEvolve tau-agent optimization

echo "OpenEvolve Tau-Agent Optimization - Example Usage"
echo "=================================================="
echo ""

# Make sure we're in the tau-bench root directory
cd "$(dirname "$0")/../../.." || exit 1

echo "Current directory: $(pwd)"
echo ""

# Example 1: Basic usage with defaults
echo "Example 1: Basic optimization with defaults (10 tasks, 50 iterations)"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt.py"
echo ""

# Example 2: Quick test with minimal parameters
echo "Example 2: Quick test (5 tasks, 10 iterations)"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt.py \\"
echo "    --num_train_samples 5 \\"
echo "    --max_iterations 10 \\"
echo "    --run_name quick_test"
echo ""

# Example 3: Full optimization run
echo "Example 3: Full optimization (20 tasks, 100 iterations, 8 parallel)"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt.py \\"
echo "    --num_train_samples 20 \\"
echo "    --max_iterations 100 \\"
echo "    --parallel_evaluations 8 \\"
echo "    --model gemini-2.0-flash \\"
echo "    --output_dir results/openevolve_tau \\"
echo "    --run_name full_optimization_v1"
echo ""

# Example 4: Monitor progress
echo "Example 4: Monitor optimization progress"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt/monitor_progress.py \\"
echo "    results/openevolve_tau/your_run_name"
echo ""

# Example 5: Run validation test
echo "Example 5: Validate setup before running"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt/test_setup.py"
echo ""

# Example 6: Run integration smoke test
echo "Example 6: Run integration smoke test"
echo "-----------------------------------------------------------------------"
echo "python my_processing_agents/openevolve_tau_opt/test_integration.py"
echo ""

echo "=================================================="
echo "For more details, see:"
echo "  my_processing_agents/openevolve_tau_opt/README.md"
echo "=================================================="



