#!/bin/bash
# Run OpenEvolve optimization with 200 iterations
# Configuration: 10 tasks, 200 iterations, 10 parallel evaluations per iteration, 1 worker (sequential iterations)
# Expected: 201 evaluations × 10 tasks = 2010 samples (1 initial + 200 iterations)

set -e  # Exit on error

# Navigate to tau-bench root
cd "$(dirname "$0")/../../.." || exit 1

echo "=========================================="
echo "OpenEvolve Tau-Agent Optimization"
echo "200 Iterations Configuration"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Training tasks: 10"
echo "  - Max iterations: 200"
echo "  - Parallel evaluations (within iteration): 10"
echo "  - Number of workers (across iterations): 1 (sequential)"
echo "  - Expected evaluations: 201 (1 initial + 200 iterations)"
echo "  - Expected samples: 2010 (201 × 10 tasks)"
echo ""
echo "Starting optimization..."
echo ""

# Run with custom config
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 200 \
    --parallel_evaluations 10 \
    --num_workers 1 \
    --model gemini-2.0-flash \
    --config my_processing_agents/openevolve_tau_opt/config_200iterations.yaml \
    --output_dir results/openevolve_tau \
    --run_name run_200iterations \
    --project_name tau-bench-200iterations

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results in: results/openevolve_tau/run_200iterations/"
echo "=========================================="
