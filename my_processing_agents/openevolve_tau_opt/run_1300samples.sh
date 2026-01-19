#!/bin/bash
# Run OpenEvolve optimization with exactly 1300 samples
# Configuration: 10 tasks, 129 iterations, 20 parallel threads
# Expected: 130 evaluations × 10 tasks = 1300 samples

set -e  # Exit on error

# Navigate to tau-bench root
cd "$(dirname "$0")/../../.." || exit 1

echo "=========================================="
echo "OpenEvolve Tau-Agent Optimization"
echo "Target: 1300 samples (10 tasks)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Training tasks: 10"
echo "  - Validation tasks: 10 (same dataset)"
echo "  - Test tasks: 10 (same dataset)"
echo "  - Max iterations: 129"
echo "  - Parallel evaluations: 20"
echo "  - Expected evaluations: 130 (1 initial + 129)"
echo "  - Expected samples: 1300 (130 × 10)"
echo ""
echo "Starting optimization..."
echo ""

# Run with custom config
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 129 \
    --parallel_evaluations 20 \
    --model gemini-2.0-flash \
    --config my_processing_agents/openevolve_tau_opt/config_1300samples.yaml \
    --output_dir results/openevolve_tau \
    --run_name run_1300samples \
    --project_name tau-bench-1300samples

echo ""
echo "=========================================="
echo "Optimization complete!"
echo "Check results in: results/openevolve_tau/run_1300samples/"
echo "=========================================="



