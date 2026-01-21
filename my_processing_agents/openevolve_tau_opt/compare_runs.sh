#!/bin/bash
# Compare OpenEvolve runs with and without feedback

echo "=========================================="
echo "OpenEvolve Comparison: Basic vs Feedback"
echo "=========================================="
echo ""
echo "This script runs two parallel experiments:"
echo "1. Basic evaluator (aggregate scores only)"
echo "2. Feedback evaluator (per-task feedback + artifacts)"
echo ""

# Configuration
NUM_SAMPLES=${NUM_SAMPLES:-10}
MAX_ITERATIONS=${MAX_ITERATIONS:-20}
MODEL=${MODEL:-"gemini-2.0-flash"}

echo "Configuration:"
echo "  Training samples: $NUM_SAMPLES"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Model: $MODEL"
echo ""
echo "=========================================="
echo ""

# Run basic version (no feedback)
echo "[1/2] Running BASIC evaluator (no feedback)..."
echo ""
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples $NUM_SAMPLES \
    --max_iterations $MAX_ITERATIONS \
    --model $MODEL \
    --parallel_evaluations 4 \
    --output_dir results/comparison/basic_no_feedback \
    --run_name "basic_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee results/comparison/basic_run.log

BASIC_EXIT_CODE=$?

echo ""
echo "=========================================="
echo ""

# Run feedback version
echo "[2/2] Running FEEDBACK evaluator (with artifacts)..."
echo ""
python my_processing_agents/openevolve_tau_opt_with_feedback.py \
    --num_train_samples $NUM_SAMPLES \
    --max_iterations $MAX_ITERATIONS \
    --model $MODEL \
    --parallel_evaluations 4 \
    --max_feedback_tasks 3 \
    --output_dir results/comparison/feedback_enabled \
    --run_name "feedback_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee results/comparison/feedback_run.log

FEEDBACK_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo ""

# Show results
if [ $BASIC_EXIT_CODE -eq 0 ] && [ $FEEDBACK_EXIT_CODE -eq 0 ]; then
    echo "✓ Both runs completed successfully"
    echo ""
    
    # Find the most recent run directories
    BASIC_DIR=$(ls -td results/comparison/basic_no_feedback/*/ 2>/dev/null | head -1)
    FEEDBACK_DIR=$(ls -td results/comparison/feedback_enabled/*/ 2>/dev/null | head -1)
    
    if [ -n "$BASIC_DIR" ] && [ -n "$FEEDBACK_DIR" ]; then
        echo "Results locations:"
        echo "  Basic:    $BASIC_DIR"
        echo "  Feedback: $FEEDBACK_DIR"
        echo ""
        
        # Compare final scores
        if command -v jq &> /dev/null; then
            echo "Final Scores:"
            BASIC_SCORE=$(jq -r '.best_score // "N/A"' "${BASIC_DIR}final_results.json" 2>/dev/null)
            FEEDBACK_SCORE=$(jq -r '.best_score // "N/A"' "${FEEDBACK_DIR}final_results.json" 2>/dev/null)
            
            echo "  Basic evaluator:    $BASIC_SCORE"
            echo "  Feedback evaluator: $FEEDBACK_SCORE"
            echo ""
            
            # Show sample artifacts (if available)
            echo "Sample artifacts from feedback run:"
            echo "---"
            jq -r '.artifacts.performance_hint // "No artifacts"' \
                "${FEEDBACK_DIR}final_results.json" 2>/dev/null | head -3
            echo "---"
            echo ""
        fi
        
        echo "To view full results:"
        echo "  cat ${BASIC_DIR}final_results.json"
        echo "  cat ${FEEDBACK_DIR}final_results.json"
        echo ""
        echo "To see evolution trace with artifacts:"
        echo "  cat ${FEEDBACK_DIR}evolution_trace.jsonl | jq '.artifacts'"
        echo ""
        echo "To compare best programs:"
        echo "  diff ${BASIC_DIR}best_program.py ${FEEDBACK_DIR}best_program.py"
    fi
else
    echo "✗ One or more runs failed"
    echo "  Basic exit code: $BASIC_EXIT_CODE"
    echo "  Feedback exit code: $FEEDBACK_EXIT_CODE"
    echo ""
    echo "Check logs for details:"
    echo "  results/comparison/basic_run.log"
    echo "  results/comparison/feedback_run.log"
fi

echo ""
echo "=========================================="
