#!/bin/bash
# Run multiple independent 1300-sample experiments
# Each gets its own directory and logs

# Configuration
NUM_RUNS=3  # Change this to run more experiments
NUM_TRAIN_SAMPLES=10
MAX_ITERATIONS=129
PARALLEL_EVALUATIONS=20
MODEL="gemini-2.0-flash"
OUTPUT_DIR="results/openevolve_tau"

# Navigate to tau-bench root
cd "$(dirname "$0")/../../.." || exit 1
echo "Working directory: $(pwd)"

# Activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tau

echo "=========================================="
echo "Running $NUM_RUNS Independent Experiments"
echo "Each with 1300 samples (130 eval × 10 tasks)"
echo "=========================================="
echo ""

# Run multiple experiments
for i in $(seq 1 $NUM_RUNS); do
    RUN_NAME="run${i}_1300samples"
    
    echo "=========================================="
    echo "Starting Experiment $i/$NUM_RUNS"
    echo "Run name: $RUN_NAME"
    echo "Start time: $(date)"
    echo "=========================================="
    
    python my_processing_agents/openevolve_tau_opt.py \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --max_iterations $MAX_ITERATIONS \
        --parallel_evaluations $PARALLEL_EVALUATIONS \
        --model $MODEL \
        --output_dir $OUTPUT_DIR \
        --run_name $RUN_NAME \
        --project_name "tau-bench-1300samples" \
        2>&1 | tee "logs/run${i}_$(date +%Y%m%d_%H%M%S).log"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Experiment $i completed successfully"
        echo "  Output: $OUTPUT_DIR/$RUN_NAME"
        echo "  Samples: $(cat $OUTPUT_DIR/$RUN_NAME/samples_counter.json)"
        echo ""
    else
        echo ""
        echo "✗ Experiment $i failed with exit code $EXIT_CODE"
        echo ""
    fi
done

echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo ""
echo "Summary:"
for i in $(seq 1 $NUM_RUNS); do
    RUN_NAME="run${i}_1300samples"
    if [ -f "$OUTPUT_DIR/$RUN_NAME/final_results.json" ]; then
        SCORE=$(cat $OUTPUT_DIR/$RUN_NAME/final_results.json | grep best_score | head -1 | awk '{print $2}' | tr -d ',')
        SAMPLES=$(cat $OUTPUT_DIR/$RUN_NAME/samples_counter.json | grep cumulative_samples | awk '{print $2}' | tr -d '}')
        echo "  Run $i: Score=$SCORE, Samples=$SAMPLES"
    else
        echo "  Run $i: FAILED or INCOMPLETE"
    fi
done
echo ""
echo "Results saved in: $OUTPUT_DIR/run{1..$NUM_RUNS}_1300samples/"



