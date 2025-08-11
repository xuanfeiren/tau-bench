#!/bin/bash

# Monitor script for ExploreAlgorithm_LLMFA runs
LOG_FILE="monitor_explore_algorithm.log"
COMPLETED_RUNS=0
TARGET_RUNS=3

echo "$(date): Starting monitor for ExploreAlgorithm_LLMFA runs" >> $LOG_FILE

# Function to check if the process is still running
check_process() {
    # Check if there's a python process running tau_agent_opt.py with ExploreAlgorithm_LLMFA
    pgrep -f "tau_agent_opt.py.*ExploreAlgorithm_LLMFA" > /dev/null
    return $?
}

# Function to run the algorithm
run_algorithm() {
    local run_number=$1
    echo "$(date): Starting ExploreAlgorithm_LLMFA run #$run_number" >> $LOG_FILE
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm_LLMFA" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --num_phases 10 --train_batch_size 2 --num_to_sample 2 --run_name "ExploreAlgorithm_v2.0_run$run_number"
    echo "$(date): Completed ExploreAlgorithm_LLMFA run #$run_number" >> $LOG_FILE
}

# Wait for current run to finish
echo "$(date): Waiting for current ExploreAlgorithm_LLMFA run to finish..." >> $LOG_FILE
while check_process; do
    echo "$(date): Current run still in progress..." >> $LOG_FILE
    sleep 300  # Wait 5 minutes
done

echo "$(date): Current run finished. Starting additional runs..." >> $LOG_FILE

# Run 3 additional runs
for i in $(seq 1 $TARGET_RUNS); do
    echo "$(date): Starting additional run $i of $TARGET_RUNS" >> $LOG_FILE
    run_algorithm $i
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
done

echo "$(date): All $COMPLETED_RUNS additional runs completed!" >> $LOG_FILE
echo "Monitor script finished. Check $LOG_FILE for details." 