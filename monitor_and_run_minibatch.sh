#!/bin/bash

# Script to monitor current job and run MinibatchwithValidation twice after completion
# Checks every 30 minutes for job completion

LOG_FILE="monitor_minibatch.log"
CHECK_INTERVAL=1800  # 30 minutes in seconds

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if any Python processes are running tau_agent_opt.py or learn_from_success.py
is_job_running() {
    # Check for tau_agent_opt.py processes
    tau_processes=$(pgrep -f "python.*tau_agent_opt.py" 2>/dev/null | wc -l)
    
    # Check for learn_from_success.py processes
    learn_processes=$(pgrep -f "python.*learn_from_success.py" 2>/dev/null | wc -l)
    
    # Check for any Python training processes
    training_processes=$(pgrep -f "python.*training" 2>/dev/null | wc -l)
    
    total_processes=$((tau_processes + learn_processes + training_processes))
    
    if [ $total_processes -gt 0 ]; then
        return 0  # Job is running
    else
        return 1  # No job running
    fi
}

# Function to run MinibatchwithValidation
run_minibatch() {
    local run_number=$1
    log_message "Starting MinibatchwithValidation run #$run_number"
    
    python tau_agent_opt.py \
        --algorithm_name "MinibatchwithValidation" \
        --eval_frequency 2 \
        --log_frequency 1 \
        --num_proposals 2 \
        --num_epochs 20 \
        --run_name "MinibatchwithValidation_auto_run_$run_number" \
        2>&1 | tee -a "minibatch_run_${run_number}.log"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_message "MinibatchwithValidation run #$run_number completed successfully"
    else
        log_message "MinibatchwithValidation run #$run_number failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Main monitoring loop
log_message "Starting job monitoring script"
log_message "Will check every $((CHECK_INTERVAL / 60)) minutes for job completion"

while true; do
    if is_job_running; then
        log_message "Job is still running, checking again in $((CHECK_INTERVAL / 60)) minutes..."
        sleep $CHECK_INTERVAL
    else
        log_message "No running jobs detected. Starting MinibatchwithValidation runs..."
        break
    fi
done

# Wait a bit to ensure the previous job has fully completed
log_message "Waiting 2 minutes to ensure previous job has fully completed..."
sleep 120

# Run MinibatchwithValidation twice
for run_num in 1 2; do
    log_message "=" | tr ' ' '=' | head -c 50
    echo ""
    
    run_minibatch $run_num
    
    if [ $? -ne 0 ]; then
        log_message "Run #$run_num failed, but continuing with next run..."
    fi
    
    # Wait between runs to avoid resource conflicts
    if [ $run_num -eq 1 ]; then
        log_message "Waiting 5 minutes before starting run #2..."
        sleep 300
    fi
done

log_message "All MinibatchwithValidation runs completed"
log_message "Check individual run logs: minibatch_run_1.log and minibatch_run_2.log"
log_message "Monitor log: $LOG_FILE"

