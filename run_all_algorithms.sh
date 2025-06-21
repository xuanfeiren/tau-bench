#!/bin/bash

# Script to run tau_trainer.py with all different algorithm types
# All other parameters will use their default values

echo "Starting training with all algorithm types..."
echo "=========================================="

# Array of all algorithm types
algorithms=("minibatch" "basicsearch" "beamsearch" "beamsearchhistory" "UCBsearch")

# Loop through each algorithm type
for algo in "${algorithms[@]}"; do
    echo ""
    echo "Running algorithm: $algo"
    echo "----------------------------------------"
    
    # Run the training script with the current algorithm
    python tau_trainer.py --algorithm_type $algo
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Algorithm $algo completed successfully"
    else
        echo "❌ Algorithm $algo failed"
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "All algorithms have been executed!"
echo "==========================================" 