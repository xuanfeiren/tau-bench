

# Run 6 separate optimizations
for run_num in {1..6}; do
    echo ""
    echo "=========================================="
    echo "Starting Run $run_num of 6"
    echo "=========================================="
    echo ""
    
    run_name="run_200iterations_${run_num}"
    
    python my_processing_agents/openevolve_tau_opt.py \
        --num_train_samples 10 \
        --max_iterations 200 \
        --parallel_evaluations 10 \
        --num_workers 1 \
        --model gemini-2.0-flash \
        --config my_processing_agents/openevolve_tau_opt/config_200iterations.yaml \
        --output_dir results/openevolve_new \
        --run_name "$run_name" \
        --project_name tau-bench-200iterations-6runs
    
    echo ""
    echo "=========================================="
    echo "Run $run_num complete!"
    echo "Results saved to: results/openevolve_new/$run_name/"
    echo "=========================================="
    echo ""
    
    # Small delay between runs to avoid any potential conflicts
    sleep 2
done

python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 200 \
    --parallel_evaluations 10 \
    --num_workers 1 \
    --model gemini-2.0-flash \
    --config my_processing_agents/openevolve_tau_opt/config_200iterations.yaml \
    --output_dir results/openevolve_new \
    --run_name run_200iterations_run_2 \
    --project_name tau-bench-200iterations-6runs

