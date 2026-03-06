uv run python my_processing_agents/openevolve_tau_opt_with_feedback.py \
    --num_train_samples 10 \
    --max_iterations 200 \
    --parallel_evaluations 10 \
    --num_workers 1 \
    --model "gemini-2.5-flash-lite" \
    --output_dir results/openevolve_feedback \
    --run_name run_1