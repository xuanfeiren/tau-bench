uv run python my_processing_agents/dspy_opt.py \
    --num_samples 10 \
    --model gemini-2.5-flash-lite \
    --max_metric_calls 2000 \
    --num_threads 20 \
    --log_frequency 2 \
    --log_dir "dspy_results/gepa_run"