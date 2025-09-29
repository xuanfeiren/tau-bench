#!/bin/bash

# Debug for memory issues

python my_processing_agents/dummy_optimize_tau_agent.py \
        --num_train_samples 50 \
        --num_validate_samples 50 \
        --num_test_samples 50 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 50 \
        --num_threads 100 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 10 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "debug" \
        --run_name "dummy-PS-dummy-agent" \
        --optoprime_version v2

python my_processing_agents/dummy_optimize_tau_agent.py \
        --num_train_samples 50 \
        --num_validate_samples 50 \
        --num_test_samples 50 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 50 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "debug" \
        --run_name "dummy-PS-real-agent" \
        --optoprime_version v2
