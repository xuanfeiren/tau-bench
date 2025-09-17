#!/bin/bash
# run_search_algorithms.sh 
# 50 tasks may make the training too slow. So we use 10 tasks to test the embedding regressor. If it works well, we can use 50 tasks.
for i in {1..3}; do
    python my_processing_agents/tau_agent_opt.py --algorithm_name "llm_search" --eval_frequency 5 --log_frequency 1 --num_epochs 200 --train_batch_size 2  --run_name "llm-search-embedding-regressor" --num_test_samples 10 --num_train_samples 10 --num_validate_samples 10 --num_generation_steps 4 --validate_batch_size 10 --num_eval_samples 10 --select_arm_by_predicted_score --num_multiple_generations 1 
done

for i in {1..3}; do
    python my_processing_agents/tau_agent_opt.py --algorithm_name "MinibatchAlgorithm"  --eval_frequency 5 --log_frequency 1  --num_epochs 100  --num_eval_samples 10 --num_test_samples 10 --num_train_samples 10 --num_validate_samples 10 --run_name "MinibatchAlgorithm"
done

# Sep 12. On 10 tasks. Only use the training samples to update the agent. 400 training budget.
for i in {1..3}; do
    python my_processing_agents/tau_agent_opt.py --algorithm_name "llm_search" --eval_frequency 10 --log_frequency 1 --num_epochs 200 --train_batch_size 2   --num_test_samples 10 --num_train_samples 10 --num_validate_samples 10 --num_generation_steps 1  --num_eval_samples 10 --select_arm_by_predicted_score --num_multiple_generations 1 --project_name "tau-bench-10-tasks-10-evals" --run_name "llm-search-embedding-regressor"

    python my_processing_agents/tau_agent_opt.py --algorithm_name "MinibatchAlgorithm"  --eval_frequency 10 --log_frequency 1  --num_epochs 200  --train_batch_size 2 --num_eval_samples 10 --num_test_samples 10 --num_train_samples 10 --num_validate_samples 10  --project_name "tau-bench-10-tasks-10-evals" --run_name "MinibatchAlgorithm"
done
for i in {1..3}; do
    python my_processing_agents/tau_agent_opt.py --algorithm_name "llm_search" --eval_frequency 5 --log_frequency 1 --num_epochs 200 --train_batch_size 2   --num_test_samples 10 --num_train_samples 10 --num_validate_samples 10 --num_generation_steps 1  --num_eval_samples 10 --select_arm_by_predicted_score --num_multiple_generations 3 --project_name "tau-bench-10-tasks-10-evals" --run_name "llm-search-embedding-regressor"
done

# Sep 13, 2025
python my_processing_agents/optimize_tau_agent.py \
    --num_train_samples 10 \
    --num_validate_samples 10 \
    --num_test_samples 10 \
    --batch_size 2 \
    --num_batches 1 \
    --num_epochs 40 \
    --num_threads 20 \
    --test_frequency 1 \
    --log_frequency 1 \
    --num_eval_samples 10 \
    --num_candidates 10 \
    --num_proposals 1 \
    --use_best_candidate_to_explore \
    --memory_size 1000 \
    --score_function mean \
    --score_range_min 0.0 \
    --score_range_max 1.0 \
    --project_name "tau-bench-10-tasks-10-evals" \
    --run_name "priority_search" 

# Sep 14, 2025 
# A better config in my mind
# In one epoch, we use 20 training samples.
for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_test_samples 1 \
        --batch_size 2 \
        --num_batches 5 \
        --num_epochs 20 \
        --num_threads 20 \
        --short_term_memory_duration 2 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "debug" \
        --use_regressor
done

# Sep 16, 2025 morning

for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 1 \
        --num_test_samples 1 \
        --batch_size 1 \
        --num_batches 1 \
        --num_epochs 1 \
        --num_threads 20 \
        --short_term_memory_duration 0 \
        --test_frequency -1 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 3 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "debug" \
        --use_regressor
done
for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 5 \
        --num_epochs 20 \
        --num_threads 20 \
        --short_term_memory_duration 0 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 2 \
        --num_proposals 3 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "PrioritySearch" 
done

# Sep 16, 2025 how to make the search algorithm concentrate well
# Sep 16 night
for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_epochs 40 \
        --num_threads 20 \
        --short_term_memory_duration 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "PrioritySearch-with-Regressor-concentrate" \
        --use_regressor \
        --optoprime_version v2
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_epochs 40 \
        --num_threads 20 \
        --short_term_memory_duration 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "PrioritySearch-concentrate" \
        --optoprime_version v2
done

# debug for the optoprime v2
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 1 \
        --num_test_samples 1 \
        --batch_size 1 \
        --num_batches 1 \
        --num_epochs 3 \
        --num_threads 20 \
        --short_term_memory_duration 4 \
        --test_frequency -1 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 1 \
        --num_proposals 10 \
        --use_best_candidate_to_explore \
        --memory_size 1000 \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "debugoptoprimev2" \
        --use_regressor\
        --optoprime_version v2