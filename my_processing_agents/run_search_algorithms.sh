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
    --long_term_memory_size 1000 \
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
        --memory_update_frequency 2 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
        --long_term_memory_size 1000 \
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
        --memory_update_frequency 0 \
        --test_frequency -1 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 3 \
        --use_best_candidate_to_explore \
        --long_term_memory_size 1000 \
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
        --memory_update_frequency 0 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 2 \
        --num_proposals 3 \
        --use_best_candidate_to_explore \
        --long_term_memory_size 1000 \
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
        --memory_update_frequency 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
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
        --memory_update_frequency 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
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
        --memory_update_frequency 4 \
        --test_frequency -1 \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 1 \
        --num_proposals 10 \
        --use_best_candidate_to_explore \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "debugoptoprimev2" \
        --use_regressor\
        --optoprime_version v2
# Try on 50 tasks. Aim for the highest score.
for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 50 \
        --num_test_samples 50 \
        --num_validate_samples 50 \
        --batch_size 2 \
        --num_batches 2 \
        --num_epochs 40 \
        --num_threads 20 \
        --memory_update_frequency 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --use_best_candidate_to_explore \
        --score_function "mean" \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-retail-compare-search-algs" \
        --run_name "PrioritySearch-with-Regressor-concentrate" \
        --use_regressor \
        --optoprime_version v2
done

# Sep 17, 2025. All large experiments crashed
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 5 \
        --num_epochs 40 \
        --num_threads 20 \
        --memory_update_frequency 4 \
        --test_frequency 4 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 1 \
        --num_proposals 1 \
        --score_range_mitin 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --run_name "PrioritySearch-with-Regressor-small" \
        --use_regressor \
        --optoprime_version v2

# Only long-term memory, debug
for i in {1..3} ; do
    python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_epochs 100 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --test_frequency 4 \
            --log_frequency 1 \
            --num_eval_samples 10 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "tau-bench-10-tasks-10-evals" \
            --run_name "try" \
            --use_regressor \
            --optoprime_version v2
done

for i in {1..3} ; do
    python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_epochs 100 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --test_frequency 4 \
            --log_frequency 1 \
            --num_eval_samples 10 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "tau-bench-10-tasks-10-evals" \
            --run_name "try" \
            --use_regressor \
            --optoprime_version v2
done
# debug for linear regressor 
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 5\
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "debug" \
        --run_name "debug-LinUCB" \
        --use_regressor \
        --regressor_type linear_ucb \
        --optoprime_version v2
# debug for llm regressor
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 5 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 1 \
        --num_candidates 2 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "debug" \
        --run_name "debug-LLM-regressor" \
        --use_regressor \
        --regressor_type llm \
        --optoprime_version v2

# New regressor 
for i in {1..3}; do
    python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_steps 50 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --test_frequency 5 \
            --log_frequency 1 \
            --num_eval_samples 10 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "tau-bench-10-tasks-10-evals" \
            --run_name "PS-Linear-regressor-Sep23-12am" \
            --use_regressor \
            --regressor_type linear \
            --optoprime_version v2
    python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_steps 50 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --test_frequency 5 \
            --log_frequency 1 \
            --num_eval_samples 10 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "tau-bench-10-tasks-10-evals" \
            --run_name "PS-Linear-UCB-regressor-Sep23-12am" \
            --use_regressor \
            --regressor_type linear_ucb \
            --optoprime_version v2
done
# debug linucb
python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_steps 50 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --log_frequency 1 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "debug" \
            --run_name "linucb-debug" \
            --use_regressor \
            --regressor_type linear_ucb \
            --optoprime_version v2

python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 50 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_transformation_exploration_factor 0.0 \
        --regressor_projection_dim 50 \
        --run_name "PS-Linear-regressor-Sep30-t_0.0-d_50-debug" \
        --use_regressor \
        --regressor_type linear \
        --optoprime_version v2

python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 101 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_projection_dim 20 \
        --regressor_regularization_strength 0.01 \
        --run_name "PS-Logistic-regressor-Sep30-d_20-debug" \
        --use_regressor \
        --regressor_type logistic \
        --optoprime_version v2
# running Sep 30 afternoon
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 51 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --test_frequency 5 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_regularization_strength 0.0001 \
        --run_name "PS-Logistic-regressor-Sep30" \
        --use_regressor \
        --regressor_type logistic \
        --optoprime_version v2
# debug linucb
python my_processing_agents/optimize_tau_agent.py \
            --num_train_samples 10 \
            --num_validate_samples 10 \
            --num_test_samples 10 \
            --batch_size 2 \
            --num_batches 2 \
            --num_steps 31 \
            --num_threads 20 \
            --memory_update_frequency 0 \
            --use_best_candidate_to_explore \
            --log_frequency 1 \
            --num_candidates 5 \
            --num_proposals 1 \
            --score_range_min 0.0 \
            --score_range_max 1.0 \
            --project_name "debug" \
            --run_name "linucb-debug" \
            --regressor_regularization_strength 0.0001 \
            --regressor_projection_dim 20 \
            --use_regressor \
            --regressor_type linear_ucb \
            --optoprime_version v2
# different linear regressor
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 31 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --test_frequency 5 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_regularization_strength 0.0001 \
        --regressor_transformation_exploration_factor 1.0 \
        --run_name "PS-Linear-regressor-Sep30-t_1.0" \
        --use_regressor \
        --regressor_type linear \
        --optoprime_version v2
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 31 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --test_frequency 5 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_regularization_strength 0.0001 \
        --regressor_transformation_exploration_factor 0.0 \
        --run_name "PS-Linear-regressor-Sep30-t_0.0" \
        --use_regressor \
        --regressor_type linear \
        --optoprime_version v2
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 2 \
        --num_steps 31 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --test_frequency 5 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 5 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_regularization_strength 0.0001 \
        --regressor_transformation_exploration_factor 0.5 \
        --run_name "PS-Linear-regressor-Sep30-t_0.5" \
        --use_regressor \
        --regressor_type linear \
        --optoprime_version v2

# use pretrained linear regressor
python my_processing_agents/optimize_tau_agent.py \
        --num_train_samples 10 \
        --num_validate_samples 10 \
        --num_test_samples 10 \
        --batch_size 2 \
        --num_batches 1 \
        --num_steps 51 \
        --num_threads 20 \
        --memory_update_frequency 0 \
        --use_best_candidate_to_explore \
        --test_frequency 5 \
        --log_frequency 1 \
        --num_eval_samples 10 \
        --num_candidates 20 \
        --num_proposals 1 \
        --score_range_min 0.0 \
        --score_range_max 1.0 \
        --project_name "tau-bench-10-tasks-10-evals" \
        --regressor_regularization_strength 0.0001 \
        --run_name "PS-pretrained-Linear-regressor-Oct3" \
        --use_regressor \
        --regressor_type pretrained_linear \
        --optoprime_version v2