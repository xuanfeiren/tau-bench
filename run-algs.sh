#!/bin/bash
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 
for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 10 --num_phases 10 --train_batch_size 2 --num_to_sample 2 --run_name "ExploreAlgorithm_v1.0"
    python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 10 --num_phases 10 --train_batch_size 2 --num_to_sample 1 --num_LLM_samples 1 --run_name "ExplorewithLLM_v1.0"
    python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 10 --num_phases 10 --train_batch_size 2 --num_to_sample 0 --num_LLM_samples 2 --run_name "ExploreOnlyLLM_v1.0"
done
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 

# For debugging
python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --num_train_samples 1 --num_validate_samples 1 --num_test_samples 1 --max_buffer_size 5 --ucb_horizon 1 --num_to_sample 1


python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1

# debug
python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 1 --num_to_sample 0 --num_phases 10 --train_batch_size 1 --num_LLM_samples 5  --num_test_samples 1
python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 1 --num_to_sample 2 --num_phases 2 --train_batch_size 3 --num_test_samples 2 --num_train_samples 2 --num_validate_samples 2 --run_name "debug-for-logging"

python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 10 --num_phases 10 --train_batch_size 2 --num_to_sample 0 --num_LLM_samples 2 --run_name "ExploreOnlyLLM_v1.0"

for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 2 --log_frequency 1 --num_proposals 2 --num_validate_samples 50 --num_epochs 20 --run_name "BasicSearchAlgorithm"
done

python tau_agent_opt.py --algorithm_name "IslandSearchAlgorithm"  --num_islands 4 --num_LLM_samples 2 --num_samples_in_prompt 2 --num_threads 20  --run_name "IslandSearchAlgorithm" --num_epochs 5 
# debug for IslandSearchAlgorithm(pass)
python tau_agent_opt.py --algorithm_name "IslandSearchAlgorithm"  --num_islands 2 --num_LLM_samples 1 --num_samples_in_prompt 1 --num_threads 20  --run_name "IslandSearchAlgorithm-debug" --num_epochs 5 --num_validate_samples 1 --num_test_samples 1 

python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 3 --run_name "MinibatchwithValidation-debug" --num_train_samples 1 --num_validate_samples 1 --num_test_samples 1

for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 20 --run_name "MinibatchwithValidation"
    python tau_agent_opt.py --algorithm_name "IslandSearchAlgorithm"  --num_islands 4 --num_LLM_samples 2 --num_samples_in_prompt 2 --num_threads 20  --run_name "IslandSearchAlgorithm" --num_epochs 5 --llm_model "gemini/gemini-2.0-flash"
    python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 2 --log_frequency 1 --num_proposals 2 --num_validate_samples 50 --num_epochs 20 --run_name "BasicSearchAlgorithm"
done

python tau_agent_opt.py --algorithm_name "IslandSearchAlgorithm"  --num_islands 2 --num_LLM_samples 4 --num_samples_in_prompt 2 --num_threads 20  --run_name "IslandSearchAlgorithm-debug" --num_epochs 5 --num_train_samples 1 --num_validate_samples 1 --num_test_samples 1

python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm"  --eval_frequency 1 --log_frequency 1  --num_epochs 3 --run_name "Minibatch-with-save-debug" --num_eval_samples 1

for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 5 --num_phases 10 --train_batch_size 2 --num_to_sample 2 --ucb_exploration_factor 0.1 --run_name "ExploreAlgorithm_v1.2"
done
# 8.1
python tau_agent_opt.py --algorithm_name "DetectCorrelation"  --num_threads 20  --run_name "DetectCorrelation" --num_epochs 200 
python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 20 --run_name "MinibatchwithUCBValidation"

## Collect parameters from multiple sequential runs
python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm" --eval_frequency 1 --log_frequency 1 --num_epochs 20 --run_name "MinibatchAlgorithm-Collect-Correlation" --num_eval_samples 

## Evaluate initial agent
python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm" --run_name "Solve-all-tasks" --num_test_samples 115

python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 20 --run_name "MinibatchwithValidation"

python learn_from_success.py --num_train_samples 50 --num_test_samples 50 --num_epochs 50 --run_name "LearnFromSuccess" --eval_frequency 5

python learn_from_success.py --num_train_samples 50 --num_test_samples 50 --num_epochs 50 --run_name "LearnFromSuccess" --eval_frequency 1

# save 10 agents
python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm" --run_name "Save-agents" --num_test_samples 1 --num_train_samples 50 --num_validate_samples 50 --num_epochs 10 --eval_frequency 1 --log_frequency 1 --save_frequency 1 --num_eval_samples 1
python best_candidate_identification.py 

python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 20 --run_name "MinibatchwithValidation"
python tau_agent_opt.py --algorithm_name "MinibatchwithValidation"  --eval_frequency 2 --log_frequency 1 --num_proposals 2  --num_epochs 20 --run_name "MinibatchwithValidation"
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 3 --log_frequency 1 --num_proposals 2 --num_validate_samples 50 --num_epochs 30 --run_name "BasicSearchAlgorithm"
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 3 --log_frequency 1 --num_proposals 2 --num_validate_samples 50 --num_epochs 30 --run_name "BasicSearchAlgorithm"


# debug for ExploreAlgorithm_LLMFA
python tau_agent_opt.py --algorithm_name "ExploreAlgorithm_LLMFA" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3  --num_phases 10 --train_batch_size 2 --num_to_sample 2 --run_name "ExploreAlgorithm_v2.0-debug" --num_test_samples 1 --num_train_samples 1 --num_validate_samples 1

python tau_agent_opt.py --algorithm_name "ExploreAlgorithm_LLMFA" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3  --num_phases 10 --train_batch_size 2 --num_to_sample 2 --run_name "ExploreAlgorithm_v2.0" 

# new ucb algorithm
python my_processing_agents/tau_agent_opt.py --algorithm_name "UCBAlgorithm" --eval_frequency 5 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "vanilla-ucb" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50 --ucb_exploration_factor 0
python my_processing_agents/tau_agent_opt.py --algorithm_name "UCBAlgorithm" --eval_frequency 5 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "ucb-with-control-variate" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50 --enable_control_variate --ucb_exploration_factor 0
python my_processing_agents/tau_agent_opt.py --algorithm_name "UCBAlgorithm" --eval_frequency 5 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "vanilla-ucb" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50 --ucb_exploration_factor 0
python my_processing_agents/tau_agent_opt.py --algorithm_name "UCBAlgorithm" --eval_frequency 5 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "ucb-with-control-variate" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50 --enable_control_variate --ucb_exploration_factor 0

# debug for llm_search
python my_processing_agents/tau_agent_opt.py --algorithm_name "llm_search" --eval_frequency 4 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "llm-search-debug" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50  --num_generation_steps 5 --validate_batch_size 20 --num_eval_samples 5