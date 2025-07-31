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

python tau_agent_opt.py --algorithm_name "DetectCorrelation"  --num_threads 20  --run_name "DetectCorrelation" --num_epochs 2000 
python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm"  --eval_frequency 1 --log_frequency 1  --num_epochs 3 --run_name "Minibatch-with-save-debug" --num_eval_samples 1

for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 5 --num_phases 10 --train_batch_size 2 --num_to_sample 2 --run_name "ExploreAlgorithm_v1.1"
done