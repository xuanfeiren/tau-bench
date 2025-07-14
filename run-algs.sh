#!/bin/bash
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 
for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1 --max_buffer_size 3 --ucb_horizon 10 --num_to_sample 2 --num_phases 10
    python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1
done
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 
python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 2 --num_validate_samples 50 --num_epochs 1 

# For debugging
python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1 --num_train_samples 1 --num_validate_samples 1 --num_test_samples 1 --max_buffer_size 5 --ucb_horizon 1 --num_to_sample 1


