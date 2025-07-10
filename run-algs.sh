#!/bin/bash

for i in {1..3}; do
    python tau_agent_opt.py --algorithm_name "MinibatchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_epochs 1 
    python tau_agent_opt.py --algorithm_name "BasicSearchAlgorithm" --eval_frequency 10 --log_frequency 10 --num_proposals 3 --num_validate_samples 50 --num_epochs 1 
    python tau_agent_opt.py --algorithm_name "ExploreAlgorithm" --eval_frequency 1 --log_frequency 1
    python tau_agent_opt.py --algorithm_name "ExplorewithLLM" --eval_frequency 1 --log_frequency 1
done




