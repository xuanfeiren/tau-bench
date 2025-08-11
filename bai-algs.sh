#!/bin/bash
# debug
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm" --run_name "llm-debug" 


python run-bai-algos.py --num_train_samples 50 --num_validate_samples 50 --num_test_samples 50 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm" --run_name "llm" 

python run-bai-algos.py --num_train_samples 50 --num_validate_samples 50 --num_test_samples 50 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "ucb" --run_name "ucb" 

python run-bai-algos.py --num_train_samples 50 --num_validate_samples 50 --num_test_samples 50 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "even" --run_name "even" 
