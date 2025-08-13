#!/bin/bash
# debug
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_model" --run_name "llm-debug" 


python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm" --run_name "llm" 

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "ucb" --run_name "ucb" 

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "even" --run_name "even" 

# debug
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_regression" --run_name "llm_regression-debug"  --enable_estimate_scores

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_generator" --run_name "llm_generator-debug"  --enable_estimate_scores --num_agents 3


# runs
python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_regression" --run_name "llm_regression-enable-estimate-scores"  --enable_estimate_scores
python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_regression" --run_name "llm_regression"  
python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_generator" --run_name "llm_generator"  --enable_estimate_scores