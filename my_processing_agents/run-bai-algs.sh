#!/bin/bash
# debug
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_model" --run_name "llm-debug" 


python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm" --run_name "llm" 

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "ucb" --run_name "ucb" 

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "even" --run_name "even" 


# debug
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_regression" --run_name "llm_regression-debug"  --enable_estimate_scores

python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 2 --bai_algo "llm_generator" --run_name "llm_generator-debug"  --enable_estimate_scores --num_agents 2

python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 2 --bai_algo "llm_onecall" --run_name "llm_onecall-debug"  --enable_estimate_scores --num_agents 2

python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 2 --bai_algo "llm_simple" --run_name "llm_simple-debug"  --enable_estimate_scores --num_agents 2

python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 2 --bai_algo "llm_thompson" --run_name "llm_thompson-debug"  --enable_estimate_scores --num_agents 2
python run-bai-algos.py --num_train_samples 4 --num_validate_samples 4 --num_test_samples 4 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 2 --bai_algo "llm_ts_onecall" --run_name "llm_ts_onecall-debug"  --enable_estimate_scores --num_agents 2

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_generator" --run_name "llm_generator-twocall-ts-debug"  --enable_estimate_scores

# 6 config runs (10 agents)
python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "ucb" --run_name "ucb" --num_agents 10

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_regression" --run_name "llm_regression"  --num_agents 10


python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_regression" --run_name "llm_regression-enable-estimate-scores"  --enable_estimate_scores --num_agents 10

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_generator" --run_name "llm_generator-2call"  --enable_estimate_scores --num_agents 10

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_onecall" --run_name "llm_generator-onecall"  --enable_estimate_scores --num_agents 10

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_simple" --run_name "llm_simple-generate"  --enable_estimate_scores --enable_using_regressor --num_agents 10

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_simple" --run_name "llm_simple-generate-without-regressor"  --enable_estimate_scores --num_agents 10


# run TS with different temperatures
python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_thompson" --run_name "llm_thompson-t-0.0"  --temperature 0.0

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_thompson" --run_name "llm_thompson-t-0.5"  --temperature 0.5

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_thompson" --run_name "llm_thompson-t-1.0"  --temperature 1.0

python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 10 --bai_algo "llm_thompson" --run_name "llm_thompson-t_2-2*sqrt(epoch/num_epochs)"  





python run-bai-algos.py --num_train_samples 10 --num_validate_samples 10 --num_test_samples 10 --num_threads 20 --eval_frequency 1 --log_frequency 1 --num_epochs 5 --bai_algo "llm_ts_onecall" --run_name "llm_ts_onecall-t-0.0"  --temperature 0.0 --num_agents 10

# Score Prediction
python my_processing_agents/run-score-prediction.py  --num_epochs 2000 --run_name "Embedding_Regression"
python my_processing_agents/run-score-prediction.py  --num_epochs 50 --run_name "Embedding_regularized_Regression" --bai_algo "embedding_regression" --validate_batch_size 10

# python run-score-prediction.py  --num_epochs 50 --run_name "Embedding_Regression"
# python run-score-prediction.py  --num_epochs 20 --run_name "Embedding_Regression"

python my_processing_agents/run-score-prediction.py  --num_epochs 200 --run_name "score_prediction_half" --bai_algo "score_prediction_half" --validate_batch_size 2 --run_name "score_prediction_half-small-batch"


# debug for projected embedding regression









# TODO: Aug 26, 2025
# Regularized Embedding Regression
for i in {1..2}; do
    python my_processing_agents/run-score-prediction.py  --num_epochs 50 --run_name "Embedding_regularized_Regression" --bai_algo "embedding_regression" --validate_batch_size 10 --alpha 1e-4 
done

for i in {1..2}; do
    python my_processing_agents/run-score-prediction.py  --num_epochs 50 --run_name "Projected_Embedding_Regression" --bai_algo "projected_embedding_regression" --validate_batch_size 10 --alpha 0.1 --lower_dim 10
done

for i in {1..3}; do
    python my_processing_agents/tau_agent_opt.py --algorithm_name "llm_search" --eval_frequency 4 --log_frequency 1 --num_epochs 20 --train_batch_size 2  --run_name "llm-search" --num_test_samples 50 --num_train_samples 50 --num_validate_samples 50  --num_generation_steps 5 --validate_batch_size 20 --num_eval_samples 5
done





