#!/bin/bash
# python tau_trainer.py --algorithm_type "UCBsearch" --num_search_iterations 100
# python tau_trainer.py --algorithm_type "UCBsearch" --num_search_iterations 100
# python tau_trainer.py --algorithm_type "UCBsearchparallel" --num_search_iterations 30
for i in {1..3}; do
    python tau_trainer.py --algorithm_type "UCBsearch" --num_search_iterations 100
    python tau_trainer.py --algorithm_type "HybridUCB_LLM" --num_search_iterations 100
    python tau_trainer.py --algorithm_type "UCBSearchFunctionApproximationAlgorithm" --num_search_iterations 100
done
for i in {1..3}; do
    python tau_trainer.py --algorithm_type "UCBsearchparallel" --num_search_iterations 50
done



