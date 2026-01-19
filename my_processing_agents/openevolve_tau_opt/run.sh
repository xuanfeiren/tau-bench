# Run 1
python my_processing_agents/openevolve_tau_opt.py \
    --run_name run1_1300samples \
    --max_iterations 129 \
    --num_train_samples 10 \
    --parallel_evaluations 20

python my_processing_agents/openevolve_tau_opt.py \
    --run_name run2_1300samples \
    --max_iterations 129 \
    --num_train_samples 10 \
    --parallel_evaluations 20

# Run 3
for i in {3..6}; do 
    python my_processing_agents/openevolve_tau_opt.py \
        --run_name run${i}_1300samples \
        --max_iterations 129 \
        --num_train_samples 10 \
        --parallel_evaluations 20
done

