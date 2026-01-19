# OpenEvolve Tau Agent Optimization

This directory contains scripts for optimizing the tau-agent's `additional_instructions` parameter using the OpenEvolve framework.

## Overview

The optimization process uses OpenEvolve's evolutionary algorithm to improve the agent's instructions by:
1. Starting with a baseline instruction
2. Using an LLM to mutate and evolve the instructions
3. Evaluating each variant on retail customer service tasks
4. Tracking cumulative training samples used
5. Selecting the best-performing instructions

## Files

- **`initial_program.py`**: Contains the initial `additional_instructions` to optimize
- **`evaluator.py`**: Evaluates evolved instructions by running the agent on tasks
- **`config.yaml`**: OpenEvolve configuration (iterations, LLM settings, etc.)
- **`monitor_progress.py`**: Utility to monitor and report optimization progress

## Usage

### Basic Usage

Run optimization with default settings (10 tasks, 50 iterations):

```bash
cd /Users/xuanfeiren/Documents/tau-bench
python my_processing_agents/openevolve_tau_opt.py
```

### Advanced Usage

Customize the optimization parameters:

```bash
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 20 \
    --max_iterations 100 \
    --parallel_evaluations 8 \
    --model gemini-2.0-flash \
    --output_dir results/my_experiment \
    --run_name experiment_001
```

### Command Line Arguments

- `--num_train_samples`: Number of tasks to evaluate on (default: 10)
- `--max_iterations`: Maximum optimization iterations (default: 50)
- `--parallel_evaluations`: Number of parallel evaluations (default: 4)
- `--model`: LLM model for evolution (default: gemini-2.0-flash)
- `--output_dir`: Base output directory (default: results/openevolve_tau)
- `--run_name`: Unique run name (default: timestamp)
- `--project_name`: Project name for organization (default: tau-bench-openevolve)
- `--config`: Custom config file path (optional)

## Output Files

After running, the output directory will contain:

- **`best_program.py`**: The best evolved instructions
- **`final_results.json`**: Final optimization results and metrics
- **`run_metadata.json`**: Run configuration and metadata
- **`samples_counter.json`**: Cumulative sample count
- **`effective_config.yaml`**: The actual config used (after overrides)
- **`evolution_trace.jsonl`**: Detailed evolution trace
- **`progress_report.json`**: Iteration-level progress summary

## Monitoring Progress

To monitor an ongoing or completed optimization:

```bash
python my_processing_agents/openevolve_tau_opt/monitor_progress.py results/openevolve_tau/run_name
```

This will generate a progress report showing:
- Total samples used
- Best score achieved
- Per-iteration statistics

## Sample Tracking

The evaluator tracks cumulative training samples as:
- Each evaluation = 1 agent × N tasks
- Cumulative samples = sum of all evaluations
- Counter is persisted in `samples_counter.json`

**Note**: Only training samples are counted. Test evaluations (if added) should not increment the counter.

## Example Workflow

1. **Start optimization**:
   ```bash
   python my_processing_agents/openevolve_tau_opt.py \
       --num_train_samples 10 \
       --max_iterations 20 \
       --run_name pilot_test
   ```

2. **Monitor progress** (in another terminal):
   ```bash
   watch -n 60 python my_processing_agents/openevolve_tau_opt/monitor_progress.py \
       results/openevolve_tau/pilot_test
   ```

3. **View results**:
   ```bash
   cat results/openevolve_tau/pilot_test/final_results.json
   cat results/openevolve_tau/pilot_test/best_program.py
   ```

## Architecture

```
openevolve_tau_opt.py (main script)
    ↓
OpenEvolve Controller
    ↓
initial_program.py → OpenEvolve LLM → evolved programs
    ↓
evaluator.py
    ↓
ToolCallingAgent_v2 → Retail Tasks
    ↓
Metrics (combined_score, cumulative_samples)
```

## Environment Variables

The main script sets these for the evaluator:
- `NUM_TRAIN_SAMPLES`: Number of tasks to evaluate
- `TAU_MODEL`: Model name for the agent
- `SAMPLES_COUNTER_FILE`: Path to samples counter
- `GEMINI_API_KEY`: API key (must be set in environment)

## Troubleshooting

### "Module not found" errors
Make sure you're running from the tau-bench root directory and the tau conda environment is activated.

### API rate limits
Reduce `parallel_evaluations` or add delays in the config.

### Out of memory
Reduce `parallel_evaluations` or `population_size` in config.yaml.

### Slow evaluations
Consider reducing `num_train_samples` for faster iteration during development.

## Customization

### Custom Initial Instructions

Edit `initial_program.py` to start from different baseline instructions:

```python
# EVOLVE-BLOCK-START
additional_instructions = """Your custom instructions here..."""
# EVOLVE-BLOCK-END
```

### Custom Config

Create a custom config file and pass it with `--config`:

```bash
python my_processing_agents/openevolve_tau_opt.py \
    --config my_custom_config.yaml
```

### Custom Evaluator Logic

Modify `evaluator.py` to change how agents are evaluated (e.g., add test set evaluation, different metrics, etc.).

## References

- [OpenEvolve Documentation](https://github.com/CarperAI/OpenEvolve)
- [Tau-Bench](https://github.com/sierra-research/tau-bench)



