# Sample Calculation Guide for OpenEvolve Tau-Agent Optimization

## Understanding Sample Counting

### Basic Formula

```
Total Samples = Number of Evaluations × Tasks per Evaluation
```

Where:
- **Number of Evaluations** = Initial program + Number of iterations
- **Tasks per Evaluation** = `NUM_TRAIN_SAMPLES` (number of tasks agent runs on)

### Example Calculation

For **1300 samples with 10 tasks**:

```
1300 samples = X evaluations × 10 tasks
X evaluations = 1300 / 10 = 130 evaluations

130 evaluations = 1 (initial) + 129 (iterations)
Therefore: max_iterations = 129
```

## What Happens at Each Iteration?

### OpenEvolve's Iteration Process

```
Iteration 0 (Implicit):
  ├─ Evaluate initial_program.py
  ├─ Tasks run: 10
  └─ Cumulative samples: 10

Iteration 1:
  ├─ Select parent program (from population)
  ├─ LLM mutates parent → new_program_1
  ├─ Evaluate new_program_1 on 10 tasks
  └─ Cumulative samples: 20

Iteration 2:
  ├─ Select parent program
  ├─ LLM mutates parent → new_program_2
  ├─ Evaluate new_program_2 on 10 tasks
  └─ Cumulative samples: 30

...

Iteration 129:
  ├─ Select parent program
  ├─ LLM mutates parent → new_program_129
  ├─ Evaluate new_program_129 on 10 tasks
  └─ Cumulative samples: 1300 ✓
```

### Key Points

1. **One evaluation per iteration**: Each iteration creates and evaluates exactly ONE new program
2. **Parallel execution**: `parallel_evaluations` only affects speed, not count
   - `parallel_evaluations: 20` means 20 iterations run simultaneously
   - But still 129 total iterations = 129 new programs
3. **Initial program**: Always evaluated first (counts as 1 evaluation)

## Configuration Breakdown

### Current Config (50 iterations, 4 parallel)

```yaml
max_iterations: 50
evaluator:
  parallel_evaluations: 4
```

With `NUM_TRAIN_SAMPLES = 10`:
- Evaluations: 1 + 50 = 51
- Samples: 51 × 10 = **510 samples**
- Time: ~50 iterations / 4 parallel ≈ 13 waves

### Target Config (1300 samples, 20 parallel)

```yaml
max_iterations: 129
evaluator:
  parallel_evaluations: 20
```

With `NUM_TRAIN_SAMPLES = 10`:
- Evaluations: 1 + 129 = 130
- Samples: 130 × 10 = **1300 samples** ✓
- Time: ~129 iterations / 20 parallel ≈ 7 waves

## Using Same Dataset for Train/Validate/Test

Since you want to use the same 10 tasks for all splits:

```python
# In evaluator.py, NUM_TRAIN_SAMPLES = 10 means:
for task_id in range(10):  # Tasks 0-9
    evaluate_agent_on_task(agent, env, task_id)
```

The dataset is defined by `task_split="test"` in RunConfig, which uses the first 10 tasks from the test split. You're already using the same tasks for all evaluations.

## Sample Tracking Mechanism

### In evaluator.py

```python
# Line 173-175
samples_this_eval = NUM_TRAIN_SAMPLES  # = 10
cumulative_samples = load_sample_counter() + samples_this_eval
save_sample_counter(cumulative_samples)
```

### Sample Counter File

```json
// samples_counter.json
{
  "cumulative_samples": 1300
}
```

This file is updated after **every evaluation** and persists across the entire run.

## Verification

After running, check:

```bash
# Check final samples
cat results/openevolve_tau/run_1300samples/samples_counter.json

# Should show:
# {"cumulative_samples": 1300}

# Check final results
cat results/openevolve_tau/run_1300samples/final_results.json

# Should include:
# "cumulative_samples": 1300
```

## Common Configurations

### Quick Test (100 samples)
```yaml
max_iterations: 9  # 1 + 9 = 10 evaluations × 10 = 100 samples
parallel_evaluations: 4
```

### Medium Run (500 samples)
```yaml
max_iterations: 49  # 1 + 49 = 50 evaluations × 10 = 500 samples
parallel_evaluations: 10
```

### Full Run (1300 samples)
```yaml
max_iterations: 129  # 1 + 129 = 130 evaluations × 10 = 1300 samples
parallel_evaluations: 20
```

### Extra Large (2000 samples)
```yaml
max_iterations: 199  # 1 + 199 = 200 evaluations × 10 = 2000 samples
parallel_evaluations: 20
```

## Parallel Evaluations Impact

`parallel_evaluations` affects **speed only**, not sample count:

| Parallel | Iterations | Time per Wave | Total Waves | Est. Time* |
|----------|-----------|---------------|-------------|-----------|
| 4        | 129       | ~5 min        | 33          | ~165 min  |
| 10       | 129       | ~5 min        | 13          | ~65 min   |
| 20       | 129       | ~5 min        | 7           | ~35 min   |

*Assumes 5 min per evaluation (agent solving 10 tasks)

## Notes

- **API rate limits**: Higher `parallel_evaluations` may hit rate limits
- **Memory**: Each parallel evaluation loads a copy of the agent/environment
- **Optimal parallel**: Usually 2-4× number of CPU cores
- **Sample count**: Always deterministic based on iterations, regardless of parallel setting

## Formula Cheat Sheet

```
max_iterations = (target_samples / tasks_per_eval) - 1

Examples:
- 1300 samples, 10 tasks: (1300 / 10) - 1 = 129 iterations
- 2000 samples, 10 tasks: (2000 / 10) - 1 = 199 iterations
- 500 samples, 5 tasks:  (500 / 5) - 1 = 99 iterations
- 1000 samples, 20 tasks: (1000 / 20) - 1 = 49 iterations
```



