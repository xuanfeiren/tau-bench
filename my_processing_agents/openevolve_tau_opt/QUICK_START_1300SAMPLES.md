# Quick Start: 1300 Samples Configuration

## TL;DR

To get exactly **1300 samples** with **10 tasks** using **20 parallel threads**:

```bash
cd /Users/xuanfeiren/Documents/tau-bench
bash my_processing_agents/openevolve_tau_opt/run_1300samples.sh
```

Or manually:

```bash
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 129 \
    --parallel_evaluations 20 \
    --config my_processing_agents/openevolve_tau_opt/config_1300samples.yaml \
    --run_name my_1300_run
```

## The Math

```
Target: 1300 samples
Tasks per evaluation: 10
Required evaluations: 1300 / 10 = 130
Required iterations: 130 - 1 = 129 (minus 1 for initial)

Config: max_iterations = 129 ✓
```

## What You'll Get

```
├─ Initial program evaluation:     10 samples
├─ Iteration 1-10:                100 samples (cumulative: 110)
├─ Iteration 11-20:               100 samples (cumulative: 210)
├─ Iteration 21-30:               100 samples (cumulative: 310)
├─ ...
├─ Iteration 121-129:              90 samples (cumulative: 1300)
└─ Final: 130 evaluations = 1300 samples ✓
```

## Configuration Overview

### config_1300samples.yaml

```yaml
max_iterations: 129           # 129 new programs
evaluator:
  parallel_evaluations: 20    # 20 threads (fast!)
  timeout: 600               # 10 min per evaluation
database:
  population_size: 100        # Keep 100 programs in memory
  num_islands: 3             # 3 separate populations
```

### Environment Variables Set Automatically

```bash
NUM_TRAIN_SAMPLES=10          # 10 tasks per evaluation
TAU_MODEL=gemini-2.0-flash   # Model for the agent
SAMPLES_COUNTER_FILE=...     # Tracks cumulative samples
```

## Expected Behavior

### Timeline (with 20 parallel)

```
Wave 1:  Evaluations 1-20   (parallel)  →  200 samples
Wave 2:  Evaluations 21-40  (parallel)  →  400 samples
Wave 3:  Evaluations 41-60  (parallel)  →  600 samples
Wave 4:  Evaluations 61-80  (parallel)  →  800 samples
Wave 5:  Evaluations 81-100 (parallel)  → 1000 samples
Wave 6:  Evaluations 101-120 (parallel) → 1200 samples
Wave 7:  Evaluations 121-130 (parallel) → 1300 samples ✓
```

**Total time**: ~7 waves × 5 min/wave = **~35 minutes**
(Assuming 5 min per agent evaluation on 10 tasks)

### Sample Tracking

Monitor in real-time:

```bash
# In another terminal
watch -n 10 cat results/openevolve_tau/my_1300_run/samples_counter.json
```

You'll see:

```json
{"cumulative_samples": 10}    → After initial
{"cumulative_samples": 210}   → After 20 iterations
{"cumulative_samples": 410}   → After 40 iterations
...
{"cumulative_samples": 1300}  → After 129 iterations ✓
```

## Using the Same 10 Tasks for Train/Val/Test

The evaluator uses tasks 0-9 from the retail environment:

```python
# Evaluator runs agent on these tasks:
for task_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    reward = agent.solve(task_id)
```

Since you set `--num_train_samples 10`, it always uses the same 10 tasks. There's no separate validation or test set in this setup - all evaluations use these 10 tasks.

## Verification Checklist

After completion, verify:

```bash
cd results/openevolve_tau/my_1300_run

# ✓ Check samples
cat samples_counter.json
# Expected: {"cumulative_samples": 1300}

# ✓ Check final results
cat final_results.json | grep cumulative_samples
# Expected: "cumulative_samples": 1300

# ✓ Check best program exists
ls -lh best_program.py
# Should exist and contain evolved instructions

# ✓ Check metadata
cat run_metadata.json | grep -E "(best_score|cumulative_samples)"
# Should show final score and sample count
```

## Troubleshooting

### If you get fewer than 1300 samples:

**Reason**: Early stopping or errors during evaluation

**Fix**: Check that early stopping is disabled:
```yaml
early_stopping_patience: null  # Must be null
```

### If evaluations are slow:

**Option 1**: Reduce tasks for testing
```bash
--num_train_samples 5  # Only 5 tasks per eval
--max_iterations 259   # To maintain 1300 samples: (1300/5)-1
```

**Option 2**: Reduce parallel threads if hitting rate limits
```bash
--parallel_evaluations 10  # Slower but more stable
```

### If you want different sample counts:

Use this formula:
```bash
max_iterations = (target_samples / num_train_samples) - 1
```

Examples:
- 500 samples: `--max_iterations 49` (500/10 - 1)
- 1000 samples: `--max_iterations 99` (1000/10 - 1)
- 2000 samples: `--max_iterations 199` (2000/10 - 1)

## Output Files

After running, you'll have:

```
results/openevolve_tau/my_1300_run/
├── best_program.py              # Best evolved instructions
├── final_results.json           # Score: X, Samples: 1300
├── samples_counter.json         # {"cumulative_samples": 1300}
├── run_metadata.json            # Run configuration
├── effective_config.yaml        # Actual config used
├── evolution_trace.jsonl        # All 130 evaluations logged
└── progress_report.json         # Per-iteration statistics
```

## Next Steps

1. **Run the optimization**:
   ```bash
   bash my_processing_agents/openevolve_tau_opt/run_1300samples.sh
   ```

2. **Monitor progress**:
   ```bash
   # While running, in another terminal:
   python my_processing_agents/openevolve_tau_opt/monitor_progress.py \
       results/openevolve_tau/my_1300_run
   ```

3. **View results**:
   ```bash
   cat results/openevolve_tau/my_1300_run/final_results.json
   cat results/openevolve_tau/my_1300_run/best_program.py
   ```

4. **Use the best instructions**:
   The optimized instructions in `best_program.py` can be copied to your agent for deployment.

## Summary

✓ Configuration ready: `config_1300samples.yaml`  
✓ Run script ready: `run_1300samples.sh`  
✓ Will evaluate: 130 programs (1 + 129)  
✓ Will use: 1300 samples (130 × 10)  
✓ Parallel threads: 20  
✓ Same 10 tasks for all evaluations  

**You're all set to run!** 🚀



