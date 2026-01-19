# Your Questions Answered

## Summary of OpenEvolve's Behavior

### Q1: After picking one agent, eval on how many tasks to propose the new one?

**Answer: ZERO tasks (with current config)**

The current configuration has `cascade_evaluation: false`, which means:

1. **Parent is selected** based on its **already cached** score (from when it was evaluated previously)
2. **LLM proposes a mutation** → new program
3. **NO evaluation** happens during the proposal phase
4. The new program goes straight to full evaluation (10 tasks)

**Visual:**
```
Iteration N:
├─ Select parent (score=0.7, already known, NO new evaluation)
├─ LLM mutates parent → new_program
└─ Evaluate new_program on 10 tasks ← ONLY evaluation, counts +10 samples
```

### Q2: Then validate the new agent on how many tasks?

**Answer: 10 tasks (NUM_TRAIN_SAMPLES)**

With `cascade_evaluation: false`:
- Each new program is evaluated **once** on all 10 training tasks
- This single evaluation determines the program's score
- Counts as **+10 samples** toward cumulative total

### Q3: Cumulative samples should contain both

**Current Setup:**
- Only **one evaluation phase** per program (10 tasks)
- **Cumulative samples = number of evaluations × 10**

**If you want TWO-STAGE evaluation (proposal + validation):**

Enable cascade evaluation and implement stage1/stage2:

```yaml
# config.yaml
evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.3]  # Must score > 0.3 in stage1 to proceed
```

Then create evaluator with both stages:

```python
def evaluate_stage1(program_path):
    """Quick evaluation on 3 tasks"""
    score = evaluate_on_tasks(agent, env, [0, 1, 2])  # 3 tasks
    return {"combined_score": score}  # +3 samples

def evaluate_stage2(program_path):
    """Full evaluation on 10 tasks (only if stage1 passes)"""
    score = evaluate_on_tasks(agent, env, [0-9])  # 10 tasks
    return {"combined_score": score}  # +10 samples
```

**Sample counting with cascade:**
- If stage1 fails (score ≤ 0.3): **+3 samples** (doesn't proceed to stage2)
- If stage1 passes (score > 0.3): **+13 samples** (3 from stage1 + 10 from stage2)

## Parent Selection and Best Agent

### Q4: At each iteration, how to decide the best agent so far?

OpenEvolve tracks "best" in **three ways**:

**1. Global Best (Overall)**
```python
database.best_program  # Highest combined_score across all iterations and islands
```

**2. Per-Island Best**
```python
database.island_best_programs[island_idx]  # Best program in each island population
```

**3. MAP-Elites Grid Cells**
```python
database.elites[complexity_bin][diversity_bin]  # Best program for each feature combination
```

**How "best" is determined:**
- Uses `combined_score` from evaluation metrics
- If `combined_score` not present: averages all numeric metrics
- Higher score is always better
- Comparison: `if new_score > current_best_score: update_best`

### Q5: Does the algorithm always pick the best to do search?

**NO!** It uses a **diversity-preserving selection strategy**:

```python
# From config.yaml (database section)
exploration_ratio: 0.2   # 20% probability
exploitation_ratio: 0.7  # 70% probability
random: 0.1              # 10% probability (implicit)
```

**Parent Selection Process:**

```
Generate random number: r ∈ [0, 1]

If r < 0.2 (20%): EXPLORATION
    ├─ Sample from current island's population
    ├─ Emphasizes diversity over quality
    └─ May pick programs with low scores but unique features

If 0.2 ≤ r < 0.9 (70%): EXPLOITATION
    ├─ Sample from elite archive (top 20 programs)
    ├─ Strongly biased toward high-scoring programs
    └─ Most likely picks the "best" or near-best

If r ≥ 0.9 (10%): RANDOM
    ├─ Sample uniformly from entire population
    └─ Pure exploration, completely random
```

**Why not always pick the best?**

1. **Avoids local optima**: The "best" might be a dead-end
2. **Maintains diversity**: Explores different solution regions
3. **Discovers better solutions**: Sometimes bad parents lead to great children
4. **MAP-Elites principle**: Optimize quality AND diversity simultaneously

**Example:**
```
Best program: score=0.9, complexity=500, diversity=0.3 (complex but not unique)
Selected parent (exploration): score=0.6, complexity=100, diversity=0.9 (simple but very unique)
Result after mutation: score=0.95, complexity=120, diversity=0.9 (better than "best"!)
```

## Complete Iteration Breakdown

### What Actually Happens

```
ITERATION N DETAILED FLOW:

1. PARENT SELECTION (uses cached score, NO new evaluation)
   ├─ Roll dice: random value r
   ├─ If r < 0.2: pick from diverse programs (exploration)
   ├─ If 0.2 ≤ r < 0.9: pick from top 20 (exploitation)
   ├─ If r ≥ 0.9: pick random program
   └─ Example: Selected program_42 (score=0.75, from previous evaluation)

2. INSPIRATION SELECTION (for LLM context)
   ├─ Get top 3 programs from same island as parent
   ├─ Get 2 diverse programs
   └─ These show LLM examples of good solutions

3. LLM PROMPT CONSTRUCTION (NO evaluation)
   ├─ Show parent code + its score (0.75)
   ├─ Show inspiration codes + their scores
   ├─ Ask: "Improve this code to increase the score"
   └─ LLM sees cached scores, doesn't run any tasks

4. LLM GENERATION (NO evaluation)
   ├─ LLM analyzes patterns
   ├─ Proposes modifications
   └─ Generates new_program_N

5. EVALUATION ← THIS IS WHERE SAMPLES ARE COUNTED
   ├─ Run new_program_N on task 0 → reward_0
   ├─ Run new_program_N on task 1 → reward_1
   ├─ ...
   ├─ Run new_program_N on task 9 → reward_9
   ├─ Average rewards → combined_score (e.g., 0.80)
   └─ Samples += 10 ✓

6. POPULATION UPDATE
   ├─ Calculate features: complexity=450, diversity=0.65
   ├─ Find MAP-Elites grid cell[4][6]
   ├─ Check current occupant of cell[4][6]
   ├─ If new_score > occupant_score: replace
   ├─ If new_score > global_best_score: update best_program
   └─ Add to island population

7. LOGGING
   ├─ Save to evolution_trace.jsonl
   ├─ Update cumulative_samples counter
   └─ Continue to next iteration
```

**Key Insight:** Only step 5 runs the agent on tasks. Steps 1-4 use cached scores from previous evaluations.

## Sample Accounting

### Example: 1300 Samples Configuration

```
max_iterations: 129
NUM_TRAIN_SAMPLES: 10

Evaluation 0 (Initial):
    Program: initial_program.py
    Tasks evaluated: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Samples: +10
    Cumulative: 10

Iteration 1:
    Selected parent: initial_program (score=0.3, cached)
    Generated: program_1
    Evaluated: program_1 on tasks [0-9]
    Samples: +10
    Cumulative: 20

Iteration 2:
    Selected parent: program_1 (score=0.4, cached)
    Generated: program_2
    Evaluated: program_2 on tasks [0-9]
    Samples: +10
    Cumulative: 30

...

Iteration 50:
    Selected parent: program_35 (score=0.7, cached)
    Generated: program_50
    Evaluated: program_50 on tasks [0-9]
    Samples: +10
    Cumulative: 510

...

Iteration 129:
    Selected parent: program_88 (score=0.85, cached)
    Generated: program_129
    Evaluated: program_129 on tasks [0-9]
    Samples: +10
    Cumulative: 1300 ✓

Final best: program_88 (score=0.88, discovered at iteration 77)
```

**Total evaluations:** 130 (1 initial + 129 iterations)
**Total samples:** 1300 (130 evaluations × 10 tasks)
**Parent selections:** 129 (all use cached scores, 0 samples)

## Testing Performance After Optimization

### Problem with Current Setup

Your current configuration uses the **same 10 tasks** for training and testing:

```python
# Current behavior
TRAIN_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TEST_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Same as train!
```

**Issue:** The agent is optimized on these exact tasks, so testing on them doesn't measure generalization.

**Result:** You'll see high "test" scores but the agent might fail on unseen tasks.

### Recommended Solution: Separate Test Set

**Option 1: Different Task IDs** (Recommended)

```python
# In evaluator_with_test.py
TRAIN_TASK_IDS = list(range(0, 10))        # Tasks 0-9 for training
TEST_TASK_IDS = list(range(100, 110))      # Tasks 100-109 for testing (unseen)
```

**Option 2: Holdout Split**

```python
# Use first 20 tasks, split 10/10
TRAIN_TASK_IDS = list(range(0, 10))   # Tasks 0-9 for training
TEST_TASK_IDS = list(range(10, 20))   # Tasks 10-19 held out for testing
```

**Option 3: Cross-Validation**

```python
# Split 10 tasks into 8 train, 2 test
# Run multiple folds and average
FOLD_1 = {"train": [0,1,2,3,4,5,6,7], "test": [8,9]}
FOLD_2 = {"train": [0,1,2,3,4,5,8,9], "test": [6,7]}
# ... repeat for 5 folds
```

### How to Properly Test

**During Optimization (uses evaluator.py):**
```bash
# Only trains on TRAIN_TASKS
# Samples counted: YES
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 129
```

**After Optimization (uses evaluator_with_test.py):**
```bash
# Tests on both TRAIN and TEST tasks
# Test samples counted: NO
python my_processing_agents/openevolve_tau_opt/evaluator_with_test.py \
    results/openevolve_tau/run_1300samples/best_program.py
```

**Output:**
```
Train Score:        0.850  (optimized on these)
Test Score:         0.720  (never seen during optimization)
Generalization Gap: 0.130  (measures overfitting)
Cumulative Samples: 1300   (only from training)
```

## Practical Configuration

### For Your 1300 Samples Experiment

```yaml
# config_1300samples.yaml
max_iterations: 129

database:
  num_islands: 3
  exploration_ratio: 0.2   # 20% diverse parents
  exploitation_ratio: 0.7  # 70% elite parents
  # implicit random: 0.1   # 10% random parents

evaluator:
  cascade_evaluation: false     # Single evaluation per program
  parallel_evaluations: 20      # Speed optimization
```

```bash
# Run optimization
export NUM_TRAIN_TASKS=10
export NUM_TEST_TASKS=10
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 129 \
    --parallel_evaluations 20 \
    --config my_processing_agents/openevolve_tau_opt/config_1300samples.yaml
```

**What happens:**
- 130 evaluations (1 + 129)
- Each evaluation: 10 training tasks
- Total samples: 1300
- Parent selection: 70% picks from top performers, 30% explores
- Best program tracked globally and per-island

### After Optimization - Testing

```bash
# Modify evaluator_with_test.py to use different test tasks
# Then run:
python my_processing_agents/openevolve_tau_opt/evaluator_with_test.py \
    results/openevolve_tau/run_1300samples/best_program.py
```

**This gives you:**
- Train score (1300 samples used)
- Test score (0 additional samples)
- Generalization gap (overfitting measure)

## Key Takeaways

1. **Proposal phase**: No evaluation with current config (parent scores are cached)
2. **Validation phase**: One evaluation on 10 tasks per program
3. **Cumulative samples**: Only counts NEW evaluations (not parent selection)
4. **Best agent**: Tracked globally, per-island, and per MAP-Elites cell
5. **Parent selection**: 70% exploitation (top programs), 30% exploration/random
6. **Testing**: Use separate test set, don't count those samples

## Files Created for You

1. **`evaluator.py`**: Training only (counts samples)
2. **`evaluator_with_test.py`**: Train + test (test doesn't count samples)
3. **`ALGORITHM_EXPLAINED.md`**: This detailed explanation
4. **`config_1300samples.yaml`**: Configuration for 1300 samples
5. **`run_1300samples.sh`**: Ready-to-use script

You're all set to run proper experiments with train/test separation! 🚀



