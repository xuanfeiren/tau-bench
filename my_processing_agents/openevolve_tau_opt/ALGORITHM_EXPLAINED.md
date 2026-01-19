# OpenEvolve Algorithm Deep Dive: How It Really Works

## Your Questions Answered

### Q1: At each step, after picking one agent, eval on how many tasks to propose the new one?

**Answer: Currently ZERO tasks for proposal.**

With the current configuration (`cascade_evaluation: false`), here's what happens:

```
Step 1: Select parent program (existing agent)
Step 2: LLM proposes modification → new_program
Step 3: Evaluate new_program on 10 tasks → get score
Step 4: Add to population if good enough
```

**There is NO separate "proposal evaluation" phase.**

The parent is selected based on its **already known** performance from when it was evaluated previously. The LLM mutates it **blindly** without evaluating the proposed change first.

### Q2: Then validate the new agent on how many tasks?

**Answer: 10 tasks (NUM_TRAIN_SAMPLES)**

The new program is evaluated ONCE on all 10 training tasks. This evaluation:
- Determines the program's score
- Decides if it gets added to the MAP-Elites grid
- Counts as +10 samples toward cumulative total

### Q3: Cumulative samples should contain both

**Current Implementation:** Only one evaluation per program (10 tasks).

**If you want proposal + validation:**

You need to enable **cascade evaluation** and implement `evaluate_stage1` and `evaluate_stage2`:

```yaml
evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.3]  # Must pass stage1 to proceed to stage2
```

Then:
- **Stage 1** (proposal): Evaluate on 3 tasks → quick filter
- **Stage 2** (validation): If stage1 score > 0.3, evaluate on full 10 tasks
- **Samples counted**: 3 (if fails stage1) OR 13 (if passes: 3 + 10)

## Parent Selection Strategy

### Q4: At each iteration, how to decide the best agent so far?

OpenEvolve tracks the best in **multiple ways**:

```python
# Global best (across all islands and iterations)
database.best_program  # Highest combined_score ever seen

# Per-island best
database.island_best_programs[island_idx]  # Best in each island

# MAP-Elites grid
database.elites  # Best program in each feature cell
```

**Combined_score** is used to determine "best":
- Returns `combined_score` if present in metrics
- Otherwise: average of all numeric metrics
- Higher is always better

### Q5: Does the algorithm always pick the best to do search?

**NO!** It uses a **mixed strategy** for diversity:

```python
# Selection probabilities (from config.yaml)
exploration_ratio: 0.2   # 20% - Sample from diverse programs
exploitation_ratio: 0.7  # 70% - Sample from elite/best programs  
random: 0.1              # 10% - Sample completely random
```

**Parent Selection Process:**

```
Random number: 0.XX

If 0.00-0.20 (20%): EXPLORATION
    ├─ Sample from current island's population
    └─ Prioritizes diversity over quality

If 0.20-0.90 (70%): EXPLOITATION
    ├─ Sample from elite archive (top 20 programs)
    └─ Heavily biased toward best performers

If 0.90-1.00 (10%): RANDOM
    ├─ Sample from entire population
    └─ Pure exploration, no bias
```

**Why not always pick the best?**
- Avoids local optima
- Maintains population diversity
- Explores different regions of solution space

## Complete Iteration Flow

Here's what actually happens at each iteration:

### Iteration N: Detailed Breakdown

```
1. SELECT PARENT (uses pre-evaluated score)
   ├─ Roll dice: exploration (20%) vs exploitation (70%) vs random (10%)
   ├─ If exploitation: pick from top 20 programs
   ├─ If exploration: pick diverse program from current island
   └─ If random: pick any program
   
2. SELECT INSPIRATIONS (for LLM context)
   ├─ Get top 3 programs from parent's island
   ├─ Get 2 diverse programs
   └─ These show the LLM what good solutions look like

3. BUILD LLM PROMPT
   ├─ Show parent code + its score
   ├─ Show inspiration programs + their scores
   ├─ Ask LLM to improve the parent
   └─ NO EVALUATION happens here

4. LLM GENERATES NEW CODE
   ├─ LLM proposes modifications
   ├─ Parse the diff or new code
   └─ Save to temp file

5. EVALUATE NEW PROGRAM ← THIS IS WHERE SAMPLES ARE COUNTED
   ├─ Run agent on task 0 → reward_0
   ├─ Run agent on task 1 → reward_1
   ├─ ...
   ├─ Run agent on task 9 → reward_9
   ├─ Average reward → combined_score
   └─ Samples += 10

6. UPDATE POPULATION
   ├─ Calculate feature coords (complexity, diversity)
   ├─ Check if new program improves its MAP-Elites cell
   ├─ If yes: replace old occupant
   ├─ If no: discard
   └─ Update best_program if better than current best

7. LOG AND CONTINUE
   └─ Save to evolution_trace.jsonl
```

**Key Insight:** Steps 1-4 use **already known** scores. Only step 5 counts new samples.

## Sample Counting Breakdown

### Example: 1300 Samples with 10 Tasks

```
Evaluation 0 (Initial Program):
    Tasks: [0,1,2,3,4,5,6,7,8,9]
    Samples: +10
    Cumulative: 10

Iteration 1:
    Parent: program_0 (score: 0.4, already known)
    LLM generates: program_1
    Evaluate program_1 on tasks [0-9]
    Samples: +10
    Cumulative: 20

Iteration 2:
    Parent: program_1 (score: 0.5, already known)
    LLM generates: program_2
    Evaluate program_2 on tasks [0-9]
    Samples: +10
    Cumulative: 30

...

Iteration 129:
    Parent: program_85 (score: 0.8, already known)
    LLM generates: program_129
    Evaluate program_129 on tasks [0-9]
    Samples: +10
    Cumulative: 1300 ✓
```

**Total evaluations:** 130 (1 initial + 129 iterations)
**Total samples:** 1300 (130 × 10)

## Testing Performance After Optimization

### Problem with Current Setup

**Issue:** You're training AND testing on the same 10 tasks!

```
Current: tasks [0-9] for both training and "testing"
Result: Overfitting, no true generalization measure
```

### Recommended Approach

**Option 1: Separate Test Set (Recommended)**

Use different tasks for train vs test:

```python
# In evaluator
TRAIN_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]       # Train on these
TEST_TASKS = [100, 101, 102, 103, 104, 105, ...]    # Test on these

# During optimization
evaluate(program) → uses TRAIN_TASKS only → counts samples

# After optimization
evaluate_test(best_program) → uses TEST_TASKS → does NOT count samples
```

**Option 2: Holdout Tasks**

```python
ALL_TASKS = list(range(20))  # 20 total tasks
TRAIN_TASKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 for training
TEST_TASKS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 10 holdout for testing
```

**Option 3: Cross-Validation** (if you really want to use same 10 tasks)

```python
# Split 10 tasks into 8 train, 2 validation
FOLD_1_TRAIN = [0,1,2,3,4,5,6,7]
FOLD_1_VAL = [8,9]

# Optimize on FOLD_1_TRAIN
# Test on FOLD_1_VAL
# Repeat with different splits and average
```

## How to Test Your Best Agent

### After Optimization Completes

```bash
# 1. Get the best program
cd results/openevolve_tau/run_1300samples
cp best_program.py best_instructions_to_test.py

# 2. Create test evaluation script
python -c "
from openevolve_tau_opt.evaluator_with_test import evaluate_with_test
result = evaluate_with_test('best_instructions_to_test.py')
print(f'Train Score: {result[\"train_score\"]:.3f}')
print(f'Test Score: {result[\"test_score\"]:.3f}')
"
```

### Compare Multiple Checkpoints

```python
# Test all checkpoints to find true best
checkpoints = [
    "results/openevolve_tau/run_1300samples/checkpoint_iter50.py",
    "results/openevolve_tau/run_1300samples/checkpoint_iter100.py",
    "results/openevolve_tau/run_1300samples/best_program.py",
]

for ckpt in checkpoints:
    test_score = evaluate_on_test_set(ckpt)
    print(f"{ckpt}: test={test_score:.3f}")
```

## MAP-Elites Grid Visualization

```
Feature Dimensions: [complexity, diversity]
Grid Size: 10 × 10 = 100 cells

Complexity →
0-100  100-200  200-300  ...  900-1000
│
Diversity
↓

Cell[0,0]:   program_5  (score: 0.4)  ← Simple & similar
Cell[0,9]:   program_12 (score: 0.6)  ← Simple & unique
Cell[9,0]:   program_45 (score: 0.7)  ← Complex & similar
Cell[5,5]:   program_88 (score: 0.9)  ← Best overall ★
```

**Each cell** keeps the best program for that feature combination.
**Best overall** might not be in every cell, just the one with highest score.

## Key Takeaways

1. **One evaluation per program** (current config)
   - No separate proposal/validation unless you enable cascade
   
2. **Parent selection is diverse**
   - 70% exploitation (best programs)
   - 20% exploration (diverse programs)
   - 10% random
   
3. **Best program tracking**
   - Global best across all iterations
   - Per-island best for each population
   - MAP-Elites best per feature cell
   
4. **Sample counting**
   - Only NEW evaluations count
   - Parent selection uses CACHED scores
   - Total samples = num_evaluations × tasks_per_eval
   
5. **For proper testing**
   - Use separate test tasks
   - Don't count test samples
   - Evaluate best program after optimization completes

## Configuration Summary

### Current Config (No Cascade)
```yaml
evaluator:
  cascade_evaluation: false
  parallel_evaluations: 20

# Per iteration:
# - 1 evaluation × 10 tasks = 10 samples
```

### With Cascade (Proposal + Validation)
```yaml
evaluator:
  cascade_evaluation: true
  cascade_thresholds: [0.3]
  parallel_evaluations: 20

# Per iteration (if passes stage1):
# - Stage 1: 1 eval × 3 tasks = 3 samples
# - Stage 2: 1 eval × 10 tasks = 10 samples
# - Total: 13 samples per successful program
# - Failed programs: only 3 samples
```

## Next Steps

1. **Use the provided `evaluator_with_test.py`** (I'll create it next)
2. **Separate train/test tasks** for proper evaluation
3. **After optimization:** Test best program on holdout set
4. **Compare:** Train score vs test score to measure overfitting
