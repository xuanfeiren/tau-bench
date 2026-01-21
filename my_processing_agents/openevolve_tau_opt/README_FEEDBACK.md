# OpenEvolve Tau Agent Optimization with Rich Feedback

This directory contains an enhanced evaluator for tau-agent prompt optimization that provides **rich per-task feedback** through OpenEvolve's artifact system.

## Key Differences from Basic Evaluator

### Basic Evaluator (`evaluator.py`)
- Returns only aggregate metrics (average reward)
- OpenEvolve sees: `combined_score: 0.65`
- Limited feedback for why prompts fail

### Enhanced Evaluator (`evaluator_with_feedback.py`)
- Returns metrics **AND** detailed artifacts
- OpenEvolve sees:
  - Aggregate score
  - Per-task success/failure breakdown
  - Detailed conversation history for failed tasks
  - Failure pattern analysis
  - Actionable improvement suggestions

## What Feedback is Provided?

The enhanced evaluator creates these artifacts:

1. **evaluation_summary**: Overall performance stats
2. **task_results**: Per-task success/failure (✓/✗)
3. **failure_analysis**: Detailed conversation history for worst N tasks
4. **failure_summary**: Count and overview of failures
5. **performance_hint**: Actionable suggestions based on success rate
6. **instructions_preview**: Preview of current instructions being tested

### Example Artifact Content

```markdown
## Last Execution Output

### evaluation_summary
Evaluated 10 tasks
Success rate: 60.0% (6/10)
Average reward: 0.600

### task_results
  Task 0: ✓ (reward=1.00)
  Task 1: ✗ (reward=0.00)
  Task 2: ✓ (reward=1.00)
  ...

### failure_analysis
=== Task 1 (reward=0.00) ===
Task 1 failed. Issues: No tools were called - agent may not understand how to use tools

Conversation:
user: I need to return my order #12345
assistant: I understand you want to return order #12345. Unfortunately, I cannot help with that.
... (truncated)

### performance_hint
Moderate success rate. Some tasks are working. 
Focus on patterns in failed tasks to refine instructions.
```

## How This Helps Evolution

With artifacts enabled, OpenEvolve's LLM can:

1. **Learn from specific failures**: "Task 3 failed because no tools were called"
2. **Identify patterns**: "All refund-related tasks are failing"
3. **Make targeted changes**: "Add explicit instructions about when to use the refund tool"
4. **Avoid repetition**: See conversation loops and fix them
5. **Debug systematically**: Use traceback and error messages

This creates a **feedback loop** where each generation learns from previous execution failures!

## Usage

### Using the Enhanced Evaluator

```bash
# Option 1: Use the helper script with feedback config
python my_processing_agents/openevolve_tau_opt.py \
    --config my_processing_agents/openevolve_tau_opt/config_with_feedback.yaml \
    --num_train_samples 10 \
    --max_iterations 50 \
    --output_dir results/openevolve_with_feedback

# Option 2: Manually specify the feedback evaluator
# Edit openevolve_tau_opt.py line 148 to use evaluator_with_feedback.py
```

### Environment Variables

The feedback evaluator supports these environment variables:

```bash
# Standard variables (same as basic evaluator)
export NUM_TRAIN_SAMPLES=10          # Number of tasks to evaluate
export TAU_MODEL="gemini-2.0-flash"  # Model to use
export PARALLEL_EVALUATIONS=4        # Parallel threads

# Feedback-specific variables
export MAX_FEEDBACK_TASKS=3          # Show detailed feedback for N worst tasks
export MAX_CONVERSATION_LENGTH=2000  # Max chars per conversation in feedback
```

### Comparing Results

Run both evaluators side-by-side to see the difference:

```bash
# Basic (no feedback)
python my_processing_agents/openevolve_tau_opt.py \
    --num_train_samples 10 \
    --max_iterations 20 \
    --output_dir results/baseline_no_feedback

# With feedback
python my_processing_agents/openevolve_tau_opt.py \
    --config my_processing_agents/openevolve_tau_opt/config_with_feedback.yaml \
    --num_train_samples 10 \
    --max_iterations 20 \
    --output_dir results/enhanced_with_feedback

# Compare final results
diff results/baseline_no_feedback/best_program.py \
     results/enhanced_with_feedback/best_program.py
```

## Configuration

Key settings in `config_with_feedback.yaml`:

```yaml
prompt:
  include_artifacts: true          # REQUIRED for feedback
  max_artifact_bytes: 20480        # 20KB limit per evaluation

evaluator:
  parallel_evaluations: 4          # Balance speed vs. resources
  timeout: 600                     # 10 minutes per evaluation
```

## Feedback Analysis Features

### 1. Failure Pattern Detection

The evaluator automatically detects:
- **No tool calls**: Agent doesn't understand tool usage
- **Repetitive behavior**: Agent stuck in loops
- **Early termination**: Agent gives up too quickly
- **Too few tool calls**: Agent not exploring enough

### 2. Conversation History

For failed tasks, full conversation history is captured:
- User requests
- Agent responses
- Tool calls with arguments
- Tool outputs
- Final resolution (or lack thereof)

### 3. Actionable Hints

Based on success rate, the evaluator provides targeted suggestions:

- **0% success**: "Instructions may be fundamentally flawed. Check tool usage clarity."
- **<30% success**: "Major improvements needed. Review failure patterns."
- **30-70% success**: "Focus on patterns in failed tasks."
- **>70% success**: "Fine-tuning may help remaining failures."
- **100% success**: "Perfect score! Instructions working well."

## File Structure

```
my_processing_agents/openevolve_tau_opt/
├── evaluator.py                    # Basic evaluator (aggregate only)
├── evaluator_with_feedback.py      # Enhanced evaluator (THIS FILE)
├── config.yaml                     # Basic config
├── config_with_feedback.yaml       # Feedback-enabled config
├── initial_program.py              # Starting prompt
└── README_FEEDBACK.md              # This file
```

## Performance Considerations

**Artifact overhead**: Minimal (<5% slowdown)
- Capturing conversation history: Already done by agent
- Formatting feedback: Simple string operations
- Artifact size: Automatically truncated to configured limit

**Benefits**: 
- Faster convergence (fewer wasted iterations)
- Better final prompts (targeted improvements)
- Easier debugging (see why things fail)

## Example: Before and After

### Without Feedback (Basic Evaluator)

```
Iteration 10: combined_score = 0.50
Iteration 11: combined_score = 0.50  
Iteration 12: combined_score = 0.50
# LLM making random changes, not learning
```

### With Feedback (Enhanced Evaluator)

```
Iteration 10: combined_score = 0.50
  Artifacts: "Failed on refund tasks - no tool calls"
  
Iteration 11: combined_score = 0.60
  Mutation: Added explicit refund tool instructions
  Artifacts: "Refund tasks now working, but cancellation failing"
  
Iteration 12: combined_score = 0.80
  Mutation: Added cancellation workflow steps
  # Systematic improvement based on feedback!
```

## Troubleshooting

### Artifacts not showing up in prompts

Check:
1. `config_with_feedback.yaml` has `include_artifacts: true`
2. Evaluator is returning `EvaluationResult` (not plain dict)
3. OpenEvolve version supports artifacts (check import)

### Feedback too verbose

Adjust environment variables:
```bash
export MAX_FEEDBACK_TASKS=1        # Reduce from 3 to 1
export MAX_CONVERSATION_LENGTH=500 # Reduce from 2000
```

### Want per-task category breakdown

Modify `categorize_results()` function to detect task types based on conversation content, then add category-specific artifacts.

## Next Steps

1. **Run with feedback enabled**: See immediate improvement in convergence
2. **Analyze artifacts in logs**: Review what feedback OpenEvolve receives
3. **Customize feedback**: Modify `analyze_task_failure()` for your specific needs
4. **Add task categories**: Extend with domain-specific failure analysis
5. **Compare results**: Run A/B test with and without feedback

## References

- OpenEvolve documentation: [Artifacts & Debugging](https://github.com/algorithmicsuperintelligence/openevolve#artifacts--debugging)
- Circle packing example with artifacts: `openevolve/examples/circle_packing_with_artifacts/`
- DSPy optimizer TeacherGuide: `my_processing_agents/dspy_opt.py:29-107`
