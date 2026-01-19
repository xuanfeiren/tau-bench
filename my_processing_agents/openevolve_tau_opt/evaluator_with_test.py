"""
Enhanced evaluator with separate train and test sets.
Test evaluation does NOT count toward cumulative samples.
"""

import os
import json
import sys
from pathlib import Path

# Add tau-bench to path
tau_bench_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tau_bench_root))

import numpy as np
import torch
import litellm

# Set seeds
np.random.seed(10)
torch.manual_seed(10)

# Configure litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from agents.tool_calling_agent import ToolCallingAgent_openevolve as ToolCallingAgent

# Global configuration
provider = "gemini"
NUM_TRAIN_TASKS = int(os.environ.get("NUM_TRAIN_TASKS", "10"))
NUM_TEST_TASKS = int(os.environ.get("NUM_TEST_TASKS", "10"))
MODEL_NAME = os.environ.get("TAU_MODEL", "gemini-2.0-flash")
SAMPLES_COUNTER_FILE = os.environ.get("SAMPLES_COUNTER_FILE", 
                                       str(tau_bench_root / "results" / "openevolve_tau" / "samples_counter.json"))

# Task splits - you can customize these
# Option 1: Use same split (current behavior)
TRAIN_TASK_IDS = list(range(0, NUM_TRAIN_TASKS))  # [0, 1, ..., 9]
TEST_TASK_IDS = list(range(0, NUM_TEST_TASKS))     # [0, 1, ..., 9] - same as train

# Option 2: Use different tasks (recommended for true testing)
# TRAIN_TASK_IDS = list(range(0, NUM_TRAIN_TASKS))       # [0-9]
# TEST_TASK_IDS = list(range(100, 100 + NUM_TEST_TASKS))  # [100-109]

# Option 3: Holdout split
# ALL_TASK_IDS = list(range(20))
# TRAIN_TASK_IDS = ALL_TASK_IDS[:NUM_TRAIN_TASKS]  # [0-9]
# TEST_TASK_IDS = ALL_TASK_IDS[NUM_TRAIN_TASKS:]   # [10-19]

def load_sample_counter():
    """Load the cumulative sample counter from file."""
    if os.path.exists(SAMPLES_COUNTER_FILE):
        try:
            with open(SAMPLES_COUNTER_FILE, 'r') as f:
                data = json.load(f)
                return data.get('cumulative_samples', 0)
        except Exception as e:
            print(f"Warning: Failed to load sample counter: {e}")
            return 0
    return 0

def save_sample_counter(cumulative_samples):
    """Save the cumulative sample counter to file."""
    try:
        os.makedirs(os.path.dirname(SAMPLES_COUNTER_FILE), exist_ok=True)
        with open(SAMPLES_COUNTER_FILE, 'w') as f:
            json.dump({'cumulative_samples': cumulative_samples}, f)
    except Exception as e:
        print(f"Warning: Failed to save sample counter: {e}")

def extract_additional_instructions(program_path):
    """Extract additional_instructions from the evolved program file."""
    try:
        with open(program_path, 'r') as f:
            program_code = f.read()
        
        namespace = {}
        exec(program_code, namespace)
        
        if 'additional_instructions' not in namespace:
            raise ValueError("No 'additional_instructions' variable found in program")
        
        return namespace['additional_instructions']
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return "Here are the additional instructions to help the agent solve the task: "

def evaluate_agent_on_task(agent, env, task_index):
    """Evaluate agent on a single task."""
    try:
        reward, messages = agent.forward(task_index)
        if reward is None:
            return 0.0
        return float(reward)
    except Exception as e:
        print(f"Error evaluating task {task_index}: {e}")
        return 0.0

def evaluate_on_tasks(agent, env, task_ids, count_samples=True):
    """Evaluate agent on a list of tasks."""
    rewards = []
    for task_id in task_ids:
        reward = evaluate_agent_on_task(agent, env, task_id)
        rewards.append(reward)
        print(f"  Task {task_id}: reward = {reward}")
    
    avg_reward = float(np.mean(rewards))
    
    # Update sample counter only if count_samples=True
    if count_samples:
        samples_this_eval = len(task_ids)
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        return avg_reward, cumulative_samples
    else:
        # For test set, don't update counter
        return avg_reward, load_sample_counter()

def evaluate(program_path):
    """
    Main evaluation function - evaluates on TRAIN tasks only.
    This is what OpenEvolve calls during optimization.
    
    Returns:
        Dictionary with combined_score (train performance) and cumulative_samples
    """
    print("-" * 80)
    print(f"Evaluating program: {program_path}")
    print(f"Training tasks: {TRAIN_TASK_IDS}")
    print("-" * 80)
    
    try:
        additional_instructions = extract_additional_instructions(program_path)
        
        # Determine max task_id for config
        max_task_id = max(TRAIN_TASK_IDS + TEST_TASK_IDS) if TEST_TASK_IDS else max(TRAIN_TASK_IDS)
        
        config = RunConfig(
            model_provider=provider,
            user_model_provider=provider,
            model=MODEL_NAME,
            user_model=MODEL_NAME,
            num_trials=1,
            env="retail",
            agent_strategy="tool-calling",
            temperature=0.0,
            task_split="test",
            task_ids=list(range(max_task_id + 10)),  # Ensure enough tasks loaded
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        
        # Set the additional instructions directly (no trace needed)
        agent.additional_instructions = additional_instructions
        agent.set_env(env)
        
        # Evaluate on TRAIN tasks (counts toward samples)
        print(f"Evaluating on {len(TRAIN_TASK_IDS)} TRAIN tasks...")
        train_score, cumulative_samples = evaluate_on_tasks(
            agent, env, TRAIN_TASK_IDS, count_samples=True
        )
        
        print(f"Train score: {train_score:.3f}")
        print(f"Samples this evaluation: {len(TRAIN_TASK_IDS)}")
        print(f"Cumulative samples: {cumulative_samples}")
        print("-" * 80)
        
        return {
            "combined_score": train_score,  # OpenEvolve uses this for optimization
            "train_score": train_score,
            "cumulative_samples": cumulative_samples,
            "num_train_tasks": len(TRAIN_TASK_IDS),
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("-" * 80)
        
        samples_this_eval = len(TRAIN_TASK_IDS)
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        
        return {
            "combined_score": 0.0,
            "train_score": 0.0,
            "cumulative_samples": cumulative_samples,
            "num_train_tasks": len(TRAIN_TASK_IDS),
            "error": str(e)
        }

def evaluate_with_test(program_path):
    """
    Extended evaluation function that evaluates on BOTH train and test.
    Call this manually after optimization to get test performance.
    
    Test evaluation does NOT count toward cumulative samples.
    
    Returns:
        Dictionary with train_score, test_score, and cumulative_samples
    """
    print("=" * 80)
    print("FULL EVALUATION (Train + Test)")
    print("=" * 80)
    
    try:
        additional_instructions = extract_additional_instructions(program_path)
        
        max_task_id = max(TRAIN_TASK_IDS + TEST_TASK_IDS) if TEST_TASK_IDS else max(TRAIN_TASK_IDS)
        
        config = RunConfig(
            model_provider=provider,
            user_model_provider=provider,
            model=MODEL_NAME,
            user_model=MODEL_NAME,
            num_trials=1,
            env="retail",
            agent_strategy="tool-calling",
            temperature=0.0,
            task_split="test",
            task_ids=list(range(max_task_id + 10)),
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        
        # Set the additional instructions directly (no trace needed)
        agent.additional_instructions = additional_instructions
        agent.set_env(env)
        
        # Evaluate on TRAIN tasks (counts samples)
        print(f"\nEvaluating on {len(TRAIN_TASK_IDS)} TRAIN tasks...")
        train_score, cumulative_samples = evaluate_on_tasks(
            agent, env, TRAIN_TASK_IDS, count_samples=True
        )
        print(f"Train score: {train_score:.3f}")
        
        # Evaluate on TEST tasks (does NOT count samples)
        print(f"\nEvaluating on {len(TEST_TASK_IDS)} TEST tasks...")
        test_score, _ = evaluate_on_tasks(
            agent, env, TEST_TASK_IDS, count_samples=False
        )
        print(f"Test score: {test_score:.3f}")
        
        print(f"\nCumulative samples (train only): {cumulative_samples}")
        print("=" * 80)
        
        return {
            "train_score": train_score,
            "test_score": test_score,
            "generalization_gap": train_score - test_score,
            "cumulative_samples": cumulative_samples,
            "num_train_tasks": len(TRAIN_TASK_IDS),
            "num_test_tasks": len(TEST_TASK_IDS),
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        
        return {
            "train_score": 0.0,
            "test_score": 0.0,
            "generalization_gap": 0.0,
            "cumulative_samples": load_sample_counter(),
            "error": str(e)
        }

# Backwards compatibility
def evaluate_stage1(program_path):
    """Stage 1 evaluation - same as evaluate"""
    return evaluate(program_path)

def evaluate_stage2(program_path):
    """Stage 2 evaluation - same as evaluate"""
    return evaluate(program_path)

# Main function for standalone testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluator_with_test.py <program_path>")
        sys.exit(1)
    
    program_path = sys.argv[1]
    print(f"Testing program: {program_path}\n")
    
    result = evaluate_with_test(program_path)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Train Score:        {result['train_score']:.3f}")
    print(f"Test Score:         {result['test_score']:.3f}")
    print(f"Generalization Gap: {result['generalization_gap']:.3f}")
    print(f"Cumulative Samples: {result['cumulative_samples']}")
    print("=" * 80)

