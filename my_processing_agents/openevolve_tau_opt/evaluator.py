"""
Evaluator for tau-agent optimization using OpenEvolve.
This evaluator loads evolved additional_instructions and evaluates the agent.
"""

import os
import json
import sys
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add tau-bench to path
tau_bench_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(tau_bench_root))

import numpy as np
import torch
import litellm
import logging

# Set seeds for reproducibility
np.random.seed(10)
torch.manual_seed(10)

# Configure litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from agents.tool_calling_agent import ToolCallingAgent_openevolve as ToolCallingAgent

# Global configuration
provider = "gemini"
NUM_TRAIN_SAMPLES = int(os.environ.get("NUM_TRAIN_SAMPLES", "10"))
MODEL_NAME = os.environ.get("TAU_MODEL", "gemini-2.0-flash")
SAMPLES_COUNTER_FILE = os.environ.get("SAMPLES_COUNTER_FILE", 
                                       str(tau_bench_root / "results" / "openevolve_tau" / "samples_counter.json"))
# Number of parallel threads for task evaluation (default: 10, can be overridden via environment)
PARALLEL_EVALUATIONS = int(os.environ.get("PARALLEL_EVALUATIONS", "10"))
# Number of parallel threads for task evaluation (default: 10, can be overridden via environment)
PARALLEL_EVALUATIONS = int(os.environ.get("PARALLEL_EVALUATIONS", "10"))

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
        
        # Execute the program code to extract the variable
        namespace = {}
        exec(program_code, namespace)
        
        if 'additional_instructions' not in namespace:
            raise ValueError("No 'additional_instructions' variable found in program")
        
        return namespace['additional_instructions']
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        # Return default if extraction fails
        return "Here are the additional instructions to help the agent solve the task: "

def evaluate_agent_on_task(agent, env, task_index):
    """Evaluate agent on a single task."""
    try:
        # Create a deep copy of the agent to avoid interference between parallel evaluations
        agent_copy = copy.deepcopy(agent)
        reward, messages = agent_copy.forward(task_index)
        
        # Handle None reward (errors)
        if reward is None:
            return 0.0
        
        return float(reward)
    except Exception as e:
        print(f"Error evaluating task {task_index}: {e}")
        return 0.0

def evaluate(program_path):
    """
    Main evaluation function called by OpenEvolve.
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        Dictionary with metrics including combined_score and cumulative_samples
    """
    print("-" * 80)
    print(f"Evaluating program: {program_path}")
    print(f"Number of training samples: {NUM_TRAIN_SAMPLES}")
    print("-" * 80)
    
    try:
        # Extract the evolved additional_instructions
        additional_instructions = extract_additional_instructions(program_path)
        print(f"Extracted instructions (first 200 chars): {additional_instructions[:200]}...")
        
        # Create configuration
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
            task_ids=list(range(NUM_TRAIN_SAMPLES)),
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
        # Initialize environment
        print("Initializing environment...")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        # Initialize agent with evolved instructions
        print("Initializing agent...")
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        
        # Set the additional instructions directly (no trace needed for OpenEvolve)
        agent.additional_instructions = additional_instructions
        agent.set_env(env)
        
        # Evaluate on all training tasks in parallel
        print(f"Evaluating on {NUM_TRAIN_SAMPLES} tasks (parallel: {PARALLEL_EVALUATIONS} threads)...")
        rewards = [None] * NUM_TRAIN_SAMPLES
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=PARALLEL_EVALUATIONS) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(evaluate_agent_on_task, agent, env, task_id): task_id
                for task_id in range(NUM_TRAIN_SAMPLES)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    reward = future.result()
                    rewards[task_id] = reward
                    print(f"Task {task_id}: reward = {reward}")
                except Exception as e:
                    print(f"Task {task_id} generated an exception: {e}")
                    rewards[task_id] = 0.0
        
        # Ensure all rewards are collected (handle any None values)
        rewards = [r if r is not None else 0.0 for r in rewards]
        
        # Compute average reward
        avg_reward = float(np.mean(rewards))
        print(f"Average reward: {avg_reward:.3f}")
        
        # Update cumulative sample counter
        # Each evaluation = 1 agent × NUM_TRAIN_SAMPLES tasks
        samples_this_eval = NUM_TRAIN_SAMPLES
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        
        print(f"Samples this evaluation: {samples_this_eval}")
        print(f"Cumulative samples: {cumulative_samples}")
        print("-" * 80)
        
        # Return metrics
        # OpenEvolve uses 'combined_score' as the primary fitness metric
        return {
            "combined_score": avg_reward,
            "cumulative_samples": cumulative_samples,
            "num_tasks": NUM_TRAIN_SAMPLES,
            "success_rate": avg_reward  # For clarity
        }
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("-" * 80)
        
        # Still update sample counter even on failure
        samples_this_eval = NUM_TRAIN_SAMPLES
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        
        return {
            "combined_score": 0.0,
            "cumulative_samples": cumulative_samples,
            "num_tasks": NUM_TRAIN_SAMPLES,
            "error": str(e)
        }

# For backwards compatibility with different evaluation stages
def evaluate_stage1(program_path):
    """Stage 1 evaluation - uses same logic but could be subset of tasks."""
    return evaluate(program_path)

def evaluate_stage2(program_path):
    """Stage 2 evaluation - full evaluation."""
    return evaluate(program_path)

