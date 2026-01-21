"""
Evaluator for tau-agent optimization using OpenEvolve with rich feedback.
This evaluator provides per-task feedback through artifacts to help OpenEvolve
understand not just aggregate scores, but WHY prompts succeed or fail.
"""

import os
import json
import sys
import copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

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

# Import OpenEvolve's EvaluationResult for artifacts support
try:
    from openevolve.evaluation_result import EvaluationResult
    ARTIFACTS_AVAILABLE = True
except ImportError:
    print("Warning: openevolve not installed, artifacts will not be available")
    ARTIFACTS_AVAILABLE = False
    # Fallback to dict-only results
    class EvaluationResult:
        def __init__(self, metrics, artifacts=None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

# Global configuration
provider = "gemini"
NUM_TRAIN_SAMPLES = int(os.environ.get("NUM_TRAIN_SAMPLES", "10"))
MODEL_NAME = os.environ.get("TAU_MODEL", "gemini-2.0-flash")
SAMPLES_COUNTER_FILE = os.environ.get("SAMPLES_COUNTER_FILE", 
                                       str(tau_bench_root / "results" / "openevolve_tau" / "samples_counter.json"))
PARALLEL_EVALUATIONS = int(os.environ.get("PARALLEL_EVALUATIONS", "10"))

# Feedback configuration
MAX_FEEDBACK_TASKS = int(os.environ.get("MAX_FEEDBACK_TASKS", "3"))  # Show detailed feedback for worst N tasks
MAX_CONVERSATION_LENGTH = int(os.environ.get("MAX_CONVERSATION_LENGTH", "2000"))  # Chars per conversation


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


def format_conversation_history(messages: List[Any], max_length: int = MAX_CONVERSATION_LENGTH) -> str:
    """
    Format message history into a readable string (similar to DSPy TeacherGuide).
    
    Args:
        messages: List of message objects or dicts
        max_length: Maximum characters to include
        
    Returns:
        Formatted conversation string
    """
    if not isinstance(messages, list):
        return str(messages)[:max_length]

    parts = []
    for msg in messages:
        # Robustly extract fields whether msg is dict or object
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            tool_calls = msg.get('tool_calls', [])
            tool_name = msg.get('name', '')
            tool_id = msg.get('tool_call_id', '')
        else:
            role = getattr(msg, 'role', 'unknown')
            content = getattr(msg, 'content', '')
            tool_calls = getattr(msg, 'tool_calls', [])
            tool_name = getattr(msg, 'name', '')
            tool_id = getattr(msg, 'tool_call_id', '')

        # Skip system messages to save space
        if role == 'system':
            continue
            
        msg_str = f"{role}: {content}"
        
        # Add tool calls
        if tool_calls:
            calls_str = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get('function', {})
                    name = func.get('name', '')
                    args = func.get('arguments', '')
                else:
                    func = getattr(tc, 'function', None)
                    name = getattr(func, 'name', '') if func else ''
                    args = getattr(func, 'arguments', '') if func else ''
                calls_str.append(f"Tool: {name}({args})")
            if calls_str:
                msg_str += f" [Tool Calls: {'; '.join(calls_str)}]"
        
        # Add tool outputs
        if role == 'tool':
            msg_str = f"tool ({tool_name}, ID: {tool_id}): {content}"
        
        parts.append(msg_str)
    
    full_text = "\n".join(parts)
    if len(full_text) > max_length:
        return full_text[:max_length] + "\n... (truncated)"
    return full_text


def analyze_task_failure(reward: float, messages: List[Any], task_id: int) -> str:
    """
    Analyze why a task failed and provide actionable feedback.
    
    Args:
        reward: Task reward (0.0 or 1.0)
        messages: Conversation history
        task_id: Task identifier
        
    Returns:
        Human-readable failure analysis
    """
    if reward == 1.0:
        return "Success"
    
    # Basic failure analysis
    conversation = format_conversation_history(messages, max_length=500)
    
    # Check for common failure patterns
    failure_hints = []
    
    # Pattern 1: Too few tool calls
    tool_call_count = sum(1 for msg in messages if isinstance(msg, dict) and msg.get('tool_calls'))
    if tool_call_count == 0:
        failure_hints.append("No tools were called - agent may not understand how to use tools")
    elif tool_call_count < 2:
        failure_hints.append("Very few tool calls - agent may have given up too early")
    
    # Pattern 2: Repetitive behavior
    if len(messages) > 10:
        recent_contents = [msg.get('content', '') if isinstance(msg, dict) else '' 
                          for msg in messages[-5:]]
        if len(set(recent_contents)) < len(recent_contents) / 2:
            failure_hints.append("Repetitive behavior detected - agent may be stuck in a loop")
    
    # Pattern 3: Early termination
    if len(messages) < 5:
        failure_hints.append("Conversation very short - agent may have terminated prematurely")
    
    failure_summary = f"Task {task_id} failed."
    if failure_hints:
        failure_summary += f" Issues: {'; '.join(failure_hints)}"
    
    return f"{failure_summary}\n\nConversation:\n{conversation}"


def evaluate_agent_on_task_with_feedback(
    agent, 
    env, 
    task_index: int
) -> Tuple[float, List[Any], str]:
    """
    Evaluate agent on a single task and return reward, messages, and feedback.
    
    Args:
        agent: Agent instance
        env: Environment instance
        task_index: Task ID
        
    Returns:
        Tuple of (reward, messages, feedback_string)
    """
    try:
        # Create a deep copy of the agent to avoid interference
        agent_copy = copy.deepcopy(agent)
        reward, messages = agent_copy.forward(task_index)
        
        # Handle None reward (errors)
        if reward is None:
            feedback = f"Task {task_index}: Critical error - reward is None"
            return 0.0, [], feedback
        
        # Generate feedback
        feedback = analyze_task_failure(reward, messages, task_index)
        
        return float(reward), messages, feedback
        
    except Exception as e:
        feedback = f"Task {task_index}: Exception during evaluation - {str(e)}"
        print(feedback)
        return 0.0, [], feedback


def categorize_results(rewards: List[float]) -> Dict[str, Any]:
    """
    Categorize task results to identify patterns.
    
    Args:
        rewards: List of rewards for each task
        
    Returns:
        Dictionary with categorization statistics
    """
    num_success = sum(1 for r in rewards if r > 0.5)
    num_failure = sum(1 for r in rewards if r <= 0.5)
    
    return {
        "successful_tasks": num_success,
        "failed_tasks": num_failure,
        "success_rate": num_success / len(rewards) if rewards else 0.0,
        "perfect_score": all(r == 1.0 for r in rewards),
        "no_successes": all(r == 0.0 for r in rewards),
    }


def evaluate(program_path):
    """
    Main evaluation function called by OpenEvolve with rich feedback artifacts.
    
    Args:
        program_path: Path to the evolved program file
        
    Returns:
        EvaluationResult with metrics and artifacts
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
        
        # Set the additional instructions
        agent.additional_instructions = additional_instructions
        agent.set_env(env)
        
        # Evaluate on all training tasks in parallel WITH FEEDBACK
        print(f"Evaluating on {NUM_TRAIN_SAMPLES} tasks (parallel: {PARALLEL_EVALUATIONS} threads)...")
        task_results = [None] * NUM_TRAIN_SAMPLES  # (reward, messages, feedback) tuples
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=PARALLEL_EVALUATIONS) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(evaluate_agent_on_task_with_feedback, agent, env, task_id): task_id
                for task_id in range(NUM_TRAIN_SAMPLES)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    task_results[task_id] = result
                    reward = result[0]
                    print(f"Task {task_id}: reward = {reward}")
                except Exception as e:
                    print(f"Task {task_id} generated an exception: {e}")
                    task_results[task_id] = (0.0, [], f"Exception: {str(e)}")
        
        # Ensure all results are collected
        for i, result in enumerate(task_results):
            if result is None:
                task_results[i] = (0.0, [], f"Task {i}: No result collected")
        
        # Extract rewards and feedback
        rewards = [r[0] for r in task_results]
        all_feedback = [r[2] for r in task_results]
        
        # Compute metrics
        avg_reward = float(np.mean(rewards))
        print(f"Average reward: {avg_reward:.3f}")
        
        # Categorize results
        categories = categorize_results(rewards)
        
        # Update cumulative sample counter
        samples_this_eval = NUM_TRAIN_SAMPLES
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        
        print(f"Samples this evaluation: {samples_this_eval}")
        print(f"Cumulative samples: {cumulative_samples}")
        
        # Prepare artifacts with rich feedback
        artifacts = {}
        
        # 1. Overall summary
        artifacts["evaluation_summary"] = (
            f"Evaluated {NUM_TRAIN_SAMPLES} tasks\n"
            f"Success rate: {categories['success_rate']:.1%} ({categories['successful_tasks']}/{NUM_TRAIN_SAMPLES})\n"
            f"Average reward: {avg_reward:.3f}"
        )
        
        # 2. Per-task results summary
        task_summary_lines = []
        for i, reward in enumerate(rewards):
            status = "✓" if reward > 0.5 else "✗"
            task_summary_lines.append(f"  Task {i}: {status} (reward={reward:.2f})")
        artifacts["task_results"] = "\n".join(task_summary_lines)
        
        # 3. Detailed feedback for failed tasks (worst performers)
        failed_tasks = [(i, rewards[i], all_feedback[i]) 
                       for i in range(len(rewards)) 
                       if rewards[i] < 1.0]
        
        if failed_tasks:
            # Sort by reward (worst first) and take top N
            failed_tasks.sort(key=lambda x: x[1])
            worst_tasks = failed_tasks[:MAX_FEEDBACK_TASKS]
            
            failure_details = []
            for task_id, reward, feedback in worst_tasks:
                failure_details.append(f"=== Task {task_id} (reward={reward:.2f}) ===\n{feedback}")
            
            artifacts["failure_analysis"] = "\n\n".join(failure_details)
            
            # Overall failure patterns
            artifacts["failure_summary"] = (
                f"Failed on {len(failed_tasks)}/{NUM_TRAIN_SAMPLES} tasks.\n"
                f"Showing detailed feedback for {len(worst_tasks)} worst performers.\n"
                f"Common issue: Review the failure_analysis for patterns."
            )
        else:
            artifacts["failure_summary"] = "All tasks succeeded! ✓"
        
        # 4. Success/failure breakdown
        if categories['perfect_score']:
            artifacts["performance_hint"] = "Perfect score! Instructions are working well."
        elif categories['no_successes']:
            artifacts["performance_hint"] = (
                "No successful tasks. Instructions may be fundamentally flawed. "
                "Consider: (1) Are tool usage instructions clear? "
                "(2) Is the task objective well-defined? "
                "(3) Are there formatting issues?"
            )
        elif categories['success_rate'] < 0.3:
            artifacts["performance_hint"] = (
                "Low success rate. Major improvements needed. "
                "Review failure_analysis to identify systematic issues."
            )
        elif categories['success_rate'] < 0.7:
            artifacts["performance_hint"] = (
                "Moderate success rate. Some tasks are working. "
                "Focus on patterns in failed tasks to refine instructions."
            )
        else:
            artifacts["performance_hint"] = (
                "Good success rate. Fine-tuning may help remaining failures."
            )
        
        # 5. Instruction preview for debugging
        artifacts["instructions_preview"] = (
            f"Current instructions (first 300 chars):\n"
            f"{additional_instructions[:300]}{'...' if len(additional_instructions) > 300 else ''}"
        )
        
        print("-" * 80)
        print(f"Success rate: {categories['success_rate']:.1%}")
        print(f"Artifacts prepared: {list(artifacts.keys())}")
        print("-" * 80)
        
        # Return EvaluationResult with metrics and artifacts
        return EvaluationResult(
            metrics={
                "combined_score": avg_reward,
                "cumulative_samples": cumulative_samples,
                "num_tasks": NUM_TRAIN_SAMPLES,
                "success_rate": categories['success_rate'],
                "num_failures": categories['failed_tasks'],
            },
            artifacts=artifacts
        )
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("-" * 80)
        
        # Still update sample counter even on failure
        samples_this_eval = NUM_TRAIN_SAMPLES
        cumulative_samples = load_sample_counter() + samples_this_eval
        save_sample_counter(cumulative_samples)
        
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "cumulative_samples": cumulative_samples,
                "num_tasks": NUM_TRAIN_SAMPLES,
            },
            artifacts={
                "stderr": str(e),
                "traceback": traceback.format_exc(),
                "failure_stage": "evaluation_critical_error",
                "suggestion": "Check program syntax and imports. Review traceback for details."
            }
        )


# For backwards compatibility with different evaluation stages
def evaluate_stage1(program_path):
    """Stage 1 evaluation - uses same logic with feedback."""
    return evaluate(program_path)


def evaluate_stage2(program_path):
    """Stage 2 evaluation - full evaluation with feedback."""
    return evaluate(program_path)
