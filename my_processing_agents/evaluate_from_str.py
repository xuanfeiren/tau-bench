#!/usr/bin/env python3
"""
Script to load and evaluate custom agents on the test set.
This script loads the agents and evaluates their performance without training.
"""

from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from optimize_tau_agent import create_retail_dataset, TeacherGuide
from opto.trainer.evaluators import evaluate
import numpy as np

provider = "gemini"

def _evaluate_agent(agent, guide, dataset, min_score=0, num_threads=20, num_eval_times=5):
    """Evaluate an agent - copied from run-bai-algos.py"""
    eval_scores = evaluate(agent, guide, dataset['inputs'], dataset['infos'],
                          min_score=min_score,
                          num_threads=num_threads,
                          num_samples=num_eval_times,
                          description=f"Evaluating agent")
    # Create table with explicit column names
    if eval_scores.ndim > 1:
        columns = [f'Eval_{i+1}' for i in range(eval_scores.shape[1])]
        all_valid_scores = [score for row in eval_scores for score in row if score is not None]
    else:
        all_valid_scores = [score for score in eval_scores if score is not None]
    test_score = np.mean(all_valid_scores) if all_valid_scores else 0
    return test_score


def evaluate_agent_from_str(
    instruction_str: str = "Here are the additional instructions to help the agent solve the task: ",
    num_test_samples: int = 10,
    num_threads: int = 20,
    num_eval_times: int = 10,
    model: str = 'gemini-2.0-flash',
    user_model: str = 'gemini-2.0-flash'
):
    """
    Evaluate an agent with custom instructions.
    
    Args:
        instruction_str: Custom instruction string to set for the agent
        num_test_samples: Number of test samples per evaluation (default: 10)
        num_threads: Number of threads for parallel processing (default: 20)
        num_eval_times: Number of evaluation runs per step (default: 10)
        model: Model to use for the agent (default: 'gemini-2.0-flash')
        user_model: Model to use for the user (default: 'gemini-2.0-flash')
    
    Returns:
        float: The evaluation score
    """
    # Create configuration - same as run-bai-algos.py
    config = RunConfig(
        model_provider=provider,
        user_model_provider=provider,
        model=model,
        user_model=user_model,
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        task_ids=list(range(num_test_samples)),
        log_dir="results",
        max_concurrency=1,
        seed=10,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None
    )
    
    # Initialize environment - same as run-bai-algos.py
    # print(f"Initializing retail environment with user strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
        task_index=0
    )
    
    # Create test dataset
    # print("Creating test dataset...")
    test_dataset = create_retail_dataset(env, num_tasks=num_test_samples)
    # print(f"Test samples: {len(test_dataset['inputs'])}")
    
    # Initialize guide for evaluation
    guide = TeacherGuide(env, config)
    
    # Create agent - same pattern as run-bai-algos.py
    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature
    )
    
    agent.set_env(env)
    agent.additional_instructions._set(instruction_str)
    
    # Evaluate agent
    score = _evaluate_agent(
        agent=agent,
        guide=guide,
        dataset=test_dataset,
        min_score=0,
        num_threads=num_threads,
        num_eval_times=num_eval_times
    )
    
    return score


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Agent from Instruction String')
    
    parser.add_argument('--instruction_str', type=str, default="Here are the additional instructions to help the agent solve the task: ",
                       help='Custom instruction string for the agent')
    parser.add_argument('--num_test_samples', type=int, default=10,
                       help='Number of test samples per evaluation')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--num_eval_times', type=int, default=10,
                       help='Number of evaluation runs per step')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    
    args = parser.parse_args()
    
    score = evaluate_agent_from_str(
        instruction_str=args.instruction_str,
        num_test_samples=args.num_test_samples,
        num_threads=args.num_threads,
        num_eval_times=args.num_eval_times,
        model=args.model,
        user_model=args.user_model
    )
    
    print(f"Final score: {score}")
