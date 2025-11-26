import numpy as np
import torch
import dspy
from dspy.teleprompt import GEPA
import argparse
import os
import time
from typing import Optional, List, Dict, Any
import litellm

# Import Agent and Environment
from tau_bench.envs import get_env
from tau_bench.types import RunConfig, SolveResult
from tau_bench.envs.base import Env
from agents.tool_calling_agent import ToolCallingAgentDSPy as ToolCallingAgent

# Set seeds
np.random.seed(10)
torch.manual_seed(10)

# Configure litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
provider = "gemini"

# -----------------------------------------------------------------------------
# 1. Teacher Guide
# -----------------------------------------------------------------------------
class TeacherGuide:
    """Guide that extracts reward and feedback from the agent's output."""
    def __init__(self, env: Env, config: RunConfig):
        self.env = env
        self.config = config
        
    def get_feedback(self, task_index, output: tuple, info: Any) -> tuple[float, str]:   
        """
        Compute score and feedback for a rollout.
        
        Args:
            task_index: The task identifier
            output: Tuple of (reward, messages) from the agent
            info: Additional info from environment or execution
            
        Returns:
            (score, feedback_string)
        """
        reward, messages = output
        
        # Handle explicit errors passed via info
        if info == "BadRequest":
            return 0.0, "BadRequestError. Please adjust the tool information to the correct form."
        
        # Success case
        if reward == 1.0:
            return 1.0, "Correct"
            
        # Failure analysis case
        conversation_history = self._format_history(messages)
        feedback = f"The agent failed to solve the task. Here is the conversation history:\n{conversation_history}"
        return float(reward), feedback

    def _format_history(self, messages: List[Any]) -> str:
        """Format message history into a readable string."""
        if not isinstance(messages, list):
            return str(messages)

        parts = []
        for msg in messages:
            # robustly extract fields whether msg is dict or object
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
            
        return "\n".join(parts)

# -----------------------------------------------------------------------------
# 2. Metric
# -----------------------------------------------------------------------------

# Global guide instance for stateless metric access
_GLOBAL_GUIDE = None

def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    DSPy Metric that delegates scoring and feedback to the TeacherGuide.
    """
    if _GLOBAL_GUIDE is None:
        return dspy.Prediction(score=0.0, feedback="System Error: Guide not initialized")
    
    # 1. Extract inputs
    x = gold.x  # Task index
    
    # 2. Extract outputs (reward and conversation history)
    # pred is the dspy.Prediction returned by ToolCallingAgent.forward
    reward = pred.reward
    messages = pred.messages
    
    # Ensure reward is float
    try:
        reward = float(reward)
    except (ValueError, TypeError):
        reward = 0.0
        
    target_output = (reward, messages)
    
    # 3. Get info (dataset info or runtime info)
    pred_info = getattr(pred, 'info', None)
    guide_info = pred_info if pred_info is not None else gold.info
    
    # 4. Compute Score & Feedback
    score, feedback = _GLOBAL_GUIDE.get_feedback(x, target_output, guide_info)
    
    return dspy.Prediction(score=score, feedback=feedback)

# -----------------------------------------------------------------------------
# 3. Setup & Main
# -----------------------------------------------------------------------------

def create_retail_dataset(num_tasks=10):
    """Create DSPy dataset where each example has x (task_id) and info."""
    return [
        dspy.Example(x=i, info=i).with_inputs("x") 
        for i in range(num_tasks)
    ]

def main():
    global _GLOBAL_GUIDE
    parser = argparse.ArgumentParser(description="Optimize Retail Agent Instructions with GEPA")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of tasks to use")
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help="LLM model name", choices=['gemini-2.5-flash-lite', 'gemini-2.0-flash'])
    parser.add_argument('--use_wandb', action='store_true',default=True, help="Enable WandB logging")
    parser.add_argument('--project', type=str, default='debug-DSPy')
    parser.add_argument('--run_name', type=str, default='DSPy_GEPA')
    parser.add_argument('--num_threads', type=int, default=20, help="Parallel evaluation threads")
    parser.add_argument('--max_metric_calls', type=int, default=2000, help="Budget for GEPA")
    parser.add_argument('--log_frequency', type=int, default=2, help="Save snapshots every N iterations")
    args = parser.parse_args()

    # A. Configure DSPy
    print(f"Configuring DSPy with {args.model}...")
    # Ensure correct model format for dspy.LM (provider/model)
    model_name = args.model if "/" in args.model else f"{provider}/{args.model}"
    lm = dspy.LM(model=model_name)
    dspy.configure(lm=lm)

    # B. Setup Environment
    print("Initializing Environment...")
    config = RunConfig(
        model_provider=provider,
        user_model_provider=provider,
        model=args.model,
        user_model=args.model,
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        task_ids=list(range(args.num_samples)),
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
    
    # Initialize Guide
    _GLOBAL_GUIDE = TeacherGuide(env, config)
    
    # C. Prepare Data
    print("Creating Datasets...")
    dataset = create_retail_dataset(args.num_samples)
    
    # D. Initialize Agent
    print("Initializing Agent...")
    dspy_agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature
    )
    dspy_agent.set_env(env)
    
    # E. Run Optimization
    print("Starting GEPA Optimization...")
    gepa = GEPA(
        metric=gepa_metric,
        reflection_lm=lm,
        candidate_selection_strategy='pareto',
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=2,
        track_stats=True,
        use_wandb=args.use_wandb,
        num_threads=args.num_threads,
        wandb_init_kwargs={'project': args.project, 'name': args.run_name} if args.use_wandb else None,
        log_dir=f"dspy_results/gepa_Nov25",
        log_frequency=args.log_frequency
    )
    
    start_time = time.time()
    optimized_program = gepa.compile(
        student=dspy_agent,
        trainset=dataset,
        valset=dataset
    )
    duration = time.time() - start_time
    
    # F. Report Results
    print(f"\nOptimization completed in {duration:.2f}s")
    print("\n" + "="*40)
    print("OPTIMIZED INSTRUCTION:")
    print("="*40)
    print(optimized_program.prog.signature.__doc__)
    print("="*40)

    # Save
    save_path = f"results/gepa_optimized_{args.run_name}.json"
    os.makedirs("results", exist_ok=True)
    optimized_program.save(save_path)
    print(f"Saved optimized program to {save_path}")

if __name__ == "__main__":
    main()
