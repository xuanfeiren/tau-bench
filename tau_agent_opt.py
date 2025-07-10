# Repeat the agent on the first task multiple times until it succeeds
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy

import opto 
from opto import trace
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 

# Copyright Sierra

import json
from litellm import completion
from typing import List, Optional, Dict, Any
import argparse

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.model_utils.model.utils import trim_conversation_messages
from opto.trainer.loggers import WandbLogger, DefaultLogger
from opto.trainer.algorithms.explore import ExploreAlgorithm, ExplorewithLLM
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm, BasicSearchAlgorithm
from opto.trainer.guide import AutoGuide

import litellm 
litellm.drop_params = True
litellm.suppress_debug_info = True
import sys
import os
import time
from datetime import datetime
os.environ["TRACE_LITELLM_MODEL"] = "gemini/gemini-2.0-flash"


OBJECTIVE = """Optimize the agent's performance by improving both tool descriptions and additional instructions in #Variables based on #Feedback.

TASK: You are optimizing a retail customer service agent by modifying:
1. Tool descriptions - to clarify tool usage and prevent errors
2. Additional instructions - to provide strategic guidance and best practices

#Variables contains: 
- Tool schemas with function names, descriptions, and parameters
- Additional instructions that guide the agent's behavior

#Feedback contains: Either "Correct" (success) or conversation history (failure analysis needed)

INSTRUCTIONS:
1. If feedback is "Correct": Make minor refinements to maintain successful patterns
2. If feedback contains conversation history: Analyze failure patterns to identify:
   - Which tools were used incorrectly or missed
   - Parameter confusion or formatting errors  
   - Workflow sequence problems
   - Missing strategic guidance or best practices

OPTIMIZATION RULES:
For Tool Information:
- ONLY modify the 'description' fields of tools
- NEVER change function names or parameter schemas
- MUST include ALL original tools in your output

For Additional Instructions:
- Provide specific guidance based on observed failures
- Include best practices for retail customer service
- Add workflow tips and common pitfall warnings
- Keep instructions concise but actionable

OUTPUT FORMAT:
Your response must contain ONLY these two sections:
1. "reasoning": Explain your analysis of the feedback and what needs to be improved
2. "suggestion": Provide both the complete optimized tool information AND the improved additional instructions

Do not include any other text, explanations, or keywords like TERMINATE."""

@trace.model
class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        super().__init__()
        self.tools_info = trace.node(tools_info, trainable=True)
        self.wiki = wiki
        self.additional_instructions = trace.node("Here are the additional instructions to help the agent solve the task: ", trainable=True)
        self.model = model
        self.provider = provider
        self.temperature = temperature

    @trace.bundle()
    def solve(self, tools_info, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info."""
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "system", "content": additional_instructions},
            {"role": "user", "content": obs},
        ]
        
        for step in range(max_num_steps):
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": tools_info,
                "temperature": self.temperature,
            }
            
            # Retry logic with exponential backoff for entire interaction
            max_retries = 10
            base_delay = 1.0
            step_successful = False
            
            for retry_attempt in range(max_retries):
                try:
                    # Step 1: Get completion from API
                    res = completion(**completion_kwargs)
                    
                    # Step 2: Process the response
                    next_message = res.choices[0].message.model_dump()
                    # if retry_attempt >= 1: #debug
                    #     print("Completion succeeded.")
                    cost = res._hidden_params.get("response_cost")
                    if cost is not None:
                        total_cost += cost
                    
                    # Step 3: Convert message to action
                    action = message_to_action(next_message)
                    
                    # Step 4: Execute action in environment
                    env_response = env.step(action)
                    
                    # If we get here, everything succeeded
                    step_successful = True
                    break
                    
                except Exception as e:
                    # print(f"Step {step}: Error: {e}, tring to retry...")
                    error_str = str(e).lower()
                    error_type = type(e).__name__.lower()
                    
                    # Check if it's a retryable error
                    retryable_errors = [
                        'rate limit', 'timeout', 'temporary', 'service unavailable',
                        'internal server error', 'bad gateway', 'service temporarily unavailable',
                        'too many requests', 'quota', 'overloaded', 'resource has been exhausted',
                        'resource_exhausted', 'ratelimiterror', 'quotaexceedederror',
                        'connection error', 'network', 'json decode'
                    ]
                    
                    # Also check specific litellm exceptions
                    retryable_exception_types = [
                        'ratelimiterror', 'timeouterror', 'apiconnectionerror', 
                        'serviceunavailableerror', 'internalservererror', 'jsondecodeerror'
                    ]
                    
                    is_retryable = (
                        any(err in error_str for err in retryable_errors) or
                        any(exc_type in error_type for exc_type in retryable_exception_types) or
                        'code": 429' in error_str or  # HTTP 429 Too Many Requests
                        'code": 503' in error_str or  # HTTP 503 Service Unavailable
                        'code": 502' in error_str or  # HTTP 502 Bad Gateway
                        'code": 500' in error_str     # HTTP 500 Internal Server Error
                    )
                    
                    if retry_attempt == max_retries - 1:
                        # Last attempt failed
                        print(f"Step {step}: Failed after {max_retries} attempts. Error: {e}")
                        break
                    elif is_retryable:
                        # Special handling for rate limit errors - use longer delays
                        is_rate_limit = (
                            'rate limit' in error_str or 'ratelimiterror' in error_type or
                            'quota' in error_str or 'resource has been exhausted' in error_str or
                            'code": 429' in error_str
                        )
                        
                        if is_rate_limit:
                            # Longer delays for rate limits: 2, 8, 18, 32, 50 seconds
                            delay = 2 * (retry_attempt + 1) ** 2 + retry_attempt
                        else:
                            # Standard exponential backoff for other errors
                            delay = base_delay * (2 ** retry_attempt) + (0.1 * retry_attempt)
                        
                        error_type_desc = "Rate limit" if is_rate_limit else "Retryable error"
                        # print(f"Step {step}: {error_type_desc} - Retry {retry_attempt + 1}/{max_retries} after {delay:.1f}s. Error: {e}")
                        time.sleep(delay)
                    else:
                        # Non-retryable error
                        print(f"Step {step}: Non-retryable error: {e}")
                        return 0, [], {}
            
            if not step_successful:
                print(f"Step {step}: Skipping step due to interaction failure")
            else:
                # Only process results if step was successful
                reward = env_response.reward
                info = {**info, **env_response.info.model_dump()}
                
                if action.name != RESPOND_ACTION_NAME:
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    messages.extend([
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ])
                else:
                    messages.extend([
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ])
                    
                if env_response.done:
                    break
                
        result = SolveResult(reward=reward, info=info, messages=messages, total_cost=total_cost)
        
        if result.reward == 1:
            return result.reward, "Correct", "Correct"
        else:
            return result.reward, result.messages, result.info
        
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility."""
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.tools_info, self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env

def message_to_action(message: Dict[str, Any]) -> Action:
    """Convert message to action."""
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})

class TeacherGuide(AutoGuide):
    """Guide that extract reward and feedback from the agent's output."""
    def __init__(self, env: Env, config: RunConfig):
        super().__init__()
        self.env = env
        self.config = config
        
    def get_feedback(self, task, output: SolveResult, info):   
        """Get feedback from the agent's output."""
        reward, messages, info = output
        if reward == 1:
            feedback = "Correct"
        else:
            conversation_parts = []
            for msg in messages:
                msg_str = f"{msg['role']}: {msg.get('content', '')}"
                
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls_str = []
                    for tool_call in msg['tool_calls']:
                        if 'function' in tool_call:
                            func_name = tool_call['function'].get('name', '')
                            func_args = tool_call['function'].get('arguments', '')
                            tool_calls_str.append(f"Tool: {func_name}({func_args})")
                    if tool_calls_str:
                        msg_str += f" [Tool Calls: {'; '.join(tool_calls_str)}]"
                
                if msg['role'] == 'tool':
                    tool_name = msg.get('name', '')
                    tool_call_id = msg.get('tool_call_id', '')
                    msg_str = f"tool ({tool_name}, ID: {tool_call_id}): {msg.get('content', '')}"
                
                conversation_parts.append(msg_str)
            
            feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join(conversation_parts)
        return reward, feedback
        
    def metric(self, task, output: SolveResult, info):
        """Metric for the agent's performance."""
        reward, messages, info = output
        return reward

def create_retail_dataset(env, num_tasks=10):
    """Create dataset from retail environment tasks."""
    inputs = []
    infos = []
    
    for task_id in range(num_tasks):
        inputs.append(task_id)
        infos.append(task_id)
    
    return {'inputs': inputs, 'infos': infos}

def main():
    """Main function for ExploreAlgorithm training."""
    parser = argparse.ArgumentParser(description='Train agent using search algorithms')
    
    # Algorithm selection
    parser.add_argument('--algorithm_name', type=str, default='MinibatchAlgorithm',
                       choices=['ExploreAlgorithm', 'ExplorewithLLM', 'MinibatchAlgorithm', 'BasicSearchAlgorithm'],
                       help='Algorithm to use for training')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=50,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=50,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--train_batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    
    # ExploreAlgorithm-specific parameters
    parser.add_argument('--max_buffer_size', type=int, default=1000,
                       help='Maximum buffer size')
    parser.add_argument('--ucb_exploration_factor', type=float, default=1.0,
                       help='UCB exploration factor')
    parser.add_argument('--num_phases', type=int, default=5,
                       help='Number of training phases')
    parser.add_argument('--ucb_horizon_factor', type=int, default=10,
                       help='UCB horizon for best candidate selection')
    parser.add_argument('--num_to_sample', type=int, default=5,
                       help='Number of candidates to sample during exploration')
    
    # MinibatchAlgorithm and BasicSearchAlgorithm-specific parameters
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--num_proposals', type=int, default=3,
                       help='Number of proposals for BasicSearchAlgorithm')
    
    # ExplorewithLLM-specific parameters
    parser.add_argument('--num_LLM_samples', type=int, default=5,
                       help='Number of LLM-generated candidates during exploration')
    parser.add_argument('--llm_model', type=str, default='gemini/gemini-2.0-flash',
                       help='LLM model to use for ExplorewithLLM')
    parser.add_argument('--num_samples_in_prompt', type=int, default=5,
                       help='Number of samples to include in LLM prompt')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = RunConfig(
            model_provider="gemini",
            user_model_provider="gemini",
            model=args.model,
            user_model=args.user_model,
            num_trials=1,
            env="retail",
            agent_strategy="tool-calling",
            temperature=0.0,
            task_split="test",
            task_ids=list(range(max(args.num_train_samples, args.num_validate_samples, args.num_test_samples))),
            log_dir="results",
            max_concurrency=1,
            seed=10,
            shuffle=0,
            user_strategy="llm",
            few_shot_displays_path=None
        )
        
        # Initialize environment
        print(f"Initializing retail environment with user strategy: {config.user_strategy}")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0
        )
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = create_retail_dataset(env, num_tasks=args.num_train_samples)
        validate_dataset = create_retail_dataset(env, num_tasks=args.num_validate_samples)
        test_dataset = create_retail_dataset(env, num_tasks=args.num_test_samples)
        
        print(f"Training samples: {len(train_dataset['inputs'])}")
        print(f"Validation samples: {len(validate_dataset['inputs'])}")
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize agent
        print(f"Initializing agent with model: {config.model}")
        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        agent.set_env(env)
        
        # Initialize guide, optimizer, and logger
        guide = TeacherGuide(env, config)
        optimizer = OptoPrime(agent.parameters(), max_tokens=8000)
        optimizer.objective = OBJECTIVE
        logger = WandbLogger(project="tau-bench-retail-compare-search-algs", verbose=True, name=args.algorithm_name)
        # logger = DefaultLogger
        # Create algorithm based on selection
        print(f"Creating {args.algorithm_name}...")
        if args.algorithm_name == 'ExploreAlgorithm':
            algorithm = ExploreAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads,
                max_buffer_size=args.max_buffer_size,
                ucb_exploration_factor=args.ucb_exploration_factor
            )
        elif args.algorithm_name == 'ExplorewithLLM':
            algorithm = ExplorewithLLM(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads,
                max_buffer_size=args.max_buffer_size,
                ucb_exploration_factor=args.ucb_exploration_factor,
                llm_model=args.llm_model,
                num_samples_in_prompt=args.num_samples_in_prompt
            )
        elif args.algorithm_name == 'MinibatchAlgorithm':
            algorithm = MinibatchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_name == 'BasicSearchAlgorithm':
            algorithm = BasicSearchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        else:
            raise ValueError(f"Unknown algorithm: {args.algorithm_name}")
        
        # Training parameters
        train_params = {
            "guide": guide,
            "train_dataset": train_dataset,
            "validation_dataset": validate_dataset,
            "validate_dataset": validate_dataset,
            "test_dataset": test_dataset,
            "train_batch_size": args.train_batch_size,
            "batch_size": args.train_batch_size,
            "num_epochs": args.num_epochs,
            "num_proposals": args.num_proposals,
            "num_threads": args.num_threads,
            "eval_frequency": args.eval_frequency,
            "log_frequency": args.log_frequency,
            "num_phases": args.num_phases,
            "ucb_horizon_factor": args.ucb_horizon_factor,
            "num_to_sample": args.num_to_sample
        }
        
        # Add ExplorewithLLM-specific parameters
        if args.algorithm_name == 'ExplorewithLLM':
            train_params["num_LLM_samples"] = args.num_LLM_samples
        
        # Start training
        print(f"Starting training with {args.algorithm_name}...")
        print(f"Training batch size: {args.train_batch_size}")
        # print(f"Number of phases: {args.num_phases}")
        print(f"Number of threads: {args.num_threads}")
        
        import time
        start_time = time.time()
        algorithm.train(**train_params)
        duration = time.time() - start_time
       
           
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 