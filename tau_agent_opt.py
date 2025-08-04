# Repeat the agent on the first task multiple times until it succeeds
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
from tau_bench.retry_utils import auto_retry_with_exponential_backoff

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
from opto.trainer.algorithms.baselines import  MinibatchwithValidation, BasicSearchAlgorithm, IslandSearchAlgorithm, MinibatchAlgorithm, DetectCorrelation
from opto.trainer.guide import AutoGuide

# Import the agent from separate module to avoid pickle issues
from agents.tool_calling_agent import ToolCallingAgent, message_to_action

import litellm 
litellm.drop_params = True
litellm.suppress_debug_info = True
import sys
import os
from datetime import datetime
# provider = "vertex_ai"
provider = "gemini"
os.environ["TRACE_LITELLM_MODEL"] = f"{provider}/gemini-2.0-flash"


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

class TeacherGuide(AutoGuide):
    """Guide that extract reward and feedback from the agent's output."""
    def __init__(self, env: Env, config: RunConfig):
        super().__init__()
        self.env = env
        self.config = config
        
    def get_feedback(self, task, output: SolveResult, info):   
        """Get feedback from the agent's output."""
        reward, messages, info = output
        if info == "BadRequest":
            return 0, "BadRequestError. Please adjust the tool information to the correct form."
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
                       choices=['ExploreAlgorithm', 'ExplorewithLLM', 'MinibatchAlgorithm', 'BasicSearchAlgorithm', 
                               'MinibatchwithValidation', 'IslandSearchAlgorithm', 'DetectCorrelation'],
                       help='Algorithm to use for training')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=50,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=50,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--train_batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    
    # ExploreAlgorithm-specific parameters
    parser.add_argument('--max_buffer_size', type=int, default=5,
                       help='Maximum buffer size')
    parser.add_argument('--ucb_exploration_factor', type=float, default=0.1,
                       help='UCB exploration factor')
    parser.add_argument('--num_phases', type=int, default=5,
                       help='Number of training phases')
    parser.add_argument('--ucb_horizon', type=int, default=50,
                       help='UCB horizon for best candidate selection')
    parser.add_argument('--num_to_sample', type=int, default=5,
                       help='Number of candidates to sample during exploration')
    parser.add_argument('--evaluation_batch_size', type=int, default=20,
                       help='Evaluation batch size')
    parser.add_argument('--num_eval_samples', type=int, default=5,
                       help='Number of samples to evaluate for each input')
    
    # MinibatchAlgorithm and BasicSearchAlgorithm-specific parameters
    parser.add_argument('--num_epochs', type=int, default=20,
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
    
    # IslandSearchAlgorithm-specific parameters
    parser.add_argument('--num_islands', type=int, default=4,
                       help='Number of islands for IslandSearchAlgorithm')
    parser.add_argument('--discard_frequency', type=int, default=2,
                       help='Frequency of discarding islands with low scores')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = RunConfig(
            model_provider=provider,
            user_model_provider=provider,
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
        logger = WandbLogger(project="tau-bench-retail-compare-search-algs", verbose=True, name=args.run_name)
        
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
        elif args.algorithm_name == 'MinibatchwithValidation':
            algorithm = MinibatchwithValidation(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_name == 'IslandSearchAlgorithm':
            algorithm = IslandSearchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_islands=args.num_islands,
                num_threads=args.num_threads,
                llm_model=args.llm_model,
                num_samples_in_prompt=args.num_samples_in_prompt,
                num_LLM_samples=args.num_LLM_samples
            )
        elif args.algorithm_name == 'DetectCorrelation':
            algorithm = DetectCorrelation(
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
            "ucb_horizon": args.ucb_horizon,
            "num_to_sample": args.num_to_sample,
            "evaluation_batch_size": args.evaluation_batch_size,
            "discard_frequency": args.discard_frequency,  # For IslandSearchAlgorithm
            "num_eval_samples": args.num_eval_samples  # For MinibatchwithValidation and others
        }
        
        # Add ExplorewithLLM and IslandSearchAlgorithm-specific parameters
        if args.algorithm_name in ['ExplorewithLLM', 'IslandSearchAlgorithm']:
            train_params["num_LLM_samples"] = args.num_LLM_samples
        
        # Start training
        print(f"Starting training with {args.algorithm_name}...")
        print(f"Training batch size: {args.train_batch_size}")
        # print(f"Number of phases: {args.num_phases}")
        print(f"Number of threads: {args.num_threads}")
        
        # Print algorithm-specific parameters
        if args.algorithm_name == 'IslandSearchAlgorithm':
            print(f"Number of islands: {args.num_islands}")
            print(f"LLM model: {args.llm_model}")
            print(f"Number of LLM samples: {args.num_LLM_samples}")
            print(f"Discard frequency: {args.discard_frequency}")
        elif args.algorithm_name in ['ExplorewithLLM']:
            print(f"LLM model: {args.llm_model}")
            print(f"Number of LLM samples: {args.num_LLM_samples}")
        elif args.algorithm_name in ['BasicSearchAlgorithm']:
            print(f"Number of proposals: {args.num_proposals}")
        elif args.algorithm_name in ['MinibatchwithValidation']:
            print(f"Number of epochs: {args.num_epochs}")
            print(f"Number of proposals: {args.num_proposals}")
        elif args.algorithm_name == 'DetectCorrelation':
            print(f"Number of epochs: {args.num_epochs}")
            print(f"Training batch size: {args.train_batch_size}")
        
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