# Train agent using PrioritySearch algorithm on tau-bench
# set np and torch seeds
import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy
from tau_bench.retry_utils import auto_retry_with_exponential_backoff

import opto 
from opto import trace
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 
from typing import Union, Optional

# Copyright Sierra

import json
from litellm import completion
import argparse

from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.model_utils.model.utils import trim_conversation_messages
from opto.trainer.loggers import WandbLogger, DefaultLogger
# from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_with_Regressor, PrioritySearch_with_Regressor_and_Generator
from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_with_Regressor_and_Generator
# from opto.features.priority_search.expansive_priority_search import ExpansivePrioritySearch as PrioritySearch_with_Regressor
# from opto.features.priority_search.expansive_priority_search import ExpansivePrioritySearch_highscore as PrioritySearch_with_Regressor
from opto.features.priority_search.exhausted_priority_search import ExhaustedPrioritySearch_v2 as PrioritySearch_with_Regressor




# from opto.features.priority_search.priority_search_with_regressor import PrioritySearch_RG_RejectionSampling as PrioritySearch_with_Regressor_and_Generator
from opto.trainer.guide import Guide
from agents.tool_calling_agent import ToolCallingAgent_v2 as ToolCallingAgent
# Import the agent from separate module to avoid pickle issues-
# from agents.tool_calling_agent import TrainedToolCallingAgent as ToolCallingAgent

import litellm 
litellm.drop_params = True
litellm.suppress_debug_info = True
import sys
import os
from datetime import datetime
# provider = "vertex_ai"
provider = "gemini"
os.environ["TRACE_LITELLM_MODEL"] = f"{provider}/gemini-2.0-flash"


# litellm._turn_on_debug()
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

def get_trajectory_from_output(output: SolveResult):
    """Get trajectory from the agent's output."""
    reward, messages, info = output
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
    
    return conversation_parts

class TeacherGuide(Guide):
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
    """Main function for PrioritySearch training."""
    parser = argparse.ArgumentParser(description='Train agent using PrioritySearch algorithm')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=10,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=1,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--num_batches', type=int, default=1,
                       help='Number of batches to use from the dataset in each iteration')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--num_steps', type=int, default=5,
                       help='Number of training steps')
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--test_frequency', type=int, default=None,
                       help='How often to run evaluation (test_frequency)')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    parser.add_argument('--save_frequency', type=int, default=None,
                       help='How often to save the agent')
    parser.add_argument('--save_path', type=str, default='checkpoints/priority_search_agent.pkl',
                       help='Path to save the agent')
    parser.add_argument('--num_eval_samples', type=int, default=1,
                       help='Number of times to evaluate each input')
    
    # PrioritySearch-specific parameters
    parser.add_argument('--num_candidates', type=int, default=2,
                       help='Number of candidates to propose for exploration')
    parser.add_argument('--num_proposals', type=int, default=1,
                       help='Number of proposals to generate per optimizer')
    parser.add_argument('--validate_exploration_candidates', action='store_true', default=False,
                       help='Whether to validate the proposed parameters for exploration')
    parser.add_argument('--use_best_candidate_to_explore', action='store_true', default=False,
                       help='Whether to use the best candidate as part of the exploration candidates')
    parser.add_argument('--memory_size', type=int, default=None,
                       help='Size of the heap memory to store the candidates; if None, no limit is set')
    parser.add_argument('--score_function', type=str, default='mean',
                       choices=['mean', 'ucb', 'time'],
                       help='Function to compute the score for the candidates')
    parser.add_argument('--long_term_memory_size', type=int, default=None,
                       help='Size of the long-term memory to store the candidates; if None, no limit is set')
    parser.add_argument('--ucb_exploration_constant', type=float, default=1.0,
                       help='Exploration constant for UCB score function')
    parser.add_argument('--score_range_min', type=float, default=0.0,
                       help='Minimum score for score range (used with UCB)')
    parser.add_argument('--score_range_max', type=float, default=1.0,
                       help='Maximum score for score range (used with UCB)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--additional_instructions_index', type=int, default=0,
                       help='Index of the additional instructions to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--project_name', type=str, default='tau-bench-priority-search',
                       help='Name of the project')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Whether to print verbose output')
    parser.add_argument('--memory_update_frequency', type=int, default=2,
                       help='Duration of the short-term memory')
    parser.add_argument('--use_regressor', action='store_true', default=False,
                       help='Whether to use the regressor')
    parser.add_argument('--regressor_type', type=str, default='logistic', choices=['logistic', 'linear', 'linear_ucb', 'llm', 'pretrained_linear', 'pretrained_logistic'],
                       help='Type of the regressor')
    parser.add_argument('--regressor_model_name', type=str, default='gemini/gemini-2.0-flash',
                       help='Model name for the regressor')
    parser.add_argument('--regressor_alpha', type=float, default=0.1,
                       help='UCB exploration parameter for the regressor')
    parser.add_argument('--regressor_regularization_strength', type=float, default=0.0001,
                       help='Regularization strength for the regressor')
    parser.add_argument('--regressor_transformation_exploration_factor', type=float, default=0.0,
                       help='Transformation exploration factor for the regressor')
    parser.add_argument('--regressor_projection_dim', type=int, default=None,
                       help='Projection dimension for the regressor')
    parser.add_argument('--regressor_rich_text', action='store_true', default=False,
                       help='Whether to use rich text with problem definition for embeddings')
    parser.add_argument('--optoprime_version', type=str, default='v2', choices=['v1', 'v2'],
                       help='Optimizer to use')
    parser.add_argument('--use_validation', action='store_true', default=False,
                       help='Whether to use validation, only matters in use_regressor version')
    
    # Generator-specific parameters
    parser.add_argument('--use_generator', action='store_true', default=False,
                       help='Whether to use the LLM generator for candidate generation')
    parser.add_argument('--generator_frequency', type=int, default=5,
                       help='Frequency of generating new candidates using LLM generator')
    parser.add_argument('--generator_attempts', type=int, default=50,
                       help='Number of attempts to generate new candidates using LLM generator')
    parser.add_argument('--generator_patience', type=int, default=3,
                       help='Number of attempts to generate new candidates using LLM generator')
    parser.add_argument('--num_generator_candidates', type=int, default=5,
                       help='Number of candidates to generate using LLM generator')
    parser.add_argument('--generator_model_name', type=str, default='gemini/gemini-2.0-flash',
                       help='Model name for the LLM generator')
    parser.add_argument('--generator_temperature', type=float, default=0.6,
                       help='Temperature for the LLM generator')
    parser.add_argument('--generator_verbose', action='store_true', default=False,
                       help='Whether to enable verbose output for the generator')
    parser.add_argument('--ablation',action='store_true', default=False,
                       help='Whether to run ablation study')
    parser.add_argument('--ucb_exploration',action='store_true', default=False,
                       help='UCB exploration')
    parser.add_argument('--epsnetPS',action='store_true', default=False,
                       help='Whether to run epsnetPS')
    
    args = parser.parse_args()

    if args.ablation:
        # ucb_exploration or epsnetPS, cannot be used together
        assert not (args.ucb_exploration and args.epsnetPS), "ucb_exploration and epsnetPS cannot be used together"
        if args.ucb_exploration:
            from opto.features.priority_search.priority_search_ablation import PrioritySearchUCBExploration as PrioritySearch
        elif args.epsnetPS:
            from opto.features.priority_search.priority_search_ablation import EpsilonNetPS as PrioritySearch
        else:
            from opto.features.priority_search.priority_search_ablation import PrioritySearch as PrioritySearch
    else:
        from opto.features.priority_search.priority_search import PrioritySearch
    

    
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
            temperature=config.temperature,
            additional_instructions_index=args.additional_instructions_index
        )
        agent.set_env(env)
        
        # Initialize guide, optimizer, and logger
        guide = TeacherGuide(env, config)
        if args.optoprime_version == 'v1':
            from opto.optimizers import OptoPrime
            optimizer = OptoPrime(agent.parameters(), max_tokens=8000)
        else:
            from opto.optimizers import OptoPrimeV2
            optimizer = OptoPrimeV2(agent.parameters(), max_tokens=25000,initial_var_char_limit=10000)
        optimizer.objective = OBJECTIVE
        
        # Prepare configuration for logging (excluding project_name and run_name)
        config_dict = {
            'num_train_samples': args.num_train_samples,
            'num_validate_samples': args.num_validate_samples,
            'num_test_samples': args.num_test_samples,
            'batch_size': args.batch_size,
            'num_batches': args.num_batches,
            'num_epochs': args.num_epochs,
            'num_steps': args.num_steps,
            'memory_update_frequency': args.memory_update_frequency,
            'num_threads': args.num_threads,
            'test_frequency': args.test_frequency,
            'log_frequency': args.log_frequency,
            'save_frequency': args.save_frequency,
            'save_path': args.save_path,
            'num_eval_samples': args.num_eval_samples,
            'num_candidates': args.num_candidates,
            'num_proposals': args.num_proposals,
            'validate_exploration_candidates': args.validate_exploration_candidates,
            'use_best_candidate_to_explore': args.use_best_candidate_to_explore,
            'memory_size': args.memory_size,
            'score_function': args.score_function,
            'ucb_exploration_constant': args.ucb_exploration_constant,
            'score_range_min': args.score_range_min,
            'score_range_max': args.score_range_max,
            'model': args.model,
            'user_model': args.user_model,
            'verbose': args.verbose,
            'use_validation': args.use_validation,
            'regressor_rich_text': args.regressor_rich_text,
            'use_generator': args.use_generator,
            'generator_frequency': args.generator_frequency,
            'generator_attempts': args.generator_attempts,
            'generator_patience': args.generator_patience,
            'num_generator_candidates': args.num_generator_candidates,
            'generator_model_name': args.generator_model_name,
            'generator_temperature': args.generator_temperature,
            'generator_verbose': args.generator_verbose,
        }
        
        logger = WandbLogger(project=args.project_name, verbose=True, name=args.run_name, config=config_dict)
        
        # Create PrioritySearch algorithm
        print("Creating PrioritySearch algorithm...")
        if args.use_regressor and args.use_generator:
            print("Using PrioritySearch with Regressor and Generator")
            algorithm = PrioritySearch_with_Regressor_and_Generator(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads,
                generator_model_name=args.generator_model_name,
                generator_temperature=args.generator_temperature,
                generator_verbose=args.generator_verbose
            )
        elif args.use_regressor:
            print("Using PrioritySearch with Regressor")
            algorithm = PrioritySearch_with_Regressor(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        else:
            print("Using basic PrioritySearch")
            algorithm = PrioritySearch(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        
        # Set score range for UCB
        score_range = (args.score_range_min, args.score_range_max) if args.score_function == 'ucb' else None
        
        # Training parameters for PrioritySearch
        train_params = {
            "guide": guide,
            "train_dataset": train_dataset,
            "validate_dataset": validate_dataset,
            "test_dataset": test_dataset,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "score_range": score_range,
            "num_epochs": args.num_epochs,
            "num_steps": args.num_steps,
            "long_term_memory_size": args.long_term_memory_size,
            "memory_update_frequency": args.memory_update_frequency,
            "num_threads": args.num_threads,
            "verbose": args.verbose,
            "test_frequency": args.test_frequency,
            "num_eval_samples": args.num_eval_samples,
            "num_test_samples": args.num_eval_samples,
            "log_frequency": args.log_frequency,
            "save_frequency": args.save_frequency,
            "save_path": args.save_path,
            # PrioritySearch specific parameters
            "num_candidates": args.num_candidates,
            "num_proposals": args.num_proposals,
            "validate_exploration_candidates": args.validate_exploration_candidates,
            "use_best_candidate_to_explore": args.use_best_candidate_to_explore,
            "memory_size": args.memory_size,
            "score_function": args.score_function,
            "ucb_exploration_constant": args.ucb_exploration_constant,
            "use_validation": args.use_validation,
            "regressor_type": args.regressor_type,
            "regressor_alpha": args.regressor_alpha,
            "regressor_transformation_exploration_factor": args.regressor_transformation_exploration_factor,
            "regressor_projection_dim": args.regressor_projection_dim,
            "regressor_regularization_strength": args.regressor_regularization_strength,
            "regressor_rich_text": args.regressor_rich_text,
            # Generator-specific parameters
            "generator_frequency": args.generator_frequency,
            "generator_attempts": args.generator_attempts,
            "generator_patience": args.generator_patience,
            "num_generator_candidates": args.num_generator_candidates,
        }
        
        # Start training
        print("Starting training with PrioritySearch...")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of batches: {args.num_batches}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Number of steps: {args.num_steps}")
        print(f"Number of threads: {args.num_threads}")
        print(f"Number of candidates: {args.num_candidates}")
        print(f"Number of proposals: {args.num_proposals}")
        print(f"Score function: {args.score_function}")
        print(f"UCB exploration constant: {args.ucb_exploration_constant}")
        print(f"Memory size: {args.memory_size}")
        print(f"Validate exploration candidates: {args.validate_exploration_candidates}")
        print(f"Use best candidate to explore: {args.use_best_candidate_to_explore}")
        print(f"Use validation: {args.use_validation}")
        print(f"Regressor type: {args.regressor_type}")
        print(f"Regressor alpha: {args.regressor_alpha}")
        print(f"Regressor regularization strength: {args.regressor_regularization_strength}")
        print(f"Use generator: {args.use_generator}")
        if args.use_generator:
            print(f"Generator frequency: {args.generator_frequency}")
            print(f"Generator attempts: {args.generator_attempts}")
            print(f"Generator patience: {args.generator_patience}")
            print(f"Number of generator candidates: {args.num_generator_candidates}")
            print(f"Generator model: {args.generator_model_name}")
            print(f"Generator temperature: {args.generator_temperature}")
            print(f"Generator verbose: {args.generator_verbose}")
        import time
        start_time = time.time()
        algorithm.train(**train_params)
        duration = time.time() - start_time
        
        print(f"Training completed in {duration:.2f} seconds")
           
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
