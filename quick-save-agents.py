# quick-save-agents.py


from agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
import litellm 
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
from opto.trainer.guide import AutoGuide
from tau_agent_opt import create_retail_dataset,TeacherGuide
from opto.trainer.algorithms.baselines import MinibatchAlgorithm
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate
import numpy as np
from opto.trainer.algorithms.BAI_algorithms import EvenlySplitAlgorithm, UCBAlgorithm, LLMSelectorAlgorithm
provider = "gemini"



def main():
    """Main function for LearnFromSuccessAlgorithm training."""
    parser = argparse.ArgumentParser(description='Best Candidate Identification')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=50,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=50,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--num_threads', type=int, default=20,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=1,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    parser.add_argument('--save_frequency', type=int, default=None,
                       help='How often to save the agent')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    parser.add_argument('--run_name', type=str, default='debug',
                       help='Name of the run')

    # Algorithm choice
    parser.add_argument('--bai_algo', type=str, default='even', choices=['even', 'ucb', 'llm'],
                        help='Which BAI algorithm to run: even (EvenlySplit), ucb (UCBAlgorithm), llm (LLMSelectorAlgorithm)')
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
        base_agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        base_agent.set_env(env)
        guide = TeacherGuide(env, config)
        logger = WandbLogger(project="tau-bench-best-candidate-identification", verbose=True, name=args.run_name)
        
        # Load 10 agents from checkpoints folder
        agents = []
        agent_to_save = ToolCallingAgent(tools_info=env.tools_info, wiki=env.wiki, model=config.model, provider=config.model_provider, temperature=config.temperature)
        agent_to_save.additional_instructions._set(""" Here are the additional instructions to help the agent solve the task:

- When a user wants to exchange or modify an item but does not have the item ID, use 'get_product_details' to find available items and their IDs. List all available options (e.g., color, size, capacity, power source, brightness) to the user, and ask the user to pick which item and options they want.

- When a user wants to return or cancel multiple items, and the user only provides item names, use 'get_product_details' to find the item IDs. For each item, present the options to the user and ask them to confirm which specific item they want to return or cancel before proceeding.

- When a user wants to return or cancel multiple items, ask for all order IDs and item IDs first before calling any tool function. Avoid unnecessary back-and-forth to improve efficiency.

- Before calling 'cancel_pending_order', 'return_delivered_order_items', 'exchange_delivered_order_items', or 'modify_pending_order_items', double check that you have the right order ID. Ensure you check the current order status by calling 'get_order_details' first.

- Minimize explicit user confirmation steps. Only ask for user confirmation once you have gathered all the necessary information and are about to take a consequential action.

- If a user says they want to cancel a charger for the tablet they lost, first check if that charger is part of the tablet order and try to cancel that order. Prioritize solving the primary issue first, as that will solve the downstream issues.

- Do not ask the user for the order date; instead, use get_user_details to retrieve the order history.

- If the user wants to exchange for a brighter or bigger item, use the `get_product_details` tool. Find the items that have the desired properties, and ask the user to pick items that have the desired properties and options. Provide available options before asking them to choose. For example, for desk lamps, ask about power source first, then brightness.
        """)
        agent_to_save.set_env(env)
        agent_to_save.save(f"checkpoints/trained_agent_1.pkl")
        return 
    except Exception as e:
        print(f"Error: {e}")
        return
        

if __name__ == "__main__":
    main() 












