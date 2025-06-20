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
from opto.trainer.loggers import WandbLogger
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm, BasicSearchAlgorithm
from opto.trainer.algorithms.beamsearch_algorithm import BeamsearchAlgorithm, BeamsearchHistoryAlgorithm
from opto.trainer.algorithms.UCBsearch import UCBSearchAlgorithm
from opto.trainer.guide import AutoGuide

import litellm 
litellm.drop_params = True

import sys
import os
from datetime import datetime
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


def create_run_config():
    """Create a RunConfig object with default parameters from run.py"""
    return RunConfig(
        model_provider="openai",
        user_model_provider="vertex_ai",
        model="gpt-4.1-nano",
        user_model="gemini-2.0-flash",
        num_trials=1,
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        task_ids=[0],  # Select task with ID 1
        log_dir="results",
        max_concurrency=1,
        seed=10,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None
    )
@trace.model
class ToolCallingAgent(Module):
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
    def solve(self,tools_info, additional_instructions, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30):
        """Agent solves the task with the given tools_info.
        Args:
            tools_info: The tools_info to use for the task. This is a node that contains information about the tools that the agent can use.
            env: The environment to use for the task.
            task_index: The index of the task to solve.
            max_num_steps: The maximum number of steps to take. This is the maximum number of steps the agent can take to solve the task.

        Returns:
            reward: The reward for the task. 1 if the task is solved correctly, 0 otherwise.
            messages: The messages for the task. If the task is solved correctly, the message will be Correct. Otherwise, the message will be the conversation history.
            info: The info for the task. If the task is solved correctly, the info will be Correct. Otherwise, the info will be the info of the task.
        """
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
        for _ in range(max_num_steps):
            # Trim messages to prevent context window errors
            # trimmed_messages = trim_conversation_messages(messages, model=self.model)
            
            # Prepare completion arguments
            completion_kwargs = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": tools_info,
                "temperature": self.temperature,
            }
            
            # Add api_base only for local/hosted providers
            # if self.provider in ["hosted_vllm", "vllm"]:
            #     completion_kwargs["api_base"] = "http://127.0.0.1:8000/v1"
            
            res = completion(**completion_kwargs)
            next_message = res.choices[0].message.model_dump()
            cost = res._hidden_params.get("response_cost")
            if cost is not None:
                total_cost += cost
            # Skip cost tracking for vLLM/local models where cost is None
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        result = SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
        if result.reward == 1:
            return result.reward,"Correct","Correct"
        else:
            return result.reward,result.messages,result.info
        
    def forward(self, task_input):
        """Forward pass of the agent for trainer compatibility.
        
        Args:
            task_input: Task ID (integer) that will be used to reset the environment
            
        Returns:
            The result from solve method
        """
        # Get environment from global context (will be set during training)
        env = getattr(self, '_env', None)
        if env is None:
            raise ValueError("Environment not set. Call set_env() before forward pass.")
        
        return self.solve(self.tools_info, self.additional_instructions, env, task_input)
    
    def set_env(self, env):
        """Set the environment for this agent."""
        self._env = env
    
        


def message_to_action(
    message: Dict[str, Any],
) -> Action:
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
        """Initialize the teacher guide."""
        super().__init__()
        self.env = env
        self.config = config
        
    def get_feedback(self,task, output: SolveResult,info):   
        """Get feedback from the agent's output."""
        reward, messages, info = output
        if reward == 1:
            feedback = "Correct"
        else:
            # Include complete message information including tool calls
            conversation_parts = []
            for msg in messages:
                msg_str = f"{msg['role']}: {msg.get('content', '')}"
                
                # Add tool calls if present
                if 'tool_calls' in msg and msg['tool_calls']:
                    tool_calls_str = []
                    for tool_call in msg['tool_calls']:
                        if 'function' in tool_call:
                            func_name = tool_call['function'].get('name', '')
                            func_args = tool_call['function'].get('arguments', '')
                            tool_calls_str.append(f"Tool: {func_name}({func_args})")
                    if tool_calls_str:
                        msg_str += f" [Tool Calls: {'; '.join(tool_calls_str)}]"
                
                # Add tool call ID and name for tool messages
                if msg['role'] == 'tool':
                    tool_name = msg.get('name', '')
                    tool_call_id = msg.get('tool_call_id', '')
                    msg_str = f"tool ({tool_name}, ID: {tool_call_id}): {msg.get('content', '')}"
                
                conversation_parts.append(msg_str)
            
            feedback = "The agent failed to solve the task. Here is the conversation history: " + "\n".join(conversation_parts)
        return reward, feedback
        
    def metric(self,task, output: SolveResult,info):
        """Metric for the agent's performance. 1 if the task is solved correctly, 0 otherwise."""
        reward, messages, info = output
        return reward
    

def create_retail_dataset(env, num_tasks=10):
    """Create dataset from retail environment tasks.
    
    Args:
        env: The retail environment
        num_tasks: Number of tasks to include in dataset
        
    Returns:
        Dictionary with 'inputs' and 'infos' keys for trainer compatibility
    """
    inputs = []
    infos = []
    
    for task_id in range(num_tasks):
        # For trainer compatibility, we use task_id as both input and info
        # The actual task content will be handled by the environment during agent execution
        inputs.append(task_id)
        infos.append(task_id)  # Using same value since TeacherGuide doesn't need ground truth
    
    return {'inputs': inputs, 'infos': infos}

def main():
    """Main function with command line argument support for algorithm selection."""
    parser = argparse.ArgumentParser(description='Train agent using various algorithms')
    
    # Algorithm parameters
    parser.add_argument('--algorithm_type', type=str, default='beamsearch',
                       choices=['minibatch', 'basicsearch', 'beamsearch', 'beamsearchhistory', 'UCBsearch'],
                       help='Type of algorithm to use')
    
    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=1,
                       help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=1,
                       help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=1,
                       help='Number of test samples')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--num_threads', type=int, default=1,
                       help='Number of threads for parallel processing')
    parser.add_argument('--eval_frequency', type=int, default=10,
                       help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='How often to log results')
    
    # Algorithm-specific parameters
    parser.add_argument('--beam_width', type=int, default=3,
                       help='Beam width for beam search algorithms')
    parser.add_argument('--num_proposals', type=int, default=2,
                       help='Number of proposals for search algorithms')
    parser.add_argument('--max_depth', type=int, default=20,
                       help='Maximum depth for beam search algorithms')
    parser.add_argument('--max_history_size', type=int, default=12,
                       help='Maximum history size for history-based algorithms')
    
    # UCB algorithm-specific parameters
    parser.add_argument('--max_buffer_size', type=int, default=10,
                       help='Maximum buffer size for UCB algorithms')
    parser.add_argument('--ucb_exploration_factor', type=float, default=1.0,
                       help='UCB exploration factor')
    parser.add_argument('--num_search_iterations', type=int, default=100,
                       help='Number of search iterations for UCB algorithms')
    parser.add_argument('--evaluation_batch_size', type=int, default=20,
                       help='Evaluation batch size for UCB algorithms')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gpt-4.1-nano',
                       help='Model to use for the agent')
    parser.add_argument('--user_model', type=str, default='gemini-2.0-flash',
                       help='Model to use for the user')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_run_config()
        
        # Update config with command line arguments
        config.model = args.model
        config.user_model = args.user_model
        config.task_ids = list(range(max(args.num_train_samples, args.num_validate_samples, args.num_test_samples)))
        
        # Initialize environment
        print(f"Initializing retail environment with user strategy: {config.user_strategy}")
        env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0  # Will be overridden during training
        )
        
        # Create dataset from retail tasks
        print("Creating dataset from retail environment tasks...")
        train_dataset = create_retail_dataset(env, num_tasks=args.num_train_samples)
        validate_dataset = create_retail_dataset(env, num_tasks=args.num_validate_samples)
        test_dataset = create_retail_dataset(env, num_tasks=args.num_test_samples)
        
        print(f"Training samples: {len(train_dataset['inputs'])}")
        print(f"Validation samples: {len(validate_dataset['inputs'])}")
        print(f"Test samples: {len(test_dataset['inputs'])}")
        
        # Initialize agent
        print(f"Initializing {config.agent_strategy} agent with model: {config.model}")

        agent = ToolCallingAgent(
            tools_info=env.tools_info,
            wiki=env.wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature
        )
        
        # Set environment on agent for trainer compatibility
        agent.set_env(env)
        
        # Initialize guide, optimizer, and logger
        guide = TeacherGuide(env, config)
        optimizer = OptoPrime(agent.parameters(), max_tokens=40000)
        optimizer.objective = OBJECTIVE
        logger = WandbLogger(project="tau-bench-retail", verbose=True)
        
        # Create algorithm based on type
        print(f"Creating {args.algorithm_type} algorithm...")
        if args.algorithm_type == 'minibatch':
            algorithm = MinibatchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_type == 'basicsearch':
            algorithm = BasicSearchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_type == 'beamsearch':
            algorithm = BeamsearchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_type == 'beamsearchhistory':
            algorithm = BeamsearchHistoryAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads
            )
        elif args.algorithm_type == 'UCBsearch':
            algorithm = UCBSearchAlgorithm(
                agent=agent,
                optimizer=optimizer,
                logger=logger,
                num_threads=args.num_threads,
                max_buffer_size=args.max_buffer_size,
                ucb_exploration_factor=args.ucb_exploration_factor
            )
        else:
            raise ValueError(f"Unknown algorithm type: {args.algorithm_type}")
        
        # Prepare training parameters
        train_params = {
            "guide": guide,
            "train_dataset": train_dataset,
            "num_epochs": args.num_epochs,
            "num_threads": args.num_threads,
            "batch_size": args.batch_size,
            "test_dataset": test_dataset,
            "eval_frequency": args.eval_frequency,
            "log_frequency": args.log_frequency,
            "ensure_improvement": False
        }
        
        # Add algorithm-specific parameters
        if args.algorithm_type in ['beamsearch', 'beamsearchhistory']:
            train_params.update({
                "beam_width": args.beam_width,
                "num_proposals": args.num_proposals,
                "max_depth": args.max_depth,
                "validation_dataset_size": args.num_validate_samples
            })
            
            if args.algorithm_type == 'beamsearchhistory':
                train_params["max_history_size"] = args.max_history_size
                
        elif args.algorithm_type == 'basicsearch':
            train_params["num_proposals"] = args.num_proposals
        
        elif args.algorithm_type == 'UCBsearch':
            train_params.update({
                "num_search_iterations": args.num_search_iterations,
                "train_batch_size": args.batch_size,
                "evaluation_batch_size": args.evaluation_batch_size,
                "validation_dataset_size": args.num_validate_samples
            })
        
        # Start training
        print(f"Starting training with {args.algorithm_type} algorithm...")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of epochs: {args.num_epochs}")
        print(f"Number of threads: {args.num_threads}")
        
        import time
        start_time = time.time()
        metrics, test_score = algorithm.train(**train_params)
        duration = time.time() - start_time
        
        print(f"\nTraining completed in {duration:.2f} seconds")
        print(f"Final score: {test_score:.4f}")
                
        # avg_train_score = sum(train_scores) / len(train_scores)
        # print(f"Average training score: {avg_train_score:.4f}")
           
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 