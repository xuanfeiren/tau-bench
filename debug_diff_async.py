# Repeat the agent on the first task multiple times until it succeeds
from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from tau_bench.envs.user import UserStrategy

import opto 
from opto import trace
from opto.optimizers import OptoPrime 
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module 
import numpy as np
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
from opto.trainer.utils import async_run
from opto.optimizers.utils import print_color
from opto.trainer.algorithms.basic_algorithms import MinibatchAlgorithm,  batchify
import litellm 
litellm.drop_params = True

import sys
import os
from datetime import datetime

def evaluate(agent, guide, inputs, infos, min_score=None, num_threads=None, description=None):
    """ Evaluate the agent on the inputs and return the scores

    Args:
        agent: The agent to evaluate
        guide: The guide to use for evaluation
        inputs: List of inputs to evaluate on
        infos: List of additional information for each input
        min_score: Minimum score to return when an exception occurs
        num_threads: Maximum number of threads to use for parallel evaluation
        description: Description to display in the progress bar
    """

    def evaluate_single(i):
        try:
            """create a new env for each thread"""
            env = get_env(
            env_name="retail",
            user_strategy="llm",
            user_model="gemini-2.0-flash",
            user_provider="vertex_ai",
            task_split="test",
            task_index=0  # Will be overridden during training
        )
            agent.set_env(env)

            output = agent(inputs[i]).data
            score = guide.metric(inputs[i], output, infos[i])
        except:
            score = min_score
        return score

    N = len(inputs)
    assert len(inputs) == len(infos), "Inputs and infos must have the same length"
    # Use asyncio if num_threads is not None and > 1
    use_asyncio = num_threads is not None and num_threads > 1
    if use_asyncio:
        # Use provided description or generate a default one
        eval_description = description or f"Evaluating {N} examples"
        scores = async_run([evaluate_single] * N, [(i,) for i in range(N)],
                          max_workers=num_threads,
                          description=eval_description) # list of tuples
    else:
        scores = [evaluate_single(i) for i in range(N)]
    return scores

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
config = create_run_config()
env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            user_provider=config.user_model_provider,
            task_split=config.task_split,
            task_index=0  # Will be overridden during training
        )
train_dataset = create_retail_dataset(env, num_tasks=10)
validate_dataset = create_retail_dataset(env, num_tasks=10)
test_dataset = create_retail_dataset(env, num_tasks=10)

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

print("Evaluating initial parameters on test set, with 1 thread")

# for _ in range(5):
#     initial_test_scores = evaluate(
#         agent,
#         guide,
#         test_dataset['inputs'],
#         test_dataset['infos'],
#         min_score=0,
#         num_threads=1,
#         description="Evaluating initial parameters on test set"
#     )
#     initial_test_score = np.mean(initial_test_scores) if all([s is not None for s in initial_test_scores]) else -np.inf
#     print_color(f"Initial test score: {initial_test_score:.4f}", 'yellow')


print("Evaluating initial parameters on test set, with 10 threads")

for _ in range(1):
    initial_test_scores = evaluate(
        agent,
        guide,
        test_dataset['inputs'],
        test_dataset['infos'],
        min_score=0,
        num_threads=10,
        description="Evaluating initial parameters on test set"
    )
    initial_test_score = np.mean(initial_test_scores) if all([s is not None for s in initial_test_scores]) else -np.inf
    print_color(f"Initial test score: {initial_test_score:.4f}", 'yellow')
